// Tiny C shim over the llama.cpp C API so Rust never has to mirror llama.h's
// by-value structs. One shim_ctx = one llama_context with a unified KV cache and
// n_seq slots. seq 0 holds the shared parent prompt; children get a seq id and
// llama_memory_seq_cp(0 -> child) (metadata only under kv_unified: zero-copy).
// NOT thread-safe: the Rust side serialises calls with a Mutex.
#define _POSIX_C_SOURCE 200809L
#include "llama.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Children interleave in ONE context, so a child's logits are only valid right
// after its own decode. Invariant: each child keeps one `pending` token that has
// been chosen but not yet decoded; shim_child_next() decodes it and samples the
// next token inside the same call, so logits are always fresh.
typedef struct {
    struct llama_sampler *grammar;
    int pos;              // position of `pending`
    llama_token pending;
    int active;
} child_t;

typedef struct {
    struct llama_model *model;
    struct llama_context *ctx;
    const struct llama_vocab *vocab;
    llama_memory_t mem;
    char *grammar;
    int n_parent;
    int n_seq;
    child_t *kids;
    struct llama_batch batch;
    int n_batch;
    llama_token_data *cur;
    long full_masks, samples;
} shim_ctx;

static void quiet(enum ggml_log_level l, const char *t, void *u) { (void)l; (void)t; (void)u; }

shim_ctx *shim_new(const char *model_path, const char *grammar, int n_ctx, int n_seq, int threads) {
    llama_log_set(quiet, NULL);
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    struct llama_model *m = llama_model_load_from_file(model_path, mp);
    if (!m) return NULL;
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx; cp.n_batch = n_ctx; cp.n_ubatch = 512;
    cp.n_seq_max = n_seq; cp.kv_unified = true;
    cp.n_threads = threads; cp.n_threads_batch = threads;
    struct llama_context *c = llama_init_from_model(m, cp);
    if (!c) { llama_model_free(m); return NULL; }
    shim_ctx *s = calloc(1, sizeof *s);
    s->model = m; s->ctx = c; s->vocab = llama_model_get_vocab(m); s->mem = llama_get_memory(c);
    s->grammar = strdup(grammar); s->n_seq = n_seq; s->kids = calloc(n_seq, sizeof(child_t));
    s->n_batch = n_ctx; s->batch = llama_batch_init(n_ctx, 0, 1);
    s->cur = malloc(sizeof(llama_token_data) * llama_vocab_n_tokens(s->vocab));
    return s;
}

void shim_free(shim_ctx *s) {
    if (!s) return;
    for (int i = 0; i < s->n_seq; i++) if (s->kids[i].grammar) llama_sampler_free(s->kids[i].grammar);
    llama_batch_free(s->batch);
    llama_free(s->ctx); llama_model_free(s->model);
    free(s->grammar); free(s->kids); free(s->cur); free(s);
}

static int tokenize(shim_ctx *s, const char *text, int add_special, llama_token **out) {
    int n = -llama_tokenize(s->vocab, text, (int)strlen(text), NULL, 0, add_special, 0);
    *out = malloc(sizeof(llama_token) * (n + 1));
    if (llama_tokenize(s->vocab, text, (int)strlen(text), *out, n, add_special, 0) != n) return -1;
    return n;
}

// decode toks on seq from pos; logits on last; returns 0 ok
static int decode(shim_ctx *s, const llama_token *t, int n, int pos, int seq) {
    for (int off = 0; off < n; off += s->n_batch) {
        int m = n - off < s->n_batch ? n - off : s->n_batch;
        s->batch.n_tokens = 0;
        for (int i = 0; i < m; i++) {
            int k = s->batch.n_tokens++;
            s->batch.token[k] = t[off + i]; s->batch.pos[k] = pos + off + i;
            s->batch.n_seq_id[k] = 1; s->batch.seq_id[k][0] = seq;
            s->batch.logits[k] = (off + i == n - 1);
        }
        if (llama_decode(s->ctx, s->batch) != 0) return -1;
    }
    return 0;
}

// Prefill the shared parent prompt on seq 0. Returns token count or -1.
int shim_set_parent(shim_ctx *s, const char *text) {
    llama_memory_clear(s->mem, true);
    llama_token *t; int n = tokenize(s, text, 1, &t);
    if (n < 0) { free(t); return -1; }
    int r = n > 0 ? decode(s, t, n, 0, 0) : 0;
    free(t);
    s->n_parent = n;
    return r == 0 ? n : -1;
}

// Start a child on seq (1..n_seq-1): seq_cp parent cells, decode suffix except
// its last token (kept as pending). If use_parent == 0 the child prefills
// parent+suffix itself (cold path). Returns tokens prefilled (incl. pending), -1 on error.
int shim_child_begin(shim_ctx *s, int seq, const char *parent, const char *suffix, int use_parent) {
    if (seq <= 0 || seq >= s->n_seq) return -1;
    child_t *k = &s->kids[seq];
    llama_memory_seq_rm(s->mem, seq, -1, -1);
    if (k->grammar) llama_sampler_free(k->grammar);
    k->grammar = llama_sampler_init_grammar(s->vocab, s->grammar, "root");
    if (!k->grammar) return -1;
    llama_token *t; int n, p0;
    if (use_parent) {
        llama_memory_seq_cp(s->mem, 0, seq, -1, -1);   // zero-copy under kv_unified
        n = tokenize(s, suffix, 0, &t); p0 = s->n_parent;
    } else {
        size_t L = strlen(parent) + strlen(suffix) + 1;
        char *full = malloc(L); strcpy(full, parent); strcat(full, suffix);
        n = tokenize(s, full, 1, &t); p0 = 0; free(full);
    }
    if (n <= 0) { free(t); return -1; }
    if (n > 1 && decode(s, t, n - 1, p0, seq) != 0) { free(t); return -1; }
    k->pending = t[n - 1]; k->pos = p0 + n - 1; k->active = 1;
    free(t);
    return n;
}

// Decode the child's pending token, sample the next one under its grammar
// (greedy; fast path tests the argmax first and masks the full vocabulary only
// if the grammar rejects it), store it as pending and write its text to buf.
// Returns piece length, -1 on end-of-generation, -2 on error.
int shim_child_next(shim_ctx *s, int seq, char *buf, int cap) {
    if (seq <= 0 || seq >= s->n_seq) return -2;
    child_t *k = &s->kids[seq];
    if (!k->active) return -1;
    if (decode(s, &k->pending, 1, k->pos, seq) != 0) return -2;
    k->pos++;
    const float *lg = llama_get_logits_ith(s->ctx, -1);
    int nv = llama_vocab_n_tokens(s->vocab), best = 0;
    for (int i = 1; i < nv; i++) if (lg[i] > lg[best]) best = i;
    s->samples++;
    llama_token_data one = { best, lg[best], 0.0f };
    llama_token_data_array a1 = { &one, 1, -1, false };
    llama_sampler_apply(k->grammar, &a1);
    llama_token tok = best;
    if (!isfinite(a1.data[0].logit)) {
        s->full_masks++;
        for (int i = 0; i < nv; i++) { s->cur[i].id = i; s->cur[i].logit = lg[i]; s->cur[i].p = 0; }
        llama_token_data_array a = { s->cur, (size_t)nv, -1, false };
        llama_sampler_apply(k->grammar, &a);
        size_t bi = 0;
        for (size_t i = 1; i < a.size; i++) if (a.data[i].logit > a.data[bi].logit) bi = i;
        if (!isfinite(a.data[bi].logit)) { k->active = 0; return -1; }
        tok = a.data[bi].id;
    }
    llama_sampler_accept(k->grammar, tok);
    if (llama_vocab_is_eog(s->vocab, tok)) { k->active = 0; return -1; }
    k->pending = tok;
    int n = llama_token_to_piece(s->vocab, tok, buf, cap - 1, 0, 1);
    if (n < 0) n = 0;
    buf[n] = 0;
    return n;
}

void shim_child_end(shim_ctx *s, int seq) {
    if (seq <= 0 || seq >= s->n_seq) return;
    llama_memory_seq_rm(s->mem, seq, -1, -1);
    if (s->kids[seq].grammar) { llama_sampler_free(s->kids[seq].grammar); s->kids[seq].grammar = NULL; }
    s->kids[seq].active = 0;
}

int shim_n_parent(shim_ctx *s) { return s->n_parent; }
long shim_full_masks(shim_ctx *s) { return s->full_masks; }
long shim_samples(shim_ctx *s) { return s->samples; }
