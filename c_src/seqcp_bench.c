#define _POSIX_C_SOURCE 200809L
// seqcp_bench: shared-prefix micro-agent decoding with the llama.cpp C API.
//
// One llama_context, kv_unified = true, n_seq_max = N+1.
//   seq 0      : parent prompt, prefilled ONCE
//   seq 1..N   : children. llama_memory_seq_cp(mem, 0, i, -1, -1) makes every
//                parent KV cell also belong to seq i. With a unified KV buffer this
//                is a metadata-only operation (no tensor data copied; see
//                llama_kv_cache::seq_cp "same stream" branch) -> true zero-copy
//                sharing of the prefix inside one process/context.
// Each child then decodes only its own suffix and generates under a GBNF grammar
// sampler (greedy => deterministic payload), stopping at the first "\n".
//
// Compared against "cold": every child prefills parent+suffix itself.
// Prints one JSON object on stdout. Everything reported is measured.
//
// Build: see c_src/Makefile (links the prebuilt libllama.so, does not rebuild it).

#include "llama.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define MAXC 64

static void quiet_log(enum ggml_log_level l, const char *t, void *u) { (void)l; (void)t; (void)u; }

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc(n + 1);
    if (fread(buf, 1, n, f) != (size_t)n) { fprintf(stderr, "read %s failed\n", path); exit(2); }
    buf[n] = 0;
    fclose(f);
    return buf;
}

static void json_str(FILE *o, const char *s) {
    fputc('"', o);
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') { fputc('\\', o); fputc(c, o); }
        else if (c == '\n') fputs("\\n", o);
        else if (c == '\t') fputs("\\t", o);
        else if (c < 0x20) fprintf(o, "\\u%04x", c);
        else fputc(c, o);
    }
    fputc('"', o);
}

typedef struct {
    llama_token *toks; int n;
} tokvec;

static tokvec tokenize(const struct llama_vocab *v, const char *text, int add_special) {
    int n = -llama_tokenize(v, text, strlen(text), NULL, 0, add_special, 0);
    tokvec t = { malloc(sizeof(llama_token) * (n + 1)), n };
    if (llama_tokenize(v, text, strlen(text), t.toks, n, add_special, 0) != n) { fprintf(stderr, "tokenize failed\n"); exit(3); }
    return t;
}

typedef struct {
    struct llama_model *model;
    struct llama_context *ctx;
    const struct llama_vocab *vocab;
    llama_memory_t mem;
    const char *grammar;
    int n_predict;
} env_t;

// --sampler fast (default): greedy argmax first, then ask the grammar about that
// one token; only if it is rejected, mask the whole vocabulary. Same result as
// grammar->greedy (argmax of the masked logits), far fewer grammar checks.
// --sampler naive: llama_sampler_chain(grammar, greedy) over the full vocabulary.
static int g_fast = 1;
static long g_full_masks = 0, g_samples = 0;
static llama_token_data *g_cur = NULL;

static struct llama_sampler *mk_sampler(env_t *e) {
    if (g_fast) {
        struct llama_sampler *g = llama_sampler_init_grammar(e->vocab, e->grammar, "root");
        if (!g) { fprintf(stderr, "grammar init failed\n"); exit(4); }
        return g;
    }
    struct llama_sampler *ch = llama_sampler_chain_init(llama_sampler_chain_default_params());
    struct llama_sampler *g = llama_sampler_init_grammar(e->vocab, e->grammar, "root");
    if (!g) { fprintf(stderr, "grammar init failed\n"); exit(4); }
    llama_sampler_chain_add(ch, g);
    llama_sampler_chain_add(ch, llama_sampler_init_greedy());
    return ch;
}

static llama_token sample_tok(env_t *e, struct llama_sampler *s, int idx) {
    g_samples++;
    if (!g_fast) return llama_sampler_sample(s, e->ctx, idx);
    const float *logits = llama_get_logits_ith(e->ctx, idx);
    int nv = llama_vocab_n_tokens(e->vocab);
    int best = 0;
    for (int i = 1; i < nv; i++) if (logits[i] > logits[best]) best = i;
    llama_token_data one = { best, logits[best], 0.0f };
    llama_token_data_array a1 = { &one, 1, -1, false };
    llama_sampler_apply(s, &a1);
    if (isfinite(a1.data[0].logit)) { llama_sampler_accept(s, best); return best; }
    g_full_masks++;
    if (!g_cur) g_cur = malloc(sizeof(llama_token_data) * nv);
    for (int i = 0; i < nv; i++) { g_cur[i].id = i; g_cur[i].logit = logits[i]; g_cur[i].p = 0.0f; }
    llama_token_data_array a = { g_cur, (size_t)nv, -1, false };
    llama_sampler_apply(s, &a);
    size_t bi = 0;
    for (size_t i = 1; i < a.size; i++) if (a.data[i].logit > a.data[bi].logit) bi = i;
    llama_token t = a.data[bi].id;
    llama_sampler_accept(s, t);
    return t;
}

// add tokens [toks, toks+n) of seq `seq` starting at pos p0; logits on last only
static void batch_add_seq(struct llama_batch *b, const llama_token *toks, int n, int p0, llama_seq_id seq, int logits_last) {
    for (int i = 0; i < n; i++) {
        int k = b->n_tokens++;
        b->token[k] = toks[i];
        b->pos[k] = p0 + i;
        b->n_seq_id[k] = 1;
        b->seq_id[k][0] = seq;
        b->logits[k] = (logits_last && i == n - 1);
    }
}

static void decode_or_die(env_t *e, struct llama_batch b) {
    int r = llama_decode(e->ctx, b);
    if (r != 0) { fprintf(stderr, "llama_decode failed: %d\n", r); exit(5); }
}

// Prefill `toks` on seq in chunks of n_batch, logits on the very last token.
static void prefill(env_t *e, struct llama_batch *b, const llama_token *toks, int n, int p0, llama_seq_id seq, int nb) {
    for (int off = 0; off < n; off += nb) {
        int m = n - off < nb ? n - off : nb;
        b->n_tokens = 0;
        batch_add_seq(b, toks + off, m, p0 + off, seq, off + m == n);
        decode_or_die(e, *b);
    }
}

typedef struct {
    double ttft_ms;      // child start -> first token sampled
    double gen_ms;       // first token -> stop
    int n_gen;           // tokens generated (incl. first)
    int n_prefilled;     // tokens this child had to prefill
    char text[512];
} child_res;

static int piece(env_t *e, llama_token t, char *buf, int cap) {
    int n = llama_token_to_piece(e->vocab, t, buf, cap - 1, 0, 1);
    if (n < 0) n = 0;
    buf[n] = 0;
    return n;
}

// Generate one child to completion from logits index `idx` (single-seq path).
static void gen_single(env_t *e, struct llama_batch *b, llama_seq_id seq, int pos, int idx,
                       double t0, child_res *r) {
    struct llama_sampler *s = mk_sampler(e);
    r->text[0] = 0;
    int len = 0;
    for (int k = 0; k < e->n_predict; k++) {
        llama_token t = sample_tok(e, s, idx);
        if (k == 0) r->ttft_ms = now_ms() - t0;
        r->n_gen = k + 1;
        if (llama_vocab_is_eog(e->vocab, t)) break;
        char p[64]; int n = piece(e, t, p, sizeof p);
        if (len + n < (int)sizeof(r->text) - 1) { memcpy(r->text + len, p, n); len += n; r->text[len] = 0; }
        if (strchr(p, '\n')) break;
        b->n_tokens = 0;
        batch_add_seq(b, &t, 1, pos++, seq, 1);
        decode_or_die(e, *b);
        idx = 0;
    }
    r->gen_ms = now_ms() - t0 - r->ttft_ms;
    llama_sampler_free(s);
}

static void print_children(FILE *o, const char *name, child_res *rs, int n) {
    fprintf(o, "\"%s\":[", name);
    for (int i = 0; i < n; i++) {
        fprintf(o, "%s{\"ttft_ms\":%.3f,\"gen_ms\":%.3f,\"n_gen\":%d,\"n_prefilled\":%d,\"text\":",
                i ? "," : "", rs[i].ttft_ms, rs[i].gen_ms, rs[i].n_gen, rs[i].n_prefilled);
        json_str(o, rs[i].text);
        fputc('}', o);
    }
    fputc(']', o);
}

int main(int argc, char **argv) {
    const char *model_path = NULL, *grammar_path = NULL, *parent_path = NULL, *children_path = NULL;
    int n_predict = 32, threads = 4, n_ctx = 4096, reps = 1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "--grammar") && i + 1 < argc) grammar_path = argv[++i];
        else if (!strcmp(argv[i], "--parent") && i + 1 < argc) parent_path = argv[++i];
        else if (!strcmp(argv[i], "--children") && i + 1 < argc) children_path = argv[++i];
        else if (!strcmp(argv[i], "--n-predict") && i + 1 < argc) n_predict = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) n_ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sampler") && i + 1 < argc) g_fast = strcmp(argv[++i], "naive") != 0;
        else { fprintf(stderr, "usage: seqcp_bench --model M --grammar G --parent P --children C [--n-predict N] [--threads T] [--ctx N] [--reps R] [--sampler fast|naive]\n"); return 64; }
    }
    if (!model_path || !grammar_path || !parent_path || !children_path) { fprintf(stderr, "missing args\n"); return 64; }

    char *parent = read_file(parent_path);
    char *children_txt = read_file(children_path);
    char *suffix[MAXC]; int nc = 0;
    for (char *line = strtok(children_txt, "\n"); line && nc < MAXC; line = strtok(NULL, "\n"))
        if (*line) {
            // "\n" in the file is written as the two chars \ n
            size_t L = strlen(line); char *s = malloc(L + 2); size_t j = 0;
            for (size_t k = 0; k < L; k++) {
                if (line[k] == '\\' && line[k + 1] == 'n') { s[j++] = '\n'; k++; } else s[j++] = line[k];
            }
            s[j] = 0; suffix[nc++] = s;
        }

    llama_log_set(quiet_log, NULL);
    llama_backend_init();
    double t_load0 = now_ms();
    struct llama_model_params mp = llama_model_default_params();
    struct llama_model *model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 6; }

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_ctx;
    cp.n_ubatch = 512;
    cp.n_seq_max = nc + 1;
    cp.kv_unified = true;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.no_perf = false;
    struct llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context init failed\n"); return 7; }
    double t_load = now_ms() - t_load0;

    env_t e = { model, ctx, llama_model_get_vocab(model), llama_get_memory(ctx), read_file(grammar_path), n_predict };

    tokvec P = tokenize(e.vocab, parent, 1);
    tokvec S[MAXC];
    int sumS = 0;
    for (int i = 0; i < nc; i++) { S[i] = tokenize(e.vocab, suffix[i], 0); sumS += S[i].n; }

    struct llama_batch b = llama_batch_init(n_ctx, 0, 1);

    // warm-up (allocates compute buffers, not timed)
    llama_memory_clear(e.mem, true);
    prefill(&e, &b, P.toks, P.n, 0, 0, cp.n_batch);
    llama_memory_clear(e.mem, true);

    static child_res sh[MAXC], cold[MAXC], shb[MAXC], coldb[MAXC];
    double best_parent_ms = 1e18, best_sh_total = 1e18, best_cold_total = 1e18;
    double shb_prefill_ms = 0, coldb_prefill_ms = 0, shb_gen_ms = 0, coldb_gen_ms = 0;
    int shb_tokens = 0, coldb_tokens = 0;

    for (int rep = 0; rep < reps; rep++) {
        // ---------- A. shared prefix, children one after another ----------
        llama_memory_clear(e.mem, true);
        double t0 = now_ms();
        prefill(&e, &b, P.toks, P.n, 0, 0, cp.n_batch);
        double parent_ms = now_ms() - t0;
        if (parent_ms < best_parent_ms) best_parent_ms = parent_ms;
        double tA = now_ms();
        for (int i = 0; i < nc; i++) {
            llama_seq_id sq = i + 1;
            double c0 = now_ms();
            llama_memory_seq_cp(e.mem, 0, sq, -1, -1);           // zero-copy (unified KV)
            prefill(&e, &b, S[i].toks, S[i].n, P.n, sq, cp.n_batch); // suffix only
            sh[i].n_prefilled = S[i].n;
            gen_single(&e, &b, sq, P.n + S[i].n, b.n_tokens - 1, c0, &sh[i]);
            llama_memory_seq_rm(e.mem, sq, -1, -1);
        }
        double shA = now_ms() - tA + parent_ms;
        if (shA < best_sh_total) best_sh_total = shA;

        // ---------- B. cold, children one after another ----------
        llama_memory_clear(e.mem, true);
        double tB = now_ms();
        for (int i = 0; i < nc; i++) {
            llama_seq_id sq = i + 1;
            double c0 = now_ms();
            int n = P.n + S[i].n;
            llama_token *full = malloc(sizeof(llama_token) * n);
            memcpy(full, P.toks, sizeof(llama_token) * P.n);
            memcpy(full + P.n, S[i].toks, sizeof(llama_token) * S[i].n);
            prefill(&e, &b, full, n, 0, sq, cp.n_batch);
            cold[i].n_prefilled = n;
            gen_single(&e, &b, sq, n, b.n_tokens - 1, c0, &cold[i]);
            llama_memory_seq_rm(e.mem, sq, -1, -1);
            free(full);
        }
        double coldB = now_ms() - tB;
        if (coldB < best_cold_total) best_cold_total = coldB;

        // ---------- C/D. batched: all children decode together ----------
        for (int mode = 0; mode < 2; mode++) {  // 0 = shared, 1 = cold
            child_res *rs = mode == 0 ? shb : coldb;
            llama_memory_clear(e.mem, true);
            double t = now_ms();
            b.n_tokens = 0;
            int idx[MAXC], pos[MAXC], active[MAXC], len[MAXC];
            if (mode == 0) {
                prefill(&e, &b, P.toks, P.n, 0, 0, cp.n_batch);
                for (int i = 0; i < nc; i++) llama_memory_seq_cp(e.mem, 0, i + 1, -1, -1);
                b.n_tokens = 0;
                for (int i = 0; i < nc; i++) {
                    batch_add_seq(&b, S[i].toks, S[i].n, P.n, i + 1, 1);
                    idx[i] = b.n_tokens - 1; pos[i] = P.n + S[i].n; rs[i].n_prefilled = S[i].n;
                }
            } else {
                for (int i = 0; i < nc; i++) {
                    batch_add_seq(&b, P.toks, P.n, 0, i + 1, 0);
                    batch_add_seq(&b, S[i].toks, S[i].n, P.n, i + 1, 1);
                    idx[i] = b.n_tokens - 1; pos[i] = P.n + S[i].n; rs[i].n_prefilled = P.n + S[i].n;
                }
            }
            decode_or_die(&e, b);
            double t_pref = now_ms() - t;
            struct llama_sampler *smp[MAXC];
            int n_active = nc, total = 0;
            for (int i = 0; i < nc; i++) { smp[i] = mk_sampler(&e); active[i] = 1; len[i] = 0; rs[i].text[0] = 0; rs[i].n_gen = 0; }
            double tg0 = now_ms();
            for (int step = 0; step < n_predict && n_active > 0; step++) {
                llama_token next[MAXC];
                for (int i = 0; i < nc; i++) {
                    if (!active[i]) continue;
                    llama_token tk = sample_tok(&e, smp[i], idx[i]);
                    if (step == 0) rs[i].ttft_ms = now_ms() - t;
                    rs[i].n_gen++; total++;
                    next[i] = tk;
                    if (llama_vocab_is_eog(e.vocab, tk)) { active[i] = 0; n_active--; rs[i].gen_ms = now_ms() - t - rs[i].ttft_ms; continue; }
                    char p[64]; int n = piece(&e, tk, p, sizeof p);
                    if (len[i] + n < (int)sizeof(rs[i].text) - 1) { memcpy(rs[i].text + len[i], p, n); len[i] += n; rs[i].text[len[i]] = 0; }
                    if (strchr(p, '\n') || step == n_predict - 1) { active[i] = 0; n_active--; rs[i].gen_ms = now_ms() - t - rs[i].ttft_ms; }
                }
                if (n_active == 0) break;
                b.n_tokens = 0;
                for (int i = 0; i < nc; i++) if (active[i]) { batch_add_seq(&b, &next[i], 1, pos[i]++, i + 1, 1); idx[i] = b.n_tokens - 1; }
                decode_or_die(&e, b);
            }
            double tg = now_ms() - tg0;
            for (int i = 0; i < nc; i++) llama_sampler_free(smp[i]);
            if (mode == 0) { shb_prefill_ms = t_pref; shb_gen_ms = tg; shb_tokens = total; }
            else { coldb_prefill_ms = t_pref; coldb_gen_ms = tg; coldb_tokens = total; }
        }
    }

    int same = 0, same_b = 0;
    for (int i = 0; i < nc; i++) { same += !strcmp(sh[i].text, cold[i].text); same_b += !strcmp(shb[i].text, coldb[i].text); }

    FILE *o = stdout;
    fprintf(o, "{\"model\":"); json_str(o, model_path);
    fprintf(o, ",\"n_ctx_train\":%d,\"n_ctx\":%d,\"kv_unified\":true,\"threads\":%d,\"reps\":%d,\"load_ms\":%.1f,\"sampler\":\"%s\",\"samples\":%ld,\"full_vocab_grammar_masks\":%ld",
            llama_model_n_ctx_train(model), n_ctx, threads, reps, t_load, g_fast ? "fast" : "naive", g_samples, g_full_masks);
    fprintf(o, ",\"n_children\":%d,\"parent_tokens\":%d,\"suffix_tokens_total\":%d", nc, P.n, sumS);
    fprintf(o, ",\"sequential\":{\"parent_prefill_ms\":%.3f,\"shared_total_ms\":%.3f,\"cold_total_ms\":%.3f,"
               "\"prefill_tokens_shared\":%d,\"prefill_tokens_cold\":%d,\"outputs_identical\":%d,",
            best_parent_ms, best_sh_total, best_cold_total, P.n + sumS, nc * P.n + sumS, same);
    print_children(o, "shared", sh, nc); fputc(',', o);
    print_children(o, "cold", cold, nc);
    fprintf(o, "},\"batched\":{\"shared_prefill_ms\":%.3f,\"cold_prefill_ms\":%.3f,"
               "\"shared_gen_ms\":%.3f,\"cold_gen_ms\":%.3f,\"shared_tokens\":%d,\"cold_tokens\":%d,"
               "\"kv_cells_shared\":%d,\"kv_cells_cold\":%d,\"outputs_identical\":%d,",
            shb_prefill_ms, coldb_prefill_ms, shb_gen_ms, coldb_gen_ms, shb_tokens, coldb_tokens,
            P.n + sumS, nc * P.n + sumS, same_b);
    print_children(o, "shared", shb, nc); fputc(',', o);
    print_children(o, "cold", coldb, nc);
    fprintf(o, "}}\n");

    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
