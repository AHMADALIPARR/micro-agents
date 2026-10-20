//! llama.cpp backend through `csrc/llama_shim.c` (linked against the prebuilt
//! libllama.so).
//!
//! [`LlamaShared`] owns ONE llama_context (unified KV cache, `n_seq` sequences).
//! The parent prompt is prefilled once into seq 0. Every [`LlamaBackend`] session
//! takes a free seq id, `llama_memory_seq_cp(0 -> seq)` (metadata-only under
//! kv_unified: the parent's KV cells are shared, not copied), decodes only its
//! suffix and generates under the GBNF grammar. Calls into the context are
//! serialised by a Mutex (llama_context is not thread-safe), so concurrent
//! workers interleave token by token; they are not batched.

use std::ffi::{c_char, c_int, c_long, CString};
use std::sync::{Arc, Mutex};

#[repr(C)]
struct ShimCtx {
    _p: [u8; 0],
}

extern "C" {
    fn shim_new(model: *const c_char, grammar: *const c_char, n_ctx: c_int, n_seq: c_int, threads: c_int) -> *mut ShimCtx;
    fn shim_free(s: *mut ShimCtx);
    fn shim_set_parent(s: *mut ShimCtx, text: *const c_char) -> c_int;
    fn shim_child_begin(s: *mut ShimCtx, seq: c_int, parent: *const c_char, suffix: *const c_char, use_parent: c_int) -> c_int;
    fn shim_child_next(s: *mut ShimCtx, seq: c_int, buf: *mut c_char, cap: c_int) -> c_int;
    fn shim_child_end(s: *mut ShimCtx, seq: c_int);
    fn shim_full_masks(s: *mut ShimCtx) -> c_long;
    fn shim_samples(s: *mut ShimCtx) -> c_long;
}

struct Inner {
    ptr: *mut ShimCtx,
    free_seqs: Vec<i32>,
}

// The raw pointer is only used under the Mutex.
unsafe impl Send for Inner {}

impl Drop for Inner {
    fn drop(&mut self) {
        unsafe { shim_free(self.ptr) }
    }
}

pub struct LlamaShared {
    inner: Mutex<Inner>,
    parent: CString,
    parent_tokens: u32,
    pub n_children: usize,
}

#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub model: String,
    pub grammar_file: String,
    pub parent: String,
    pub n_children: usize,
    pub n_ctx: u32,
    pub threads: u32,
}

impl LlamaConfig {
    pub fn defaults(parent: impl Into<String>) -> Self {
        LlamaConfig {
            model: "/workspace/gguf-lab-october/runs/small-cpu-v3/small-v3-final.gguf".into(),
            grammar_file: "/workspace/sovereign-cmd/grammar/commands.llama.gbnf".into(),
            parent: parent.into(),
            n_children: 8,
            n_ctx: 4096,
            threads: 8,
        }
    }
}

impl LlamaShared {
    pub fn load(cfg: &LlamaConfig) -> Result<Arc<Self>, String> {
        let grammar = std::fs::read_to_string(&cfg.grammar_file).map_err(|e| format!("grammar: {}", e))?;
        let model = CString::new(cfg.model.clone()).map_err(|e| e.to_string())?;
        let g = CString::new(grammar).map_err(|e| e.to_string())?;
        let n_seq = cfg.n_children as c_int + 1;
        let ptr = unsafe { shim_new(model.as_ptr(), g.as_ptr(), cfg.n_ctx as c_int, n_seq, cfg.threads as c_int) };
        if ptr.is_null() {
            return Err(format!("failed to load model {}", cfg.model));
        }
        let inner = Inner { ptr, free_seqs: (1..n_seq).rev().collect() };
        let parent = CString::new(cfg.parent.clone()).map_err(|e| e.to_string())?;
        let n = unsafe { shim_set_parent(inner.ptr, parent.as_ptr()) };
        if n < 0 {
            return Err("parent prefill failed".into());
        }
        Ok(Arc::new(LlamaShared { inner: Mutex::new(inner), parent, parent_tokens: n as u32, n_children: cfg.n_children }))
    }

    pub fn parent_tokens(&self) -> u32 {
        self.parent_tokens
    }

    pub fn stats(&self) -> (i64, i64) {
        let g = self.inner.lock().unwrap();
        unsafe { (shim_samples(g.ptr) as i64, shim_full_masks(g.ptr) as i64) }
    }

    /// New session. `share_parent = false` = cold (child prefills parent itself).
    pub fn session(self: &Arc<Self>, share_parent: bool) -> LlamaBackend {
        LlamaBackend { shared: self.clone(), seq: None, share_parent }
    }
}

pub struct LlamaBackend {
    shared: Arc<LlamaShared>,
    seq: Option<i32>,
    share_parent: bool,
}

impl crate::backend::InferenceBackend for LlamaBackend {
    fn begin(&mut self, context: &str) -> Result<u32, String> {
        let suffix = CString::new(context).map_err(|e| e.to_string())?;
        let mut g = self.shared.inner.lock().unwrap();
        let seq = g.free_seqs.pop().ok_or("no free sequence slot (n_children exhausted)")?;
        self.seq = Some(seq);
        let n = unsafe {
            shim_child_begin(g.ptr, seq, self.shared.parent.as_ptr(), suffix.as_ptr(), self.share_parent as c_int)
        };
        if n < 0 {
            return Err("child prefill failed".into());
        }
        Ok(n as u32)
    }

    fn next_token(&mut self) -> Result<Option<String>, String> {
        let seq = self.seq.ok_or("session not started")?;
        let mut buf = [0u8; 128];
        let g = self.shared.inner.lock().unwrap();
        let n = unsafe { shim_child_next(g.ptr, seq, buf.as_mut_ptr() as *mut c_char, buf.len() as c_int) };
        match n {
            -1 => Ok(None),
            n if n < 0 => Err("decode failed".into()),
            n => Ok(Some(String::from_utf8_lossy(&buf[..n as usize]).into_owned())),
        }
    }

    fn end(&mut self) {
        if let Some(seq) = self.seq.take() {
            let mut g = self.shared.inner.lock().unwrap();
            unsafe { shim_child_end(g.ptr, seq) };
            g.free_seqs.push(seq);
        }
    }
}

impl Drop for LlamaBackend {
    fn drop(&mut self) {
        crate::backend::InferenceBackend::end(self);
    }
}
