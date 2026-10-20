use std::time::Duration;

/// An inference session. One per worker; `begin` starts it, `next_token` yields
/// text pieces until `Ok(None)`, `end` releases resources (also on drop).
pub trait InferenceBackend: Send {
    /// Start a session for `context`. Returns the number of tokens prefilled.
    fn begin(&mut self, context: &str) -> Result<u32, String>;
    /// Next generated text piece, or `None` at end of generation.
    fn next_token(&mut self) -> Result<Option<String>, String>;
    /// Release the session.
    fn end(&mut self) {}
    /// Stop after the first complete line (grammar-constrained LLMs keep going).
    fn stop_at_newline(&self) -> bool {
        true
    }
}

/// Deterministic test backend (the user's mock): replays `script` one piece per
/// call, sleeping `token_delay` before each.
#[derive(Debug, Clone)]
pub struct MockBackend {
    pub script: Vec<String>,
    pub token_delay: Duration,
    pub prefill: u32,
    idx: usize,
}

impl MockBackend {
    /// Split `output` into word-ish pieces (each space/newline is its own token).
    pub fn new(output: &str, token_delay: Duration) -> Self {
        let mut script = Vec::new();
        let mut cur = String::new();
        for ch in output.chars() {
            if ch == ' ' || ch == '\n' {
                if !cur.is_empty() {
                    script.push(std::mem::take(&mut cur));
                }
                script.push(ch.to_string());
            } else {
                cur.push(ch);
            }
        }
        if !cur.is_empty() {
            script.push(cur);
        }
        MockBackend { script, token_delay, prefill: 0, idx: 0 }
    }

    /// Endless stream of tokens (for cancel/timeout tests).
    pub fn endless(token_delay: Duration) -> Self {
        MockBackend { script: Vec::new(), token_delay, prefill: 0, idx: usize::MAX }
    }
}

impl InferenceBackend for MockBackend {
    fn begin(&mut self, context: &str) -> Result<u32, String> {
        self.prefill = context.split_whitespace().count() as u32;
        if self.idx != usize::MAX {
            self.idx = 0;
        }
        Ok(self.prefill)
    }

    fn next_token(&mut self) -> Result<Option<String>, String> {
        if !self.token_delay.is_zero() {
            std::thread::sleep(self.token_delay);
        }
        if self.idx == usize::MAX {
            return Ok(Some("x".into()));
        }
        let t = self.script.get(self.idx).cloned();
        self.idx += 1;
        Ok(t)
    }

    fn stop_at_newline(&self) -> bool {
        false
    }
}
