use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// Process-unique worker id from a global atomic counter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WorkerId(pub u64);

impl WorkerId {
    pub fn next() -> Self {
        WorkerId(NEXT_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl fmt::Display for WorkerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "w{}", self.0)
    }
}

/// Immutable per-worker configuration (the worker's context slice).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerConfig {
    pub task_id: String,
    /// Max tokens to generate. `u32`: a negative budget is unrepresentable (bug 6).
    pub budget_tokens: u32,
    /// Wall-clock deadline from worker creation; 0 = none. Enforced (bug 4).
    pub deadline_ms: u64,
    /// The child-specific prompt (suffix after the shared parent prompt).
    pub context: String,
}

impl WorkerConfig {
    pub fn new(task_id: impl Into<String>, budget_tokens: u32, deadline_ms: u64, context: impl Into<String>) -> Self {
        WorkerConfig { task_id: task_id.into(), budget_tokens, deadline_ms, context: context.into() }
    }
}

/// Measured latencies (microseconds).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct LatencyProfile {
    /// `InferenceWorker::new()` -> first iteration of the generate loop (bug 5:
    /// this used to time the whole generation).
    pub spawn_latency_us: u64,
    /// First loop iteration -> first token returned by the backend (includes the
    /// child's prefill).
    pub first_token_latency_us: u64,
    /// First loop iteration -> generation finished.
    pub generation_us: u64,
    /// new() -> result ready (incl. validation).
    pub total_us: u64,
    /// Tokens this worker had to prefill (suffix only when the parent is shared).
    pub prefilled_tokens: u32,
}

impl LatencyProfile {
    pub fn tokens_per_sec(&self, tokens: u32) -> f64 {
        if self.generation_us == 0 { 0.0 } else { tokens as f64 * 1e6 / self.generation_us as f64 }
    }
}

/// Deterministic payload of a successful worker.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkerResult {
    pub worker_id: WorkerId,
    pub task_id: String,
    /// Raw generated text.
    pub output: String,
    /// Canonical, contract-approved command lines.
    pub commands: Vec<String>,
    pub tokens_generated: u32,
    pub latency: LatencyProfile,
}

#[derive(Debug, Clone, PartialEq)]
pub enum WorkerError {
    Cancelled { tokens: u32 },
    Timeout { elapsed_ms: u64, deadline_ms: u64, tokens: u32 },
    /// Budget used up before a complete command sequence was produced.
    BudgetExhausted { tokens: u32, partial: String },
    /// Output is not in the grammar.
    Parse(String),
    /// Output parsed but violated the ExecutionContract.
    ContractViolation { index: usize, message: String },
    Backend(String),
}

impl WorkerError {
    pub fn kind(&self) -> &'static str {
        match self {
            WorkerError::Cancelled { .. } => "cancelled",
            WorkerError::Timeout { .. } => "timeout",
            WorkerError::BudgetExhausted { .. } => "budget_exhausted",
            WorkerError::Parse(_) => "parse",
            WorkerError::ContractViolation { .. } => "contract_violation",
            WorkerError::Backend(_) => "backend",
        }
    }
}

impl fmt::Display for WorkerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WorkerError::Cancelled { tokens } => write!(f, "cancelled after {} tokens", tokens),
            WorkerError::Timeout { elapsed_ms, deadline_ms, tokens } => {
                write!(f, "deadline {} ms exceeded after {} ms ({} tokens)", deadline_ms, elapsed_ms, tokens)
            }
            WorkerError::BudgetExhausted { tokens, .. } => write!(f, "token budget {} exhausted", tokens),
            WorkerError::Parse(m) => write!(f, "{}", m),
            WorkerError::ContractViolation { index, message } => write!(f, "instruction {}: {}", index, message),
            WorkerError::Backend(m) => write!(f, "backend: {}", m),
        }
    }
}

impl std::error::Error for WorkerError {}
