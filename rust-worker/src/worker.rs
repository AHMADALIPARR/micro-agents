use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use sovereign_cmd::{parse_program, ExecutionContract};

use crate::backend::InferenceBackend;
use crate::types::{LatencyProfile, WorkerConfig, WorkerError, WorkerId, WorkerResult};

/// One ephemeral worker. Holds only its own config, cancel flag and clock.
pub struct InferenceWorker {
    pub id: WorkerId,
    pub config: WorkerConfig,
    cancel: Arc<AtomicBool>,
    created: Instant,
    contract: ExecutionContract,
}

impl InferenceWorker {
    pub fn new(config: WorkerConfig) -> Self {
        Self::with_flag(WorkerId::next(), config, Arc::new(AtomicBool::new(false)))
    }

    pub fn with_flag(id: WorkerId, config: WorkerConfig, cancel: Arc<AtomicBool>) -> Self {
        InferenceWorker { id, config, cancel, created: Instant::now(), contract: ExecutionContract::default() }
    }

    pub fn strict_contract(mut self) -> Self {
        self.contract = ExecutionContract::strict();
        self
    }

    pub fn cancel_flag(&self) -> Arc<AtomicBool> {
        self.cancel.clone()
    }

    fn deadline_hit(&self) -> Option<u64> {
        if self.config.deadline_ms == 0 {
            return None;
        }
        let el = self.created.elapsed().as_millis() as u64;
        if el >= self.config.deadline_ms { Some(el) } else { None }
    }

    /// Generate under budget, cancel flag and deadline; then validate.
    pub fn generate(&mut self, backend: &mut dyn InferenceBackend) -> Result<WorkerResult, WorkerError> {
        let mut lat = LatencyProfile::default();
        let mut out = String::new();
        let mut tokens: u32 = 0;
        let mut loop_start: Option<Instant> = None;

        // Cancel/deadline may already be due before we start.
        if self.cancel.load(Ordering::Acquire) {
            return Err(WorkerError::Cancelled { tokens: 0 });
        }

        let mut began = false;
        let res: Result<(), WorkerError> = loop {
            if loop_start.is_none() {
                let now = Instant::now();
                lat.spawn_latency_us = (now - self.created).as_micros() as u64;
                loop_start = Some(now);
            }
            if self.cancel.load(Ordering::Acquire) {
                break Err(WorkerError::Cancelled { tokens });
            }
            if let Some(el) = self.deadline_hit() {
                break Err(WorkerError::Timeout { elapsed_ms: el, deadline_ms: self.config.deadline_ms, tokens });
            }
            if !began {
                match backend.begin(&self.config.context) {
                    Ok(n) => lat.prefilled_tokens = n,
                    Err(e) => break Err(WorkerError::Backend(e)),
                }
                began = true;
                continue; // re-check cancel/deadline after a (possibly long) prefill
            }
            if tokens >= self.config.budget_tokens {
                break Err(WorkerError::BudgetExhausted { tokens, partial: out.clone() });
            }
            match backend.next_token() {
                Ok(Some(piece)) => {
                    tokens += 1;
                    if tokens == 1 {
                        lat.first_token_latency_us = loop_start.unwrap().elapsed().as_micros() as u64;
                    }
                    out.push_str(&piece);
                    if backend.stop_at_newline() && piece.contains('\n') {
                        break Ok(());
                    }
                }
                Ok(None) => break Ok(()),
                Err(e) => break Err(WorkerError::Backend(e)),
            }
        };
        backend.end();
        lat.generation_us = loop_start.map(|t| t.elapsed().as_micros() as u64).unwrap_or(0);
        res?;

        // Truncate after the first newline for line-oriented backends.
        if backend.stop_at_newline() {
            if let Some(p) = out.find('\n') {
                out.truncate(p + 1);
            }
        }
        let prog = parse_program(&out).map_err(|e| WorkerError::Parse(e.to_string()))?;
        self.contract
            .validate_program(&prog)
            .map_err(|(i, m)| WorkerError::ContractViolation { index: i, message: m.to_string() })?;
        lat.total_us = self.created.elapsed().as_micros() as u64;
        Ok(WorkerResult {
            worker_id: self.id,
            task_id: self.config.task_id.clone(),
            output: out,
            commands: prog.iter().map(|c| c.to_string()).collect(),
            tokens_generated: tokens,
            latency: lat,
        })
    }
}
