//! Ephemeral inference workers.
//!
//! * D2.1 [`spawn_worker`]: one OS thread per worker, registered in a global
//!   registry (`Mutex<HashMap<WorkerId, Arc<AtomicBool>>>`, no DashMap), returns a
//!   [`WorkerHandle`] (id + result channel + join handle).
//! * D2.3 ephemeral session: a worker owns its backend session (a llama.cpp
//!   sequence id in a shared context, or a mock) and releases it on exit; it
//!   never receives a reference to shared application state.
//! * D2.4 budget: `budget_tokens: u32`, enforced in the generate loop.
//! * D2.5 cancel & timeout: [`cancel_worker`] flips the worker's `AtomicBool` via
//!   the registry; `deadline_ms` is checked every iteration.
//!
//! Outputs are validated with `sovereign_cmd::parse_program` +
//! `ExecutionContract`; a failing worker returns an error and exits.

pub mod backend;
pub mod json;
pub mod ledger;
pub mod registry;
pub mod types;
pub mod worker;

#[cfg(feature = "llama")]
pub mod llama;

pub use backend::{InferenceBackend, MockBackend};
pub use ledger::Ledger;
pub use registry::{active_workers, cancel_worker, is_registered, spawn_worker, spawn_worker_with, SpawnOptions, WorkerHandle};
pub use types::{LatencyProfile, WorkerConfig, WorkerError, WorkerId, WorkerResult};
pub use worker::InferenceWorker;
