use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{sync_channel, Receiver};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::JoinHandle;

use crate::backend::InferenceBackend;
use crate::types::{WorkerConfig, WorkerError, WorkerId, WorkerResult};
use crate::worker::InferenceWorker;

type Registry = Mutex<HashMap<WorkerId, Arc<AtomicBool>>>;

fn registry() -> &'static Registry {
    static R: OnceLock<Registry> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

fn lock() -> std::sync::MutexGuard<'static, HashMap<WorkerId, Arc<AtomicBool>>> {
    // A panicking worker cannot poison us into a dead registry.
    registry().lock().unwrap_or_else(|p| p.into_inner())
}

/// RAII: removes the worker's entry when the worker thread ends, including on
/// panic (bug 7).
struct RegistryGuard(WorkerId);

impl Drop for RegistryGuard {
    fn drop(&mut self) {
        lock().remove(&self.0);
    }
}

/// Handle returned by [`spawn_worker`].
pub struct WorkerHandle {
    pub id: WorkerId,
    pub result: Receiver<Result<WorkerResult, WorkerError>>,
    pub join: JoinHandle<()>,
}

impl WorkerHandle {
    /// Block until the worker finishes; joins the thread.
    pub fn wait(self) -> Result<WorkerResult, WorkerError> {
        let r = self
            .result
            .recv()
            .unwrap_or_else(|_| Err(WorkerError::Backend("worker exited without a result (panic)".into())));
        let _ = self.join.join();
        r
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SpawnOptions {
    /// Thread stack size. Small by default: a worker only runs the generate loop.
    pub stack_size: usize,
    pub strict_contract: bool,
}

impl Default for SpawnOptions {
    fn default() -> Self {
        SpawnOptions { stack_size: 64 * 1024, strict_contract: false }
    }
}

/// D2.1: spawn a worker on its own thread (bug 2: the original dropped it
/// immediately). Registered *before* the thread starts so `cancel_worker` works
/// at once.
pub fn spawn_worker(config: WorkerConfig, backend: Box<dyn InferenceBackend>) -> std::io::Result<WorkerHandle> {
    spawn_worker_with(config, backend, SpawnOptions::default())
}

pub fn spawn_worker_with(
    config: WorkerConfig,
    mut backend: Box<dyn InferenceBackend>,
    opts: SpawnOptions,
) -> std::io::Result<WorkerHandle> {
    let id = WorkerId::next();
    let flag = Arc::new(AtomicBool::new(false));
    // new() is called here, so spawn_latency includes thread creation.
    let mut worker = InferenceWorker::with_flag(id, config, flag.clone());
    if opts.strict_contract {
        worker = worker.strict_contract();
    }
    lock().insert(id, flag);
    let guard = RegistryGuard(id);
    let (tx, rx) = sync_channel(1);
    let join = std::thread::Builder::new()
        .name(format!("worker-{}", id.0))
        .stack_size(opts.stack_size)
        .spawn(move || {
            let _guard = guard; // dropped when the thread exits (incl. panic)
            let r = worker.generate(backend.as_mut());
            drop(backend);
            let _ = tx.send(r);
        });
    match join {
        Ok(join) => Ok(WorkerHandle { id, result: rx, join }),
        Err(e) => Err(e), // guard was moved into the closure, which is dropped => entry removed
    }
}

/// D2.5: set the worker's cancel flag. Returns whether the worker was found
/// (bug 3: the original was a no-op).
pub fn cancel_worker(id: WorkerId) -> bool {
    match lock().get(&id) {
        Some(f) => {
            f.store(true, Ordering::Release);
            true
        }
        None => false,
    }
}

pub fn is_registered(id: WorkerId) -> bool {
    lock().contains_key(&id)
}

pub fn active_workers() -> usize {
    lock().len()
}
