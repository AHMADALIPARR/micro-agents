use std::time::{Duration, Instant};

use rust_worker::*;

fn cfg(id: &str, budget: u32, deadline: u64) -> WorkerConfig {
    WorkerConfig::new(id, budget, deadline, "ctx")
}

fn mock(out: &str) -> Box<dyn InferenceBackend> {
    Box::new(MockBackend::new(out, Duration::ZERO))
}

fn wait_unregistered(id: WorkerId) {
    let t = Instant::now();
    while is_registered(id) {
        assert!(t.elapsed() < Duration::from_secs(2), "registry entry leaked for {}", id);
        std::thread::yield_now();
    }
}

// ---- the user's three tests (reconstructed from the deliverables) ----------

/// D2.1: spawn_worker returns unique ids and the worker actually runs.
#[test]
fn test_spawn_worker() {
    let a = spawn_worker(cfg("a", 32, 0), mock("READ x\n")).unwrap();
    let b = spawn_worker(cfg("b", 32, 0), mock("READ y\n")).unwrap();
    assert_ne!(a.id, b.id);
    let (ia, ib) = (a.id, b.id);
    let ra = a.wait().unwrap();
    let rb = b.wait().unwrap();
    assert_eq!(ra.commands, vec!["READ x"]);
    assert_eq!(rb.commands, vec!["READ y"]);
    assert_eq!(ra.worker_id, ia);
    assert_eq!(rb.worker_id, ib);
}

/// D2.4: the token budget is enforced.
#[test]
fn test_budget_enforced() {
    let w = spawn_worker(cfg("b", 3, 0), Box::new(MockBackend::endless(Duration::ZERO))).unwrap();
    match w.wait() {
        Err(WorkerError::BudgetExhausted { tokens, .. }) => assert_eq!(tokens, 3),
        other => panic!("expected BudgetExhausted, got {:?}", other),
    }
}

/// D2.5: a cancelled worker stops generating.
#[test]
fn test_cancel_stops_generation() {
    let w = spawn_worker(cfg("c", u32::MAX, 0), Box::new(MockBackend::endless(Duration::from_millis(1)))).unwrap();
    std::thread::sleep(Duration::from_millis(20));
    assert!(cancel_worker(w.id));
    match w.wait() {
        Err(WorkerError::Cancelled { tokens }) => assert!(tokens > 0),
        other => panic!("expected Cancelled, got {:?}", other),
    }
}

// ---- added tests ------------------------------------------------------------------

#[test]
fn cancel_via_id_reports_found_and_not_found() {
    let w = spawn_worker(cfg("c2", u32::MAX, 0), Box::new(MockBackend::endless(Duration::from_millis(1)))).unwrap();
    let id = w.id;
    assert!(is_registered(id));
    assert!(cancel_worker(id));
    assert!(matches!(w.wait(), Err(WorkerError::Cancelled { .. })));
    wait_unregistered(id);
    assert!(!cancel_worker(id), "cancel of a finished worker must report not-found");
    assert!(!cancel_worker(WorkerId(u64::MAX)));
}

#[test]
fn deadline_enforced_as_timeout() {
    let t = Instant::now();
    let w = spawn_worker(cfg("t", u32::MAX, 30), Box::new(MockBackend::endless(Duration::from_millis(5)))).unwrap();
    match w.wait() {
        Err(WorkerError::Timeout { elapsed_ms, deadline_ms, .. }) => {
            assert_eq!(deadline_ms, 30);
            assert!(elapsed_ms >= 30);
        }
        other => panic!("expected Timeout, got {:?}", other),
    }
    // detection granularity = one backend call (5 ms here)
    assert!(t.elapsed() < Duration::from_millis(500));
}

struct Panicky;
impl InferenceBackend for Panicky {
    fn begin(&mut self, _: &str) -> Result<u32, String> {
        Ok(0)
    }
    fn next_token(&mut self) -> Result<Option<String>, String> {
        panic!("backend blew up")
    }
}

#[test]
fn registry_cleanup_on_success_error_and_panic() {
    let ok = spawn_worker(cfg("r1", 8, 0), mock("READ x\n")).unwrap();
    let bad = spawn_worker(cfg("r2", 8, 0), mock("WRITE 0x1\n")).unwrap();
    let boom = spawn_worker(cfg("r3", 8, 0), Box::new(Panicky)).unwrap();
    let ids = [ok.id, bad.id, boom.id];
    assert!(ok.wait().is_ok());
    assert!(bad.wait().is_err());
    assert!(matches!(boom.wait(), Err(WorkerError::Backend(_))));
    for id in ids {
        wait_unregistered(id);
    }
}

#[test]
fn many_workers_leave_no_registry_entries() {
    let hs: Vec<_> = (0..500).map(|i| spawn_worker(cfg(&format!("m{}", i), 16, 0), mock("SET a 1\n")).unwrap()).collect();
    let ids: Vec<_> = hs.iter().map(|h| h.id).collect();
    for h in hs {
        assert!(h.wait().is_ok());
    }
    for id in ids {
        wait_unregistered(id);
    }
}

#[test]
fn contract_failure_self_terminates_and_never_touches_ledger() {
    let mut ledger = Ledger::default();
    let good = spawn_worker(cfg("g", 16, 0), mock("WRITE 0x2000 --force 42\n")).unwrap().wait().unwrap();
    ledger.apply(&good);
    let snapshot = ledger.clone();

    for (out, want_idx) in [("WRITE 0xFFF 1\n", 0usize), ("RUN /opt/sovereign/bin/ok\nRUN /tmp/evil\n", 1)] {
        let h = spawn_worker(cfg("bad", 32, 0), mock(out)).unwrap();
        let id = h.id;
        match h.wait() {
            Err(WorkerError::ContractViolation { index, message }) => {
                assert_eq!(index, want_idx);
                assert!(message.contains("CONTRACT VIOLATION"));
            }
            other => panic!("expected ContractViolation, got {:?}", other),
        }
        wait_unregistered(id);
    }
    assert_eq!(ledger, snapshot);
}

#[test]
fn parse_failure_is_reported() {
    let r = spawn_worker(cfg("p", 16, 0), mock("read x\n")).unwrap().wait();
    assert!(matches!(r, Err(WorkerError::Parse(_))));
}

#[test]
fn strict_contract_option() {
    let r = spawn_worker_with(cfg("s", 16, 0), mock("RUN foo\n"), SpawnOptions { strict_contract: true, ..Default::default() })
        .unwrap()
        .wait();
    assert!(matches!(r, Err(WorkerError::ContractViolation { .. })));
    assert!(spawn_worker(cfg("s2", 16, 0), mock("RUN foo\n")).unwrap().wait().is_ok());
}

#[test]
fn latency_profile_is_measured() {
    let r = spawn_worker(cfg("l", 16, 0), Box::new(MockBackend::new("READ x\n", Duration::from_millis(2))))
        .unwrap()
        .wait()
        .unwrap();
    // spawn latency (new -> first iteration) must not include generation (bug 5)
    assert!(r.latency.first_token_latency_us >= 2_000);
    assert!(r.latency.generation_us >= r.latency.first_token_latency_us);
    assert!(r.latency.spawn_latency_us < r.latency.generation_us);
}
