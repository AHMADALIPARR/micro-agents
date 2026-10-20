#![cfg(feature = "llama")]
use std::sync::Arc;
use rust_worker::llama::{LlamaConfig, LlamaShared};
use rust_worker::*;

const PARENT: &str = include_str!("../../priv/llm/parent.txt");

fn shared() -> Option<Arc<LlamaShared>> {
    let cfg = LlamaConfig { n_children: 4, threads: 4, ..LlamaConfig::defaults(PARENT) };
    if !std::path::Path::new(&cfg.model).exists() {
        eprintln!("model missing, skipping");
        return None;
    }
    Some(LlamaShared::load(&cfg).expect("load"))
}

#[test]
fn llama_shared_parent_matches_cold_and_validates() {
    let Some(s) = shared() else { return };
    let ctx = "Task: run the agent.\nCommand:";
    let warm = spawn_worker_with(WorkerConfig::new("w", 32, 0, ctx), Box::new(s.session(true)), SpawnOptions { stack_size: 8 << 20, ..Default::default() })
        .unwrap()
        .wait();
    let cold = spawn_worker_with(WorkerConfig::new("c", 32, 0, ctx), Box::new(s.session(false)), SpawnOptions { stack_size: 8 << 20, ..Default::default() })
        .unwrap()
        .wait();
    let (w, c) = (warm.expect("warm"), cold.expect("cold"));
    assert_eq!(w.output, c.output, "seq_cp prefix must give the same greedy output");
    assert!(w.latency.prefilled_tokens < c.latency.prefilled_tokens);
    assert_eq!(c.latency.prefilled_tokens, w.latency.prefilled_tokens + s.parent_tokens());
    assert_eq!(w.commands.len(), 1);
}

#[test]
fn llama_concurrent_children_release_their_sequences() {
    let Some(s) = shared() else { return };
    let kids = ["Task: open the log file.\nCommand:", "Task: read the config.\nCommand:", "Task: fix the bug.\nCommand:", "Task: free the cache.\nCommand:"];
    for _round in 0..2 {
        let hs: Vec<_> = kids
            .iter()
            .enumerate()
            .map(|(i, k)| spawn_worker_with(WorkerConfig::new(format!("k{}", i), 32, 0, *k), Box::new(s.session(true)), SpawnOptions { stack_size: 8 << 20, ..Default::default() }).unwrap())
            .collect();
        for h in hs {
            // Ok or a contract/parse error: either way the seq must come back
            let _ = h.wait();
        }
    }
    // 4 slots, 8 sessions run: would fail with "no free sequence slot" if leaked
}

#[test]
fn llama_budget_and_cancel() {
    let Some(s) = shared() else { return };
    let r = spawn_worker_with(WorkerConfig::new("b", 2, 0, "Task: fix the bug.\nCommand:"), Box::new(s.session(true)), SpawnOptions { stack_size: 8 << 20, ..Default::default() }).unwrap().wait();
    assert!(matches!(r, Err(WorkerError::BudgetExhausted { tokens: 2, .. })));
    let r = spawn_worker_with(WorkerConfig::new("t", 64, 1, "Task: fix the bug.\nCommand:"), Box::new(s.session(true)), SpawnOptions { stack_size: 8 << 20, ..Default::default() }).unwrap().wait();
    assert!(matches!(r, Err(WorkerError::Timeout { .. })), "{:?}", r);
}
