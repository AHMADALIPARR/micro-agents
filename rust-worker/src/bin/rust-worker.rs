//! Line-delimited JSON worker host (used by the Elixir Codex.Worker Port).
//!
//! stdin, one JSON object per line:
//!   {"op":"task","task_id":"t1","budget_tokens":32,"deadline_ms":5000,"context":"...",
//!    "backend":"mock"|"llama", "mock_output":"READ x\n", "token_delay_ms":0,
//!    "share_parent":true, "strict":false}
//!   {"op":"cancel","task_id":"t1"}      (or the bare line: cancel  -> cancels all)
//! stdout, one JSON object per line:
//!   {"event":"ready","pid":N}
//!   {"event":"spawned","task_id":..,"worker_id":N}
//!   {"event":"result","task_id":..,"worker_id":N,"output":..,"commands":[..],"tokens":N,
//!    "spawn_latency_us":..,"first_token_latency_us":..,"generation_us":..,"total_us":..,
//!    "prefilled_tokens":N,"tokens_per_sec":..}
//!   {"event":"error","task_id":..,"worker_id":N,"kind":"cancelled|timeout|budget_exhausted|parse|contract_violation|backend","message":..}
//!   {"event":"cancel_ack","task_id":..,"found":bool}
//! On stdin EOF every live worker is cancelled and the process exits, so closing
//! the Port (or the owning Erlang process dying) never leaves an orphan.
//!
//! Flags: --parent-file F (shared llama parent prompt), --model M, --grammar G,
//!        --slots N (max concurrent llama children), --threads T

use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rust_worker::json::{esc, esc_list, parse_object, Val};
use rust_worker::{cancel_worker, spawn_worker_with, InferenceBackend, MockBackend, SpawnOptions, WorkerConfig, WorkerId};

type Out = Arc<Mutex<std::io::Stdout>>;

fn emit(out: &Out, line: String) {
    let mut o = out.lock().unwrap();
    let _ = writeln!(o, "{}", line);
    let _ = o.flush();
}

struct Args {
    parent_file: Option<String>,
    model: Option<String>,
    grammar: Option<String>,
    slots: usize,
    threads: u32,
}

fn args() -> Args {
    let mut a = Args { parent_file: None, model: None, grammar: None, slots: 8, threads: 8 };
    let v: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < v.len() {
        let next = v.get(i + 1).cloned();
        match v[i].as_str() {
            "--parent-file" => a.parent_file = next,
            "--model" => a.model = next,
            "--grammar" => a.grammar = next,
            "--slots" => a.slots = next.and_then(|s| s.parse().ok()).unwrap_or(8),
            "--threads" => a.threads = next.and_then(|s| s.parse().ok()).unwrap_or(8),
            "-h" | "--help" => {
                println!("see src/bin/rust-worker.rs header for the protocol");
                std::process::exit(0)
            }
            other => {
                eprintln!("unknown arg {}", other);
                std::process::exit(64)
            }
        }
        i += 2;
    }
    a
}

#[cfg(feature = "llama")]
struct Llama(Option<Arc<rust_worker::llama::LlamaShared>>);
#[cfg(not(feature = "llama"))]
struct Llama(Option<()>);

fn main() {
    let a = args();
    let out: Out = Arc::new(Mutex::new(std::io::stdout()));
    let tasks: Arc<Mutex<HashMap<String, WorkerId>>> = Arc::new(Mutex::new(HashMap::new()));
    let mut llama = Llama(None);
    emit(&out, format!("{{\"event\":\"ready\",\"pid\":{}}}", std::process::id()));

    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "cancel" {
            for (_, id) in tasks.lock().unwrap().iter() {
                cancel_worker(*id);
            }
            continue;
        }
        let m = match parse_object(line) {
            Ok(m) => m,
            Err(e) => {
                emit(&out, format!("{{\"event\":\"error\",\"task_id\":null,\"kind\":\"protocol\",\"message\":{}}}", esc(&e)));
                continue;
            }
        };
        let s = |k: &str| m.get(k).and_then(Val::as_str).map(str::to_string);
        let n = |k: &str, d: u64| m.get(k).and_then(Val::as_u64).unwrap_or(d);
        let task_id = s("task_id").unwrap_or_default();
        match s("op").as_deref() {
            Some("cancel") => {
                let found = tasks.lock().unwrap().get(&task_id).map(|id| cancel_worker(*id)).unwrap_or(false);
                emit(&out, format!("{{\"event\":\"cancel_ack\",\"task_id\":{},\"found\":{}}}", esc(&task_id), found));
            }
            Some("task") => {
                let cfg = WorkerConfig {
                    task_id: task_id.clone(),
                    budget_tokens: n("budget_tokens", 64).min(u32::MAX as u64) as u32,
                    deadline_ms: n("deadline_ms", 0),
                    context: s("context").unwrap_or_default(),
                };
                let backend: Result<Box<dyn InferenceBackend>, String> = match s("backend").as_deref().unwrap_or("mock") {
                    "mock" => Ok(Box::new(MockBackend::new(
                        &s("mock_output").unwrap_or_else(|| "READ /var/log/app.log\n".into()),
                        Duration::from_millis(n("token_delay_ms", 0)),
                    ))),
                    "llama" => make_llama(&a, &mut llama, m.get("share_parent").and_then(Val::as_bool).unwrap_or(true)),
                    other => Err(format!("unknown backend {}", other)),
                };
                let backend = match backend {
                    Ok(b) => b,
                    Err(e) => {
                        emit(&out, format!("{{\"event\":\"error\",\"task_id\":{},\"kind\":\"backend\",\"message\":{}}}", esc(&task_id), esc(&e)));
                        continue;
                    }
                };
                let opts = SpawnOptions {
                    strict_contract: m.get("strict").and_then(Val::as_bool).unwrap_or(false),
                    // llama decode needs a real stack
                    stack_size: if s("backend").as_deref() == Some("llama") { 8 << 20 } else { 256 << 10 },
                };
                match spawn_worker_with(cfg, backend, opts) {
                    Ok(h) => {
                        tasks.lock().unwrap().insert(task_id.clone(), h.id);
                        emit(&out, format!("{{\"event\":\"spawned\",\"task_id\":{},\"worker_id\":{}}}", esc(&task_id), h.id.0));
                        let out2 = out.clone();
                        let tasks2 = tasks.clone();
                        let tid = task_id.clone();
                        let wid = h.id.0;
                        std::thread::spawn(move || {
                            let r = h.wait();
                            tasks2.lock().unwrap().remove(&tid);
                            let line = match r {
                                Ok(r) => format!(
                                    "{{\"event\":\"result\",\"task_id\":{},\"worker_id\":{},\"output\":{},\"commands\":{},\"tokens\":{},\"spawn_latency_us\":{},\"first_token_latency_us\":{},\"generation_us\":{},\"total_us\":{},\"prefilled_tokens\":{},\"tokens_per_sec\":{:.2}}}",
                                    esc(&tid), wid, esc(&r.output), esc_list(&r.commands), r.tokens_generated,
                                    r.latency.spawn_latency_us, r.latency.first_token_latency_us, r.latency.generation_us,
                                    r.latency.total_us, r.latency.prefilled_tokens, r.latency.tokens_per_sec(r.tokens_generated)
                                ),
                                Err(e) => format!(
                                    "{{\"event\":\"error\",\"task_id\":{},\"worker_id\":{},\"kind\":\"{}\",\"message\":{}}}",
                                    esc(&tid), wid, e.kind(), esc(&e.to_string())
                                ),
                            };
                            emit(&out2, line);
                        });
                    }
                    Err(e) => emit(&out, format!("{{\"event\":\"error\",\"task_id\":{},\"kind\":\"backend\",\"message\":{}}}", esc(&task_id), esc(&e.to_string()))),
                }
            }
            _ => emit(&out, format!("{{\"event\":\"error\",\"task_id\":{},\"kind\":\"protocol\",\"message\":\"unknown op\"}}", esc(&task_id))),
        }
    }
    // stdin closed: the owner is gone. Cancel everything and exit immediately.
    for (_, id) in tasks.lock().unwrap().iter() {
        cancel_worker(*id);
    }
    std::process::exit(0);
}

#[cfg(feature = "llama")]
fn make_llama(a: &Args, l: &mut Llama, share: bool) -> Result<Box<dyn InferenceBackend>, String> {
    use rust_worker::llama::{LlamaConfig, LlamaShared};
    if l.0.is_none() {
        let parent = match &a.parent_file {
            Some(f) => std::fs::read_to_string(f).map_err(|e| e.to_string())?,
            None => String::new(),
        };
        let mut cfg = LlamaConfig::defaults(parent);
        if let Some(m) = &a.model { cfg.model = m.clone(); }
        if let Some(g) = &a.grammar { cfg.grammar_file = g.clone(); }
        cfg.n_children = a.slots;
        cfg.threads = a.threads;
        l.0 = Some(LlamaShared::load(&cfg)?);
    }
    Ok(Box::new(l.0.as_ref().unwrap().session(share)))
}

#[cfg(not(feature = "llama"))]
fn make_llama(_a: &Args, _l: &mut Llama, _share: bool) -> Result<Box<dyn InferenceBackend>, String> {
    Err("built without the `llama` feature".into())
}
