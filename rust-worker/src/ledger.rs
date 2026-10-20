/// The "main state" on the Rust side. Workers never receive it; only the owner
/// applies *successful* results, so a failing worker cannot touch it.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Ledger {
    pub version: u64,
    pub applied: Vec<String>,
}

impl Ledger {
    pub fn apply(&mut self, r: &crate::types::WorkerResult) {
        self.version += 1;
        self.applied.extend(r.commands.iter().cloned());
    }
}
