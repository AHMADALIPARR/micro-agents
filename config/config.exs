import Config

config :micro_agents,
  # Path to the GBNF grammar the native parser mirrors (informational; the
  # native parser is a hand port, see MicroAgents.Grammar).
  grammar_file: "/workspace/sovereign-cmd/grammar/commands.gbnf",
  # Optional Rust CLI backend (sovereign-cmd). Overridable with
  # SOVEREIGN_CMD_BIN at runtime.
  rust_cli: "/workspace/sovereign-cmd/target/release/sovereign-cmd",
  # min_heap_size (words) for raw-spawned workers. The BEAM rounds this up to
  # the smallest heap size in its size table.
  worker_min_heap_words: 1,
  worker_timeout_ms: 5_000

config :logger, level: :warning

import_config "#{config_env()}.exs"
