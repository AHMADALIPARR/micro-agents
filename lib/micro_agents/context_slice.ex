defmodule MicroAgents.ContextSlice do
  @moduledoc """
  The immutable slice of context a micro-agent receives at spawn time.

  It is captured by the worker closure, so the BEAM copies it into the worker's own
  heap when the process is created (a real copy, deliberately small). The one
  exception is `:prompt`: a binary larger than 64 bytes is a reference-counted
  off-heap binary, so only a ~48-byte ProcBin handle is copied and every worker
  shares the same bytes. That sharing is zero-copy.

  Fields:
    * `:id` - caller-chosen tag, echoed in the result
    * `:allowed_verbs` - `nil` (all) or a list of verb atoms
    * `:max_instructions` - `nil` or a positive integer
    * `:prompt` - optional shared parent prompt (binary), never mutated
    * `:mode` - `:approve` (payload = canonical command list, violations exit
      the worker) or `:predicate` (payload = boolean, no abnormal exit)
    * `:contract` - `:default` (Rust `ExecutionContract::default()`) or `:strict`
      (Rust `--strict-contract`)
  """
  defstruct id: nil, allowed_verbs: nil, max_instructions: nil, prompt: nil, mode: :approve, contract: :default

  @type t :: %__MODULE__{
          id: term,
          allowed_verbs: nil | [atom],
          max_instructions: nil | pos_integer,
          prompt: nil | binary,
          mode: :approve | :predicate,
          contract: :default | :strict
        }

  def new(opts \\ []), do: struct!(__MODULE__, opts)
end
