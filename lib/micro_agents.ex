defmodule MicroAgents do
  @moduledoc """
  Micro-agent spawning on the BEAM actor model. See README.md.

  Convenience wrappers around `MicroAgents.Coordinator`.
  """
  alias MicroAgents.{ContextSlice, Coordinator}

  @doc "Validate `input` in a fresh worker and apply it if approved."
  def run(input, slice_opts \\ [], opts \\ []) do
    Coordinator.run(ContextSlice.new(slice_opts), input, opts)
  end

  @doc "Ask a worker for a boolean verdict only (never mutates the machine)."
  def check(input, slice_opts \\ [], opts \\ []) do
    Coordinator.run(ContextSlice.new(Keyword.put(slice_opts, :mode, :predicate)), input, opts)
  end
end
