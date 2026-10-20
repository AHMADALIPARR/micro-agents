defmodule MicroAgents.Validator do
  @moduledoc """
  Grammar + contract validation behind one function, with two backends:

    * `:native` (default) - `MicroAgents.Grammar` + `MicroAgents.Contract`, pure
      Elixir, runs inside the worker process.
    * `:rust` - `MicroAgents.Validator.RustPort`, which shells out to the
      sovereign-cmd CLI through an Erlang Port. Used for cross-checking; it costs an
      OS process per call, so it is not the per-worker default.
  """
  alias MicroAgents.{Grammar, Contract}

  @type verdict ::
          {:ok, [binary]}
          | {:error, {:structural, term}}
          | {:error, {:contract, term}}

  @spec validate(binary, map, :native | :rust) :: verdict
  def validate(input, slice \\ %{}, backend \\ :native)

  def validate(input, slice, :native) do
    case Grammar.parse(input) do
      {:ok, instrs} ->
        case Contract.check(instrs, slice) do
          :ok -> {:ok, Enum.map(instrs, &Grammar.canonical/1)}
          {:error, reason} -> {:error, {:contract, reason}}
        end

      {:error, err} ->
        {:error, {:structural, err}}
    end
  end

  def validate(input, slice, :rust) do
    MicroAgents.Validator.RustPort.validate(input, strict: Map.get(slice, :contract, :default) == :strict)
  end
end
