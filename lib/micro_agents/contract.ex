defmodule MicroAgents.Contract do
  @moduledoc """
  ExecutionContract (pure Elixir mirror of the sovereign-cmd contract).

    * `WRITE` whose target is a hex address `< 0x1000` is rejected
      (`{:write_below_floor, addr}`).
    * `RUN` whose target is a file path not under `/opt/sovereign/bin/` is rejected
      (`{:run_outside_allowlist, path}`). Like Rust's component-wise
      `Path::starts_with`, the directory `/opt/sovereign/bin` itself matches.

  Two modes, identical to `sovereign_cmd::ExecutionContract`:
    * `:default` - exactly the rules above (Rust `ExecutionContract::default()`).
      Gap, kept on purpose so both sides agree: `RUN <identifier>` and `RUN <0xADDR>`
      pass.
    * `:strict` - Rust `--strict-contract`: also rejects a RUN target that is not a
      path (`:run_target_not_path`), `RUN /opt/sovereign/bin` itself
      (`:run_target_is_dir`), and WRITE with an address *argument* `< 0x1000`
      (`:write_arg_below_floor`).

  The slice selects the mode with `contract: :default | :strict`.

  Prefix matching is safe here because the grammar cannot produce `..`, `.` path
  segments, or `//`: a filepath is `/seg(/seg)*(.ext)?` with non-empty segments.

  The context slice may tighten (never loosen) the contract with
  `:allowed_verbs` and `:max_instructions`.
  """

  @write_floor 0x1000
  @run_prefix "/opt/sovereign/bin/"

  def write_floor, do: @write_floor
  def run_prefix, do: @run_prefix

  @spec check([MicroAgents.Grammar.instruction()], map) :: :ok | {:error, term}
  def check(instrs, slice \\ %{}) do
    with :ok <- check_count(instrs, slice) do
      check_each(instrs, slice, 0)
    end
  end

  defp check_count(instrs, %{max_instructions: max}) when is_integer(max) do
    n = length(instrs)
    if n <= max, do: :ok, else: {:error, {:too_many_instructions, n, max}}
  end

  defp check_count(_, _), do: :ok

  defp check_each([], _slice, _i), do: :ok

  defp check_each([instr | rest], slice, i) do
    case check_one(instr, slice) do
      :ok -> check_each(rest, slice, i + 1)
      {:error, reason} -> {:error, {i, reason}}
    end
  end

  @doc "Check a single instruction against the contract and the slice."
  def check_one({verb, target, args} = _instr, slice \\ %{}) do
    cond do
      not verb_allowed?(verb, slice) -> {:error, {:verb_not_allowed, verb}}
      true -> base(verb, target, args, Map.get(slice, :contract, :default))
    end
  end

  defp verb_allowed?(_verb, %{allowed_verbs: nil}), do: true
  defp verb_allowed?(verb, %{allowed_verbs: allowed}) when is_list(allowed), do: verb in allowed
  defp verb_allowed?(_verb, _), do: true

  @bin_dir "/opt/sovereign/bin"

  defp base(:write, {:addr, a}, _args, _mode) when a < @write_floor, do: {:error, {:write_below_floor, a}}

  defp base(:write, _target, args, :strict) do
    case Enum.find(args, &match?({:addr, x} when x < @write_floor, &1)) do
      nil -> :ok
      {:addr, x} -> {:error, {:write_arg_below_floor, x}}
    end
  end

  defp base(:run, {:path, <<@run_prefix, _::binary>>}, _args, _mode), do: :ok
  defp base(:run, {:path, @bin_dir}, _args, :default), do: :ok
  defp base(:run, {:path, @bin_dir}, _args, :strict), do: {:error, {:run_target_is_dir, @bin_dir}}
  defp base(:run, {:path, p}, _args, _mode), do: {:error, {:run_outside_allowlist, p}}
  defp base(:run, target, _args, :strict), do: {:error, {:run_target_not_path, target}}
  defp base(_, _, _, _), do: :ok
end
