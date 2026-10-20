defmodule MicroAgents.CrossCheck do
  @moduledoc """
  Run every corpus program through the native validator and the Rust CLI (both
  contract modes) and compare: verdict class (`:ok` / `:structural` / `:contract`)
  and, when accepted, the canonical command lines.
  """
  alias MicroAgents.{Validator, Corpus}
  alias MicroAgents.Validator.RustPort

  def run(corpus \\ Corpus.all(), modes \\ [:default, :strict]) do
    unless RustPort.available?(), do: raise("Rust CLI not found at #{RustPort.bin()}")

    rows =
      for mode <- modes, input <- corpus do
        slice = %{contract: mode}
        n = Validator.validate(input, slice, :native)
        r = Validator.validate(input, slice, :rust)
        {mode, input, class(n), class(r), agree?(n, r)}
      end

    disagreements = for {m, i, cn, cr, false} <- rows, do: %{mode: m, input: i, native: cn, rust: cr}

    by_class =
      rows
      |> Enum.frequencies_by(fn {m, _, c, _, _} -> {m, c} end)
      |> Enum.sort()

    %{cases: length(rows), agree: length(rows) - length(disagreements), disagreements: disagreements, by_class: by_class}
  end

  defp class({:ok, _}), do: :ok
  defp class({:error, {:structural, _}}), do: :structural
  defp class({:error, {:contract, _}}), do: :contract
  defp class(other), do: {:other, other}

  defp agree?({:ok, a}, {:ok, b}), do: a == b
  defp agree?(a, b), do: class(a) == class(b)
end
