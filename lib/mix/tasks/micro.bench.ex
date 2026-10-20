defmodule Mix.Tasks.Micro.Bench do
  @shortdoc "BEAM micro-agent benchmarks (memory, spawn+validate+terminate, throughput)"
  @moduledoc """
      mix micro.bench [--sizes 1000,10000,100000] [--out bench/results/beam.json]
  """
  use Mix.Task

  @impl true
  def run(args) do
    {o, _, _} = OptionParser.parse(args, strict: [sizes: :string, out: :string])
    Mix.Task.run("app.start")
    sizes = (o[:sizes] || "1000,10000,100000") |> String.split(",") |> Enum.map(&String.to_integer/1)
    out = o[:out] || "bench/results/beam.json"
    r = MicroAgents.Bench.all(sizes)
    File.mkdir_p!(Path.dirname(out))
    File.write!(out, JSON.encode!(jsonable(r)))
    IO.inspect(r, limit: :infinity, pretty: true)
    Mix.shell().info("wrote #{out}")
  end

  @doc false
  def jsonable(m) when is_map(m) and not is_struct(m), do: Map.new(m, fn {k, v} -> {key(k), jsonable(v)} end)
  def jsonable(l) when is_list(l), do: Enum.map(l, &jsonable/1)
  def jsonable(t) when is_tuple(t), do: t |> Tuple.to_list() |> Enum.map(&jsonable/1)
  def jsonable(a) when is_atom(a) and a not in [nil, true, false], do: Atom.to_string(a)
  def jsonable(v), do: v

  defp key(k) when is_binary(k), do: k
  defp key(k) when is_atom(k), do: Atom.to_string(k)
  defp key(k), do: inspect(k)
end
