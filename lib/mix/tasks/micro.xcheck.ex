defmodule Mix.Tasks.Micro.Xcheck do
  @shortdoc "Cross-check native Elixir validator vs the sovereign-cmd Rust CLI"
  @moduledoc "    mix micro.xcheck [--random 2000] [--out bench/results/xcheck.json]"
  use Mix.Task

  @impl true
  def run(args) do
    {o, _, _} = OptionParser.parse(args, strict: [random: :integer, out: :string])
    Mix.Task.run("app.start")
    out = o[:out] || "bench/results/xcheck.json"
    corpus = MicroAgents.Corpus.all(o[:random] || 2000)
    {us, r} = :timer.tc(fn -> MicroAgents.CrossCheck.run(corpus) end)
    r = Map.put(r, :wall_ms, us / 1000) |> Map.put(:rust_cli, MicroAgents.Validator.RustPort.bin())
    File.mkdir_p!(Path.dirname(out))
    File.write!(out, JSON.encode!(Mix.Tasks.Micro.Bench.jsonable(r)))
    IO.inspect(Map.delete(r, :disagreements) |> Map.put(:disagreements, length(r.disagreements)), pretty: true)
    if r.disagreements != [], do: Mix.raise("disagreements: #{inspect(Enum.take(r.disagreements, 5))}")
  end
end
