defmodule Mix.Tasks.Micro.LlmBench do
  @shortdoc "LLM benchmarks: llama-server slots + in-process seq_cp helper + pipeline"
  @moduledoc """
      mix micro.llm_bench [--out bench/results/llm.json]

  Needs /workspace/llama.cpp/build (llama-server, libllama.so), the GGUF model and
  a C compiler for c_src/seqcp_bench.c (built into priv/ on first use).
  """
  use Mix.Task

  @impl true
  def run(args) do
    {o, _, _} = OptionParser.parse(args, strict: [out: :string])
    Mix.Task.run("app.start")
    out = o[:out] || "bench/results/llm.json"
    r = MicroAgents.LLM.Bench.all()
    File.mkdir_p!(Path.dirname(out))
    File.write!(out, JSON.encode!(Mix.Tasks.Micro.Bench.jsonable(r)))
    Mix.shell().info("wrote #{out}")
  end
end
