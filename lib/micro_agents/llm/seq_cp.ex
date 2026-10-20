defmodule MicroAgents.LLM.SeqCp do
  @moduledoc """
  Elixir side of `c_src/seqcp_bench.c`: one llama.cpp context, parent prompt
  prefilled once on seq 0, `llama_memory_seq_cp` to children (metadata-only with a
  unified KV cache), grammar-constrained greedy decode per child, compared with
  cold per-child prefill. The helper is linked against the prebuilt
  `/workspace/llama.cpp/build/bin/libllama.so`.
  """

  @app_root Path.expand("../../..", __DIR__)

  def helper, do: Path.join(@app_root, "priv/seqcp_bench")

  def build! do
    {out, s} = System.cmd("make", ["-C", Path.join(@app_root, "c_src")], stderr_to_stdout: true)
    if s != 0, do: raise("helper build failed:\n" <> out)
    helper()
  end

  def ensure_built! do
    if File.exists?(helper()), do: helper(), else: build!()
  end

  def run(opts \\ []) do
    bin = ensure_built!()
    llm = MicroAgents.LLM.Server.defaults()

    args = [
      "--model", Keyword.get(opts, :model, llm[:model]),
      "--grammar", Keyword.get(opts, :grammar, llm[:grammar]),
      "--parent", Keyword.get(opts, :parent, Path.join(@app_root, "priv/llm/parent.txt")),
      "--children", Keyword.get(opts, :children, Path.join(@app_root, "priv/llm/children.txt")),
      "--threads", to_string(Keyword.get(opts, :threads, 8)),
      "--n-predict", to_string(Keyword.get(opts, :n_predict, 32)),
      "--reps", to_string(Keyword.get(opts, :reps, 3)),
      "--sampler", to_string(Keyword.get(opts, :sampler, :fast))
    ]

    case System.cmd(bin, args, stderr_to_stdout: false) do
      {out, 0} -> {:ok, JSON.decode!(out)}
      {out, s} -> {:error, {s, out}}
    end
  end

  def app_root, do: @app_root
end
