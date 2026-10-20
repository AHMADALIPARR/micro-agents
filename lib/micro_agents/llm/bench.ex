defmodule MicroAgents.LLM.Bench do
  @moduledoc """
  Measured LLM benchmarks against llama-server (HTTP, `-np N` slots) and the
  in-process seq_cp helper, plus the end-to-end pipeline: generated command lines
  -> micro-agent workers -> Coordinator.
  """
  alias MicroAgents.LLM.{Server, Client, SeqCp}
  alias MicroAgents.{Coordinator, ContextSlice}

  def parent, do: File.read!(Path.join(SeqCp.app_root(), "priv/llm/parent.txt"))

  def children do
    Path.join(SeqCp.app_root(), "priv/llm/children.txt")
    |> File.read!()
    |> String.split("\n", trim: true)
    |> Enum.map(&String.replace(&1, "\\n", "\n"))
  end

  defp body(prompt, extra) do
    Map.merge(
      %{prompt: prompt, n_predict: 32, temperature: 0, stop: ["\n"], cache_prompt: false},
      Map.new(extra)
    )
  end

  defp row(i, {:ok, r}) do
    t = r.timings || %{}

    %{
      child: i,
      ttft_ms: r.ttft_us && r.ttft_us / 1000,
      total_ms: r.total_us / 1000,
      prompt_n: t["prompt_n"],
      cache_n: t["cache_n"],
      prompt_ms: t["prompt_ms"],
      predicted_n: t["predicted_n"],
      predicted_ms: t["predicted_ms"],
      predicted_per_second: t["predicted_per_second"],
      content: r.content
    }
  end

  defp row(i, {:error, e}), do: %{child: i, error: inspect(e)}

  defp seq(url, kids, p, extra_fun) do
    for {s, i} <- Enum.with_index(kids), do: row(i, Client.stream_completion(url, body(p <> s, extra_fun.(i))))
  end

  defp conc(url, kids, p, extra_fun) do
    {wall, rows} =
      :timer.tc(fn ->
        kids
        |> Enum.with_index()
        |> Enum.map(fn {s, i} -> Task.async(fn -> row(i, Client.stream_completion(url, body(p <> s, extra_fun.(i)))) end) end)
        |> Enum.map(&Task.await(&1, 120_000))
      end)

    %{wall_ms: wall / 1000, rows: rows}
  end

  defp prime(url, p, n) do
    {us, _} =
      :timer.tc(fn ->
        for i <- 0..(n - 1) do
          {:ok, _} = Client.completion(url, body(p, n_predict: 0, cache_prompt: true, id_slot: i))
        end
      end)

    us / 1000
  end

  defp summarize(rows) do
    ok = Enum.filter(rows, &Map.has_key?(&1, :ttft_ms))
    ttft = Enum.map(ok, & &1.ttft_ms) |> Enum.reject(&is_nil/1)
    pn = Enum.map(ok, & &1.prompt_n) |> Enum.reject(&is_nil/1)
    pps = Enum.map(ok, & &1.predicted_per_second) |> Enum.reject(&is_nil/1)

    %{
      ttft_ms_mean: mean(ttft),
      ttft_ms_min: ttft != [] && Enum.min(ttft),
      ttft_ms_max: ttft != [] && Enum.max(ttft),
      prompt_tokens_evaluated_total: Enum.sum(pn),
      server_decode_tok_per_s_mean: mean(pps),
      errors: length(rows) - length(ok)
    }
  end

  defp mean([]), do: nil
  defp mean(l), do: Float.round(Enum.sum(l) / length(l), 3)

  @doc "llama-server scenarios. Starts and stops its own server."
  def server(opts \\ []) do
    kids = children()
    p = parent()
    n = length(kids)
    {:ok, srv} = Server.start(Keyword.merge([slots: n], opts))
    url = srv.url

    try do
      # warm-up: allocate buffers (not reported)
      _ = Client.completion(url, body(p <> hd(kids), []))

      cold_seq = seq(url, kids, p, fn i -> [id_slot: i, cache_prompt: false] end)
      prime_ms = prime(url, p, n)
      warm_seq = seq(url, kids, p, fn i -> [id_slot: i, cache_prompt: true] end)

      # auto slot selection (no id_slot): lets the server pick a slot by prompt
      # similarity and/or restore the prefix from its host-RAM prompt cache
      _ = Client.completion(url, body(p, n_predict: 0, cache_prompt: true))
      # distinct suffixes so no slot can hold this exact prompt from an earlier run:
      # only the parent prefix is reusable
      kids_auto = Enum.map(kids, &String.replace(&1, "Task:", "Next task:"))
      warm_auto = seq(url, kids_auto, p, fn _ -> [cache_prompt: true] end)

      # host-RAM prompt cache / cross-slot reuse probe: only slot 0 is primed, the
      # child goes to slot 1
      _ = Client.completion(url, body("unrelated filler prompt to evict.", id_slot: 1, cache_prompt: true, n_predict: 0))
      _ = Client.completion(url, body(p, id_slot: 0, n_predict: 0, cache_prompt: true))
      cross = row(0, Client.stream_completion(url, body(p <> hd(kids), id_slot: 1, cache_prompt: true)))

      cold_conc = conc(url, kids, p, fn i -> [id_slot: i, cache_prompt: false] end)
      _ = prime(url, p, n)
      warm_conc = conc(url, kids, p, fn i -> [id_slot: i, cache_prompt: true] end)

      # n_cmpl: one prompt, N completions; children receive the parent's KV via
      # seq_cp inside the server (cells shared under --kv-unified)
      first = p <> hd(kids)

      {ncmpl_us, {:ok, ncmpl}} =
        :timer.tc(fn -> Client.completion(url, body(first, n_cmpl: n, cache_prompt: false)) end)

      same_cold =
        conc(url, List.duplicate(hd(kids), n), p, fn i -> [id_slot: i, cache_prompt: false] end)

      %{
        server_args: srv.args,
        slots: n,
        cold_sequential: %{summary: summarize(cold_seq), rows: cold_seq},
        prime_all_slots_ms: prime_ms,
        warm_slot_sequential: %{summary: summarize(warm_seq), rows: warm_seq},
        warm_auto_slot_sequential: %{summary: summarize(warm_auto), rows: warm_auto},
        cross_slot_probe: cross,
        cold_concurrent: %{wall_ms: cold_conc.wall_ms, summary: summarize(cold_conc.rows), rows: cold_conc.rows},
        warm_concurrent: %{wall_ms: warm_conc.wall_ms, summary: summarize(warm_conc.rows), rows: warm_conc.rows},
        n_cmpl: %{
          wall_ms: ncmpl_us / 1000,
          contents: Enum.map(List.wrap(ncmpl), & &1["content"]),
          same_prompt_x_n_concurrent_cold_wall_ms: same_cold.wall_ms
        }
      }
    after
      Server.stop(srv)
    end
  end

  @doc "Feed generated lines to micro-agent workers via the Coordinator."
  def pipeline(lines) do
    :ok = Coordinator.reset()
    items = for l <- lines, do: {ContextSlice.new(id: :llm), ensure_nl(l)}
    m0 = Coordinator.machine()
    {us, %{results: results, summary: sum}} = :timer.tc(fn -> Coordinator.run_batch(items) end)
    m1 = Coordinator.machine()

    %{
      lines: Enum.zip_with(lines, results, fn l, r -> %{line: l, outcome: inspect(r)} end),
      summary: sum,
      wall_us: us,
      machine_versions_applied: m1.version - m0.version
    }
  end

  defp ensure_nl(l), do: if(String.ends_with?(l, "\n"), do: l, else: l <> "\n")

  def all(opts \\ []) do
    seqcp_fast = SeqCp.run(sampler: :fast)
    seqcp_naive = SeqCp.run(sampler: :naive)
    # config 1: --kv-unified with llama-server defaults (cache-idle-slots on)
    srv = server(Keyword.merge([extra: []], opts))
    # config 2: --kv-unified --no-cache-idle-slots (slots keep their own prefix)
    srv_keep = server(Keyword.merge([extra: ["--no-cache-idle-slots"]], opts))

    lines =
      Enum.map(srv_keep.warm_slot_sequential.rows, &Map.get(&1, :content, "")) ++
        case seqcp_fast do
          {:ok, d} -> Enum.map(d["sequential"]["shared"], & &1["text"])
          _ -> []
        end

    %{
      system: MicroAgents.Bench.system(),
      seqcp_fast: unwrap(seqcp_fast),
      seqcp_naive: unwrap(seqcp_naive),
      server_kvu_default: srv,
      server_kvu_keep_slots: srv_keep,
      pipeline: pipeline(lines)
    }
  end

  defp unwrap({:ok, d}), do: d
  defp unwrap(e), do: %{error: inspect(e)}
end
