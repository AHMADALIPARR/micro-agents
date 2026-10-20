defmodule MicroAgents.Bench do
  @moduledoc """
  Measured-only benchmarks for the BEAM side. Every number comes from a run on
  the current machine; nothing is extrapolated.
  """
  alias MicroAgents.{ContextSlice, Coordinator, Worker}

  # Mixed workload: 50% valid, 25% contract violations, 25% structural failures.
  @inputs [
    "WRITE 0x2000 --force 42\n",
    "RUN /opt/sovereign/bin/agent --dry-run\n",
    "WRITE 0xFFF 1\n",
    "READ x  y\n"
  ]
  def inputs, do: @inputs

  defp slice, do: ContextSlice.new(id: :bench, allowed_verbs: nil)

  # ---------------------------------------------------------------------------
  # Memory
  # ---------------------------------------------------------------------------

  @doc "Per-process memory (bytes, from :erlang.process_info(pid, :memory))."
  def memory(n \\ 1000) do
    s = ContextSlice.new(id: 1, allowed_verbs: [:write, :run], max_instructions: 4, prompt: :binary.copy("p", 4096))
    input = hd(@inputs)
    heap = Worker.min_heap()

    bare = measure_idle(n, fn -> :erlang.spawn_opt(fn -> receive do: (:stop -> :ok) end, [{:min_heap_size, heap}]) end)

    raw_worker =
      measure_idle(n, fn ->
        :erlang.spawn_opt(
          fn -> receive do: (:stop -> Worker.run(self(), s, input, :native)) end,
          [{:min_heap_size, heap}]
        )
      end)

    sup_worker =
      measure_idle(n, fn ->
        {:ok, pid} =
          Task.Supervisor.start_child(MicroAgents.WorkerSup, fn -> receive do: (:stop -> :ok) end, restart: :temporary)

        pid
      end)

    hibernated = measure_idle(n, fn -> spawn(fn -> :erlang.hibernate(__MODULE__, :wait_stop, []) end) end)

    # Peak memory of a worker right after it validated (before it exits).
    parent = self()

    peak =
      for inp <- @inputs, into: %{} do
        :erlang.spawn_opt(
          fn ->
            _ = MicroAgents.Validator.validate(inp, s, :native)
            {:memory, m} = :erlang.process_info(self(), :memory)
            send(parent, {:peak, m})
          end,
          [{:min_heap_size, heap}]
        )

        receive do: ({:peak, m} -> {String.trim(inp), m})
      end

    %{
      wordsize: :erlang.system_info(:wordsize),
      min_heap_size_words_requested: heap,
      idle_bare_spawn_opt: bare,
      idle_raw_worker_with_slice: raw_worker,
      idle_supervised_task_worker: sup_worker,
      hibernated_process: hibernated,
      worker_after_validation_bytes: peak,
      note:
        "the 4 KiB prompt in the slice is an off-heap refc binary: only a ProcBin handle is in each worker heap"
    }
  end

  def wait_stop, do: receive(do: (:stop -> :ok))

  defp measure_idle(n, spawner) do
    :erlang.garbage_collect()
    before = :erlang.memory(:processes)
    pids = for _ <- 1..n, do: spawner.()
    Process.sleep(50)
    after_ = :erlang.memory(:processes)
    mems = for p <- pids, do: elem(:erlang.process_info(p, :memory), 1)
    Enum.each(pids, &send(&1, :stop))
    wait_dead(pids)

    %{
      per_process_bytes_min: Enum.min(mems),
      per_process_bytes_max: Enum.max(mems),
      per_process_bytes_mean: Float.round(Enum.sum(mems) / n, 1),
      vm_processes_delta_per_process: Float.round((after_ - before) / n, 1)
    }
  end

  defp wait_dead(pids) do
    for p <- pids do
      r = Process.monitor(p)
      receive do: ({:DOWN, ^r, _, _, _} -> :ok)
    end
  end

  # ---------------------------------------------------------------------------
  # Spawn + validate + terminate
  # ---------------------------------------------------------------------------

  @doc """
  Sequential: one worker at a time; each iteration = spawn, validate, deliver
  result/exit, and the `:DOWN` (i.e. the process is gone). `mode` is `:raw`
  (spawn_opt + monitor) or `:supervised` (Task.Supervisor child, restart: :temporary).
  """
  def sequential(n, mode \\ :raw) do
    s = slice()
    inputs = List.to_tuple(@inputs)
    k = tuple_size(inputs)
    procs_before = length(Process.list())
    lat = :array.new(n, default: 0)

    {wall, lat} =
      :timer.tc(fn ->
        Enum.reduce(0..(n - 1), lat, fn i, acc ->
          t0 = System.monotonic_time()
          input = elem(inputs, rem(i, k))

          {pid, mref} =
            case mode do
              :raw -> Worker.spawn_raw(self(), s, input)
              :supervised -> Worker.start_supervised(self(), s, input)
            end

          receive do
            {:DOWN, ^mref, :process, ^pid, _reason} -> :ok
          end

          # drop the result message if one was sent
          receive do
            {:micro_result, ^pid, _} -> :ok
          after
            0 -> :ok
          end

          :array.set(i, System.monotonic_time() - t0, acc)
        end)
      end)

    lats = lat |> :array.to_list() |> Enum.map(&System.convert_time_unit(&1, :native, :nanosecond))
    Process.sleep(10)

    %{
      kind: :sequential,
      mode: mode,
      n: n,
      wall_ms: wall / 1000,
      mean_us: Float.round(wall / n, 3),
      throughput_per_s: round(n / (wall / 1_000_000)),
      latency_us: percentiles(lats),
      leaked_processes: length(Process.list()) - procs_before
    }
  end

  @doc """
  Through the Coordinator: `run_batch` with up to `max_concurrency` live workers.
  Every accepted payload mutates the machine (hash chain), so this measures
  the full path, not just spawning.
  """
  def coordinator(n, opts \\ []) do
    s = slice()
    inputs = @inputs
    k = length(inputs)
    items = for i <- 0..(n - 1), do: {s, Enum.at(inputs, rem(i, k))}
    :ok = Coordinator.reset()
    procs_before = length(Process.list())
    m0 = Coordinator.machine()

    {wall, res} =
      :timer.tc(fn ->
        Coordinator.run_batch(items,
          collect: :summary,
          latencies: true,
          max_concurrency: Keyword.get(opts, :max_concurrency, 1024),
          spawn: Keyword.get(opts, :spawn, :raw)
        )
      end)

    m1 = Coordinator.machine()
    Process.sleep(10)

    %{
      kind: :coordinator_batch,
      spawn: Keyword.get(opts, :spawn, :raw),
      max_concurrency: Keyword.get(opts, :max_concurrency, 1024),
      n: n,
      wall_ms: wall / 1000,
      amortized_us_per_worker: Float.round(wall / n, 3),
      throughput_per_s: round(n / (wall / 1_000_000)),
      latency_us_spawn_to_down: percentiles(Enum.map(res.latencies_us, &(&1 * 1000))),
      summary: res.summary,
      machine_versions_applied: m1.version - m0.version,
      leaked_processes: length(Process.list()) - procs_before,
      coordinator_pending_after: Coordinator.pending_count()
    }
  end

  defp percentiles(ns_list) do
    sorted = Enum.sort(ns_list)
    n = length(sorted)
    t = List.to_tuple(sorted)
    p = fn q -> Float.round(elem(t, min(n - 1, trunc(q * n))) / 1000, 2) end
    %{p50: p.(0.50), p90: p.(0.90), p99: p.(0.99), max: Float.round(elem(t, n - 1) / 1000, 2)}
  end

  @doc "Run the full BEAM benchmark suite and return a map."
  def all(sizes \\ [1_000, 10_000, 100_000]) do
    # warm up the code paths
    _ = sequential(2_000, :raw)

    %{
      system: system(),
      memory: memory(),
      sequential_raw: Enum.map(sizes, &sequential(&1, :raw)),
      sequential_supervised: Enum.map(sizes, &sequential(&1, :supervised)),
      coordinator_raw: Enum.map(sizes, &coordinator(&1, spawn: :raw)),
      coordinator_supervised: Enum.map(sizes, &coordinator(&1, spawn: :supervised))
    }
  end

  def system do
    %{
      otp: :erlang.system_info(:otp_release) |> to_string(),
      erts: :erlang.system_info(:version) |> to_string(),
      elixir: System.version(),
      schedulers: :erlang.system_info(:schedulers_online),
      cpu: cpu_model(),
      date: DateTime.utc_now() |> DateTime.to_iso8601()
    }
  end

  defp cpu_model do
    case File.read("/proc/cpuinfo") do
      {:ok, s} ->
        case Regex.run(~r/model name\s*:\s*(.+)/, s) do
          [_, m] -> m
          _ -> "unknown"
        end

      _ ->
        "unknown"
    end
  end
end
