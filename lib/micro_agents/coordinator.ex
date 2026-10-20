defmodule MicroAgents.Coordinator do
  @moduledoc """
  GenServer owning the main state machine (`MicroAgents.Machine`).

  Invariants:
    * The machine is mutated in exactly one place (`finalize/3`) and only when a
      worker (a) sent an approved `{:commands, cmds}` payload **and** (b) then exited
      with reason `:normal`. A worker that sends a result and then crashes, or
      crashes without sending, never changes the machine.
    * Workers are monitored, never linked: a worker crash produces a `:DOWN`
      message, not an exit signal, so it cannot take the Coordinator down.
    * No locks exist. Workers only send messages; the Coordinator serialises them in
      its mailbox. There is nothing a dying worker could "hold".
    * Violations are recorded in `stats` / `violations` (bounded log), which are
      bookkeeping, not part of the machine.
    * A reply for a worker is only sent after its `:DOWN` arrives, so when
      `run/3` or `run_batch/2` returns, every worker it started is dead.
  """
  use GenServer
  alias MicroAgents.{Machine, Worker, ContextSlice}

  @log_max 100

  # ---- client API --------------------------------------------------------------

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: Keyword.get(opts, :name, __MODULE__))
  end

  @doc """
  Spawn one worker for `input` with `slice`, wait for it to terminate, and return
  its outcome:

    * `{:accepted, {:commands, cmds}}` - machine advanced
    * `{:predicate, boolean}` - predicate-mode answer (machine untouched)
    * `{:rejected, {:contract_violation, reason}}`
    * `{:rejected, {:structural_violation, reason}}`
    * `{:crashed, reason}` / `:timeout` / `{:discarded, :sealed}`

  Options: `:spawn` (`:raw` | `:supervised`, default `:raw`), `:backend`
  (`:native` | `:rust`), `:timeout` (worker deadline in ms), `:server`.
  """
  def run(%ContextSlice{} = slice, input, opts \\ []) do
    {server, opts} = Keyword.pop(opts, :server, __MODULE__)
    GenServer.call(server, {:run, slice, input, opts}, :infinity)
  end

  @doc """
  Run many `{slice, input}` items. At most `:max_concurrency` (default 1024)
  workers are alive at once. Returns `%{results: [...] | nil, summary: map,
  latencies_us: [...] | nil}`. Pass `collect: :summary` to skip per-item
  results and `latencies: true` to collect spawn->DOWN latency per worker.
  """
  def run_batch(items, opts \\ []) when is_list(items) do
    {server, opts} = Keyword.pop(opts, :server, __MODULE__)
    GenServer.call(server, {:run_batch, items, opts}, :infinity)
  end

  def machine(server \\ __MODULE__), do: GenServer.call(server, :machine)
  def stats(server \\ __MODULE__), do: GenServer.call(server, :stats)
  def violations(server \\ __MODULE__), do: GenServer.call(server, :violations)
  def pending_count(server \\ __MODULE__), do: GenServer.call(server, :pending_count)
  def seal(server \\ __MODULE__), do: GenServer.call(server, :seal)
  def reset(server \\ __MODULE__), do: GenServer.call(server, :reset)

  # ---- server --------------------------------------------------------------------

  @impl true
  def init(opts) do
    {:ok, fresh(opts)}
  end

  defp fresh(opts) do
    %{
      machine: Machine.new(),
      pending: %{},
      batches: %{},
      stats: %{
        spawned: 0,
        accepted: 0,
        predicate_true: 0,
        predicate_false: 0,
        contract_violations: 0,
        structural_violations: 0,
        crashes: 0,
        timeouts: 0,
        discarded: 0
      },
      violations: [],
      violations_len: 0,
      timeout: Keyword.get(opts, :worker_timeout, Application.get_env(:micro_agents, :worker_timeout_ms, 5_000)),
      opts: opts
    }
  end

  @impl true
  def handle_call({:run, slice, input, opts}, from, st) do
    {:noreply, start_worker(st, slice, input, opts, {:single, from})}
  end

  def handle_call({:run_batch, items, opts}, from, st) do
    bref = make_ref()
    maxc = Keyword.get(opts, :max_concurrency, 1024)
    collect = Keyword.get(opts, :collect, :results)

    batch = %{
      from: from,
      queue: Enum.with_index(items),
      remaining: length(items),
      results: if(collect == :results, do: %{}, else: nil),
      summary: %{},
      latencies: if(Keyword.get(opts, :latencies, false), do: [], else: nil),
      opts: opts
    }

    st = put_in(st.batches[bref], batch)

    if batch.remaining == 0 do
      {:noreply, maybe_finish_batch(st, bref)}
    else
      {:noreply, fill_batch(st, bref, maxc)}
    end
  end

  def handle_call(:machine, _from, st), do: {:reply, st.machine, st}
  def handle_call(:stats, _from, st), do: {:reply, st.stats, st}
  def handle_call(:violations, _from, st), do: {:reply, Enum.reverse(st.violations), st}
  def handle_call(:pending_count, _from, st), do: {:reply, map_size(st.pending), st}
  def handle_call(:seal, _from, st), do: {:reply, :ok, %{st | machine: Machine.seal(st.machine)}}

  def handle_call(:reset, _from, st) do
    if map_size(st.pending) == 0,
      do: {:reply, :ok, fresh(st.opts)},
      else: {:reply, {:error, :busy}, st}
  end

  @impl true
  def handle_info({:micro_result, pid, payload}, st) do
    case st.pending do
      %{^pid => w} -> {:noreply, put_in(st.pending[pid], %{w | payload: {:ok, payload}})}
      _ -> {:noreply, st}
    end
  end

  def handle_info({:DOWN, mref, :process, pid, reason}, st) do
    case st.pending do
      %{^pid => %{mref: ^mref} = w} ->
        {w_pending, pending} = {w, Map.delete(st.pending, pid)}
        cancel_timer(w_pending.timer)
        {:noreply, finalize(%{st | pending: pending}, w_pending, reason)}

      _ ->
        {:noreply, st}
    end
  end

  def handle_info({:timeout, _tref, {:worker_deadline, pid}}, st) do
    case st.pending do
      %{^pid => w} ->
        Process.exit(pid, :kill)
        {:noreply, put_in(st.pending[pid], %{w | timer: nil, timed_out: true})}

      _ ->
        {:noreply, st}
    end
  end

  def handle_info(_msg, st), do: {:noreply, st}

  # ---- internals -------------------------------------------------------------------

  defp start_worker(st, slice, input, opts, owner) do
    {pid, mref} =
      case Keyword.get(opts, :spawn, :raw) do
        :raw -> Worker.spawn_raw(self(), slice, input, opts)
        :supervised -> Worker.start_supervised(self(), slice, input, opts)
      end

    timer =
      case Keyword.get(opts, :timeout, st.timeout) do
        :infinity -> nil
        ms -> :erlang.start_timer(ms, self(), {:worker_deadline, pid})
      end

    w = %{
      mref: mref,
      timer: timer,
      owner: owner,
      payload: nil,
      timed_out: false,
      t0: System.monotonic_time()
    }

    %{st | pending: Map.put(st.pending, pid, w), stats: bump(st.stats, :spawned)}
  end

  defp cancel_timer(nil), do: :ok
  defp cancel_timer(t), do: :erlang.cancel_timer(t, async: true, info: false)

  # The single place the machine can change.
  defp finalize(st, w, reason) do
    {outcome, st} =
      case {reason, w.payload} do
        {:normal, {:ok, {:commands, cmds} = p}} ->
          case Machine.apply_commands(st.machine, cmds) do
            {:ok, m} -> {{:accepted, p}, %{st | machine: m, stats: bump(st.stats, :accepted)}}
            {:error, :sealed} -> {{:discarded, :sealed}, %{st | stats: bump(st.stats, :discarded)}}
          end

        {:normal, {:ok, b}} when is_boolean(b) ->
          key = if b, do: :predicate_true, else: :predicate_false
          {{:predicate, b}, %{st | stats: bump(st.stats, key)}}

        {{:contract_violation, _} = v, _} ->
          {{:rejected, v}, record(st, :contract_violations, v)}

        {{:structural_violation, _} = v, _} ->
          {{:rejected, v}, record(st, :structural_violations, v)}

        {:killed, _} when w.timed_out ->
          {:timeout, record(st, :timeouts, :timeout)}

        {other, _} ->
          r = if other == :normal, do: :no_result, else: other
          {{:crashed, r}, record(st, :crashes, {:crashed, r})}
      end

    deliver(st, w, outcome)
  end

  defp record(st, key, entry) do
    {log, len} =
      if st.violations_len >= @log_max,
        do: {[entry | Enum.take(st.violations, @log_max - 1)], @log_max},
        else: {[entry | st.violations], st.violations_len + 1}

    %{st | stats: bump(st.stats, key), violations: log, violations_len: len}
  end

  defp bump(stats, key), do: Map.update!(stats, key, &(&1 + 1))

  defp deliver(st, %{owner: {:single, from}}, outcome) do
    GenServer.reply(from, outcome)
    st
  end

  defp deliver(st, %{owner: {:batch, bref, idx}, t0: t0}, outcome) do
    b = st.batches[bref]
    tag = outcome_tag(outcome)

    b = %{
      b
      | remaining: b.remaining - 1,
        results: b.results && Map.put(b.results, idx, outcome),
        summary: Map.update(b.summary, tag, 1, &(&1 + 1)),
        latencies:
          b.latencies &&
            [System.convert_time_unit(System.monotonic_time() - t0, :native, :microsecond) | b.latencies]
    }

    st = put_in(st.batches[bref], b)
    st = fill_batch(st, bref, 1)
    maybe_finish_batch(st, bref)
  end

  defp outcome_tag({:accepted, _}), do: :accepted
  defp outcome_tag({:predicate, b}), do: {:predicate, b}
  defp outcome_tag({:rejected, {k, _}}), do: k
  defp outcome_tag({:crashed, _}), do: :crashed
  defp outcome_tag({:discarded, _}), do: :discarded
  defp outcome_tag(:timeout), do: :timeout

  defp fill_batch(st, _bref, 0), do: st

  defp fill_batch(st, bref, n) do
    b = st.batches[bref]

    case b.queue do
      [] ->
        st

      [{{slice, input}, idx} | rest] ->
        st = put_in(st.batches[bref].queue, rest)
        st = start_worker(st, slice, input, b.opts, {:batch, bref, idx})
        fill_batch(st, bref, n - 1)
    end
  end

  defp maybe_finish_batch(st, bref) do
    b = st.batches[bref]

    if b.remaining == 0 do
      results =
        b.results && Enum.map(0..(map_size(b.results) - 1)//1, &Map.fetch!(b.results, &1))

      GenServer.reply(b.from, %{results: results, summary: b.summary, latencies_us: b.latencies})
      %{st | batches: Map.delete(st.batches, bref)}
    else
      st
    end
  end
end
