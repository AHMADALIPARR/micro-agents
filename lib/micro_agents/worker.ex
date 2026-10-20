defmodule MicroAgents.Worker do
  @moduledoc """
  An ephemeral micro-agent.

  Life cycle (one message out, then death):

    1. created with its immutable `ContextSlice` and an input (a candidate command
       sequence) captured in the closure,
    2. validates against the grammar and the ExecutionContract,
    3. sends exactly one `{:micro_result, self(), payload}` to `reply_to`, then
       returns (exit reason `:normal`), or
    4. on a structural/contract failure in `:approve` mode it sends nothing and
       exits with `{:contract_violation, reason}` / `{:structural_violation, reason}`.

  A worker never touches shared state: it holds no locks, no ETS handles, no
  references to the Coordinator's state. Its only side effect is one message (or
  its exit signal), so a crash at any point cannot leave the main state machine
  half-updated.

  Two ways to start one:
    * `spawn_raw/4` - `:erlang.spawn_opt/2` with `[:monitor, {:min_heap_size, n}]`,
      i.e. the cost of a bare `spawn_monitor`. Not linked to anybody.
    * `start_supervised/4` - a `:temporary` child of the `MicroAgents.WorkerSup`
      Task.Supervisor (restart: :temporary, so a failed worker is never restarted).
  """
  alias MicroAgents.{Validator, ContextSlice}

  @doc "Spawn an unlinked, monitored worker. Returns `{pid, monitor_ref}`."
  def spawn_raw(reply_to, %ContextSlice{} = slice, input, opts \\ []) do
    heap = Keyword.get(opts, :min_heap_size, min_heap())
    backend = Keyword.get(opts, :backend, :native)

    :erlang.spawn_opt(
      fn -> run(reply_to, slice, input, backend) end,
      [:monitor, {:min_heap_size, heap}]
    )
  end

  @doc """
  Start a worker as a `:temporary` child of `MicroAgents.WorkerSup`. The worker
  blocks until it receives `{:go, ref}` so the caller can monitor it first (no
  lost exit reasons). Returns `{pid, monitor_ref}`.
  """
  def start_supervised(reply_to, %ContextSlice{} = slice, input, opts \\ []) do
    backend = Keyword.get(opts, :backend, :native)
    sup = Keyword.get(opts, :supervisor, MicroAgents.WorkerSup)
    go = make_ref()

    {:ok, pid} =
      Task.Supervisor.start_child(
        sup,
        fn ->
          receive do
            {:go, ^go} -> run(reply_to, slice, input, backend)
          end
        end,
        restart: :temporary
      )

    mref = Process.monitor(pid)
    send(pid, {:go, go})
    {pid, mref}
  end

  @doc false
  def run(reply_to, %ContextSlice{mode: mode} = slice, input, backend) do
    case Validator.validate(input, slice, backend) do
      {:ok, cmds} ->
        payload = if mode == :predicate, do: true, else: {:commands, cmds}
        send(reply_to, {:micro_result, self(), payload})
        :ok

      {:error, {:cli, status, msg}} ->
        # backend failure (not a verdict about the input): crash loudly
        exit({:backend_error, {status, msg}})

      {:error, {kind, reason}} ->
        if mode == :predicate do
          send(reply_to, {:micro_result, self(), false})
          :ok
        else
          # Self-terminate. No message is sent: the exit reason *is* the report.
          exit({violation(kind), reason})
        end
    end
  end

  defp violation(:contract), do: :contract_violation
  defp violation(:structural), do: :structural_violation

  def min_heap, do: Application.get_env(:micro_agents, :worker_min_heap_words, 1)
end
