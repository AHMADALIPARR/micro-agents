defmodule MicroAgents.Application do
  @moduledoc """
  Supervision tree:

      MicroAgents.Supervisor (one_for_one)
      ├── MicroAgents.WorkerSup   Task.Supervisor; ephemeral workers, restart: :temporary
      └── MicroAgents.Coordinator GenServer; owns the main state machine

  The Task.Supervisor is started first so the Coordinator can always start
  supervised workers. Workers are never restarted: a failed micro-agent is a
  verdict, not a fault to recover from.
  """
  use Application

  @impl true
  def start(_type, _args) do
    MicroAgents.LogFilter.install()

    children = [
      {Task.Supervisor, name: MicroAgents.WorkerSup},
      MicroAgents.Coordinator
    ]

    Supervisor.start_link(children, strategy: :one_for_one, name: MicroAgents.Supervisor)
  end
end
