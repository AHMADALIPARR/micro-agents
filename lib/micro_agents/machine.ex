defmodule MicroAgents.Machine do
  @moduledoc """
  The main state machine owned by the Coordinator. Pure data + pure transitions;
  only `MicroAgents.Coordinator` calls `apply_commands/2`, and only for results of
  workers that exited `:normal` after sending an approved payload.

  `chain` is a SHA-256 hash chain over every applied command batch, so two machines
  are equal iff they applied the same accepted sequences in the same order.
  """
  defstruct phase: :open, version: 0, applied: 0, chain: <<0::256>>, recent: []

  @recent 16

  @type t :: %__MODULE__{
          phase: :open | :sealed,
          version: non_neg_integer,
          applied: non_neg_integer,
          chain: <<_::256>>,
          recent: [binary]
        }

  def new, do: %__MODULE__{}

  @spec apply_commands(t, [binary]) :: {:ok, t} | {:error, :sealed}
  def apply_commands(%__MODULE__{phase: :sealed}, _cmds), do: {:error, :sealed}

  def apply_commands(%__MODULE__{} = m, cmds) when is_list(cmds) do
    chain = :crypto.hash(:sha256, [m.chain | Enum.map(cmds, &[&1, ?\n])])

    {:ok,
     %__MODULE__{
       m
       | version: m.version + 1,
         applied: m.applied + length(cmds),
         chain: chain,
         recent: Enum.take(Enum.reverse(cmds, m.recent), @recent)
     }}
  end

  def seal(%__MODULE__{} = m), do: %__MODULE__{m | phase: :sealed}
end
