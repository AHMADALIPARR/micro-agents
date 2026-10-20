defmodule MicroAgents.LogFilter do
  @moduledoc """
  Supervised (`Task.Supervisor`) workers that self-terminate with
  `{:contract_violation, _}` / `{:structural_violation, _}` would otherwise produce a
  Task crash report each. Those exits are expected verdicts, already recorded by the
  Coordinator, so this primary logger filter drops exactly those reports and nothing
  else.
  """
  def install do
    _ = :logger.remove_primary_filter(:micro_agents_verdicts)
    :logger.add_primary_filter(:micro_agents_verdicts, {&__MODULE__.filter/2, []})
  end

  def filter(%{msg: {:report, report}} = event, _) do
    if verdict_report?(report), do: :stop, else: event
  end

  def filter(event, _), do: event

  defp verdict_report?(%{label: {Task.Supervisor, :terminating}, report: %{reason: reason}}),
    do: verdict?(reason)

  defp verdict_report?(%{label: {:proc_lib, :crash}, report: [info | _]}) when is_list(info) do
    case List.keyfind(info, :error_info, 0) do
      {:error_info, {:exit, reason, _}} -> verdict?(reason)
      _ -> false
    end
  end

  defp verdict_report?(_), do: false

  defp verdict?({{tag, _}, _stack}) when tag in [:contract_violation, :structural_violation], do: true
  defp verdict?({tag, _}) when tag in [:contract_violation, :structural_violation], do: true
  defp verdict?(_), do: false
end
