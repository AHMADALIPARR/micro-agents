defmodule MicroAgents.LLM.Server do
  @moduledoc """
  Starts/stops a `llama-server` OS process (reused from the existing
  `/workspace/llama.cpp/build/bin` build; nothing is rebuilt) through an Erlang
  Port, with `-np N` parallel slots, `--kv-unified` and `--grammar-file`.

  The Port is owned by the calling process; `stop/1` closes it and kills the OS
  process.
  """

  @defaults [
    bin: "/workspace/llama.cpp/build/bin/llama-server",
    model: "/workspace/gguf-lab-october/runs/small-cpu-v3/small-v3-final.gguf",
    grammar: "/workspace/sovereign-cmd/grammar/commands.llama.gbnf",
    slots: 8,
    ctx: 2048,
    threads: 8,
    port: 18431,
    kv_unified: true,
    extra: []
  ]

  def defaults, do: @defaults

  def start(opts \\ []) do
    o = Keyword.merge(@defaults, opts)

    for k <- [:bin, :model, :grammar], not File.exists?(o[k]) do
      raise "#{k} not found: #{o[k]}"
    end

    args =
      [
        "-m", o[:model],
        "-np", to_string(o[:slots]),
        "-c", to_string(o[:ctx]),
        "-t", to_string(o[:threads]),
        "--grammar-file", o[:grammar],
        "--host", "127.0.0.1",
        "--port", to_string(o[:port]),
        if(o[:kv_unified], do: "--kv-unified", else: "--no-kv-unified")
      ] ++ o[:extra]

    # Watchdog: the shell starts llama-server in the background and blocks on
    # stdin. When the Port closes (or the BEAM dies) stdin hits EOF and the shell
    # kills the server, so an OS process can never outlive its owner.
    wrapper = ~s("$0" "$@" & pid=$!; echo "LLAMA_PID $pid"; trap 'kill $pid 2>/dev/null' EXIT; ) <>
              ~s(cat >/dev/null; kill $pid 2>/dev/null; wait $pid)

    port =
      Port.open({:spawn_executable, "/bin/sh"}, [
        :binary, :exit_status, :stderr_to_stdout, args: ["-c", wrapper, o[:bin] | args]
      ])

    os_pid = await_pid(port)
    url = "http://127.0.0.1:#{o[:port]}"

    case wait_healthy(url, port, 300) do
      :ok -> {:ok, %{port: port, os_pid: os_pid, url: url, opts: o, args: args}}
      {:error, r} -> stop(%{port: port, os_pid: os_pid}) && {:error, r}
    end
  end

  def stop(%{port: port, os_pid: os_pid}) do
    try do
      Port.close(port)
    rescue
      _ -> :ok
    end

    System.cmd("kill", ["-TERM", to_string(os_pid)], stderr_to_stdout: true)
    true
  end

  defp await_pid(port, buf \\ "") do
    receive do
      {^port, {:data, d}} ->
        buf = buf <> d

        case Regex.run(~r/LLAMA_PID (\d+)/, buf) do
          [_, pid] -> String.to_integer(pid)
          nil -> await_pid(port, buf)
        end
    after
      5_000 -> raise "llama-server wrapper did not report a pid"
    end
  end

  defp wait_healthy(_url, _port, 0), do: {:error, :health_timeout}

  defp wait_healthy(url, port, n) do
    receive do
      {^port, {:exit_status, s}} -> {:error, {:exited, s}}
    after
      100 ->
        case MicroAgents.LLM.Client.get(url <> "/health") do
          {:ok, 200, _} -> :ok
          _ -> wait_healthy(url, port, n - 1)
        end
    end
  end
end
