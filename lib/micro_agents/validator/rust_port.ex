defmodule MicroAgents.Validator.RustPort do
  @moduledoc """
  Optional backend: run the `sovereign-cmd` Rust CLI through an Erlang Port.

  CLI contract (from `/workspace/sovereign-cmd/src/main.rs`): the program comes in on
  stdin; exit 0 = parsed and passed the contract (`--canonical` prints canonical
  lines), 1 = parse error, 2 = contract violation, 3 = denied lint, 64 = usage,
  74 = I/O.

  An Erlang Port cannot half-close the child's stdin, so the input is written to a
  private temp file and fed with `/bin/sh -c 'exec "$0" "$@" < in 2> err'`. That's one
  OS process per call (milliseconds), which is fine for cross-checking but is
  why the native parser is the per-worker default.
  """

  @doc "Path to the CLI binary (env `SOVEREIGN_CMD_BIN` wins over config)."
  def bin do
    System.get_env("SOVEREIGN_CMD_BIN") ||
      Application.get_env(:micro_agents, :rust_cli, "/workspace/sovereign-cmd/target/release/sovereign-cmd")
  end

  def available?, do: File.regular?(bin())

  @doc """
  Validate `input`. Options: `strict: true` adds `--strict-contract`.
  Returns `{:ok, canonical_lines} | {:error, {:structural, msg}} | {:error, {:contract, msg}}`
  or `{:error, {:cli, status, msg}}` for anything else.
  """
  def validate(input, opts \\ []) do
    dir = System.tmp_dir!()
    tag = :erlang.unique_integer([:positive])
    inf = Path.join(dir, "ma-in-#{System.pid()}-#{tag}")
    errf = inf <> ".err"
    File.write!(inf, input)

    flags = ["--canonical"] ++ if(Keyword.get(opts, :strict, false), do: ["--strict-contract"], else: [])

    try do
      port =
        Port.open({:spawn_executable, "/bin/sh"}, [
          :binary,
          :exit_status,
          :use_stdio,
          args: ["-c", ~s(exec "$0" "$@" < "$MA_IN" 2> "$MA_ERR"), bin() | flags],
          env: [{~c"MA_IN", String.to_charlist(inf)}, {~c"MA_ERR", String.to_charlist(errf)}]
        ])

      {status, out} = collect(port, [])
      err = File.read!(errf) |> String.trim()

      case status do
        0 -> {:ok, String.split(out, "\n", trim: true)}
        1 -> {:error, {:structural, err}}
        2 -> {:error, {:contract, err}}
        s -> {:error, {:cli, s, err}}
      end
    after
      File.rm(inf)
      File.rm(errf)
    end
  end

  defp collect(port, acc) do
    receive do
      {^port, {:data, d}} -> collect(port, [acc | d])
      {^port, {:exit_status, s}} -> {s, IO.iodata_to_binary(acc)}
    after
      10_000 ->
        Port.close(port)
        {:timeout, IO.iodata_to_binary(acc)}
    end
  end
end
