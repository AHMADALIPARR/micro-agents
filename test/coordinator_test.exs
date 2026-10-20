defmodule MicroAgents.CoordinatorTest do
  use ExUnit.Case, async: false
  alias MicroAgents.{Coordinator, ContextSlice, Machine}

  setup do
    :ok = Coordinator.reset()
    procs = length(Process.list())
    on_exit(fn -> System.delete_env("SOVEREIGN_CMD_BIN") end)
    {:ok, procs: procs}
  end

  defp slice(opts \\ []), do: ContextSlice.new(opts)

  defp assert_no_leaks(procs_before) do
    # allow the VM a moment to reap
    Process.sleep(20)
    assert Coordinator.pending_count() == 0
    assert Task.Supervisor.children(MicroAgents.WorkerSup) == []
    assert length(Process.list()) <= procs_before
  end

  for spawn <- [:raw, :supervised] do
    describe "spawn: #{spawn}" do
      @spawn spawn

      test "valid command is accepted and advances the machine", %{procs: procs} do
        m0 = Coordinator.machine()
        assert {:accepted, {:commands, ["WRITE 0x2000 --force 42"]}} =
                 Coordinator.run(slice(), "WRITE 0x2000 --force 42\n", spawn: @spawn)

        m1 = Coordinator.machine()
        assert m1.version == m0.version + 1
        assert m1.applied == 1
        assert m1.recent == ["WRITE 0x2000 --force 42"]
        assert m1.chain != m0.chain
        assert_no_leaks(procs)
      end

      test "WRITE below 0x1000 kills the worker; machine unchanged", %{procs: procs} do
        coord = Process.whereis(Coordinator)
        {:accepted, _} = Coordinator.run(slice(), "SET a 1\n", spawn: @spawn)
        m0 = Coordinator.machine()

        assert {:rejected, {:contract_violation, {0, {:write_below_floor, 0xFFF}}}} =
                 Coordinator.run(slice(), "WRITE 0xFFF 1\n", spawn: @spawn)

        assert Coordinator.machine() == m0
        assert Process.whereis(Coordinator) == coord
        assert Coordinator.stats().contract_violations == 1
        assert [{:contract_violation, _}] = Coordinator.violations()
        assert_no_leaks(procs)
      end

      test "RUN outside /opt/sovereign/bin/ kills the worker; machine unchanged", %{procs: procs} do
        coord = Process.whereis(Coordinator)
        {:accepted, _} = Coordinator.run(slice(), "SET a 1\n", spawn: @spawn)
        m0 = Coordinator.machine()

        assert {:rejected, {:contract_violation, {1, {:run_outside_allowlist, "/tmp/evil"}}}} =
                 Coordinator.run(slice(), "RUN /opt/sovereign/bin/ok\nRUN /tmp/evil\n", spawn: @spawn)

        # the valid first line of a violating program must not be applied either
        assert Coordinator.machine() == m0
        assert Process.whereis(Coordinator) == coord
        assert_no_leaks(procs)
      end

      test "structural failure self-terminates the worker", %{procs: procs} do
        m0 = Coordinator.machine()
        assert {:rejected, {:structural_violation, {:double_space, _}}} =
                 Coordinator.run(slice(), "READ x  y\n", spawn: @spawn)

        assert Coordinator.machine() == m0
        assert_no_leaks(procs)
      end
    end
  end

  test "workers really exit with {:contract_violation, reason} and are not restarted" do
    s = slice()
    {pid, mref} = MicroAgents.Worker.start_supervised(self(), s, "WRITE 0x10\n")
    assert_receive {:DOWN, ^mref, :process, ^pid, {:contract_violation, {0, {:write_below_floor, 16}}}}
    Process.sleep(10)
    assert Task.Supervisor.children(MicroAgents.WorkerSup) == []

    {pid, mref} = MicroAgents.Worker.spawn_raw(self(), s, "RUN /bin/sh\n")
    assert_receive {:DOWN, ^mref, :process, ^pid, {:contract_violation, {0, {:run_outside_allowlist, "/bin/sh"}}}}
    refute_received {:micro_result, ^pid, _}
  end

  test "crash isolation: a worker that crashes (broken backend) does not affect the Coordinator", %{procs: procs} do
    coord = Process.whereis(Coordinator)
    {:accepted, _} = Coordinator.run(slice(), "SET a 1\n")
    m0 = Coordinator.machine()

    # 1) genuine exception inside the worker (non-iodata input to the Rust backend)
    assert {:crashed, {%File.Error{reason: :badarg}, _stack}} = Coordinator.run(slice(), nil, backend: :rust)
    # 2) backend failure: CLI path is not executable -> explicit {:backend_error, _} exit
    System.put_env("SOVEREIGN_CMD_BIN", "/")
    assert {:crashed, {:backend_error, {126, _}}} = Coordinator.run(slice(), "SET b 2\n", backend: :rust)
    assert Coordinator.stats().crashes == 2

    assert Process.whereis(Coordinator) == coord
    assert Coordinator.machine() == m0
    # Coordinator still serves
    assert {:accepted, _} = Coordinator.run(slice(), "SET c 3\n")
    assert_no_leaks(procs)
  end

  test "crash isolation: a worker killed mid-flight (timeout) leaves state unchanged", %{procs: procs} do
    script = Path.join(System.tmp_dir!(), "ma-slow-#{System.unique_integer([:positive])}.sh")
    File.write!(script, "#!/bin/sh\nsleep 1\n")
    File.chmod!(script, 0o755)
    System.put_env("SOVEREIGN_CMD_BIN", script)

    m0 = Coordinator.machine()
    assert :timeout = Coordinator.run(slice(), "SET b 2\n", backend: :rust, timeout: 50)
    assert Coordinator.machine() == m0
    assert Coordinator.stats().timeouts == 1
    assert_no_leaks(procs)
    File.rm(script)
  end

  test "forged result messages from non-workers are ignored" do
    m0 = Coordinator.machine()
    fake = spawn(fn -> :ok end)
    send(Coordinator, {:micro_result, fake, {:commands, ["WRITE 0x0"]}})
    send(Coordinator, {:micro_result, self(), {:commands, ["WRITE 0x0"]}})
    assert Coordinator.machine() == m0
  end

  test "predicate mode returns booleans and never mutates the machine" do
    m0 = Coordinator.machine()
    assert {:predicate, true} = MicroAgents.check("READ x\n")
    assert {:predicate, false} = MicroAgents.check("WRITE 0x1\n")
    assert {:predicate, false} = MicroAgents.check("nope\n")
    assert Coordinator.machine() == m0
  end

  test "sealed machine discards late results" do
    :ok = Coordinator.seal()
    m0 = Coordinator.machine()
    assert {:discarded, :sealed} = Coordinator.run(slice(), "READ x\n")
    assert Coordinator.machine() == m0
  end

  test "batch of 10k mixed workers: counts add up, no leaked processes", %{procs: procs} do
    inputs = MicroAgents.Bench.inputs()
    items = for i <- 0..9_999, do: {slice(), Enum.at(inputs, rem(i, 4))}

    for spawn <- [:raw, :supervised] do
      :ok = Coordinator.reset()
      %{summary: sum} = Coordinator.run_batch(items, collect: :summary, spawn: spawn, max_concurrency: 512)
      assert sum == %{accepted: 5000, contract_violation: 2500, structural_violation: 2500}
      m = Coordinator.machine()
      assert m.version == 5000
      assert Coordinator.stats().spawned == 10_000
      assert_no_leaks(procs)
    end
  end

  test "payloads are deterministic and the chain is reproducible with ordered execution" do
    inputs = MicroAgents.Bench.inputs()
    items = for i <- 0..399, do: {slice(), Enum.at(inputs, rem(i, 4))}

    run = fn ->
      :ok = Coordinator.reset()
      %{results: r} = Coordinator.run_batch(items, max_concurrency: 1)
      {r, Coordinator.machine()}
    end

    {r1, m1} = run.()
    {r2, m2} = run.()
    assert r1 == r2
    assert m1 == m2

    # concurrent run: same per-item payloads (order of application may differ)
    :ok = Coordinator.reset()
    %{results: r3} = Coordinator.run_batch(items, max_concurrency: 256)
    assert r3 == r1
  end

  test "Machine.apply_commands is pure" do
    m = Machine.new()
    {:ok, a} = Machine.apply_commands(m, ["READ x"])
    {:ok, b} = Machine.apply_commands(m, ["READ x"])
    assert a == b
    assert m == Machine.new()
  end
end
