defmodule MicroAgents.MemoryTest do
  use ExUnit.Case, async: false

  test "idle actor (spawn_opt, min_heap_size small) is <= 4 KiB" do
    pid = :erlang.spawn_opt(fn -> receive do: (:stop -> :ok) end, [{:min_heap_size, MicroAgents.Worker.min_heap()}])
    {:memory, bytes} = :erlang.process_info(pid, :memory)
    send(pid, :stop)
    IO.puts("\n  idle actor memory: #{bytes} bytes")
    assert bytes <= 4096
  end

  test "idle worker holding a ContextSlice with a 4 KiB shared prompt is <= 4 KiB" do
    s = MicroAgents.ContextSlice.new(prompt: :binary.copy("p", 4096), allowed_verbs: [:read])
    parent = self()

    pid =
      :erlang.spawn_opt(
        fn -> receive do: (:go -> MicroAgents.Worker.run(parent, s, "READ x\n", :native)) end,
        [{:min_heap_size, 1}]
      )

    {:memory, bytes} = :erlang.process_info(pid, :memory)
    send(pid, :go)
    assert_receive {:micro_result, ^pid, {:commands, ["READ x"]}}
    assert bytes <= 4096
  end
end
