defmodule MicroAgents.CrossCheckTest do
  use ExUnit.Case, async: true
  @moduletag :rust
  @moduletag timeout: 300_000

  test "native validator and Rust CLI agree on the full corpus (default + strict contract)" do
    r = MicroAgents.CrossCheck.run(MicroAgents.Corpus.all(2000))
    assert r.disagreements == [], inspect(Enum.take(r.disagreements, 10))
    assert r.cases == 2 * (length(MicroAgents.Corpus.handpicked()) + 2000)
  end

  test "Rust backend inside a worker reports violations the same way" do
    s = MicroAgents.ContextSlice.new()
    {pid, mref} = MicroAgents.Worker.spawn_raw(self(), s, "WRITE 0xFFF\n", backend: :rust)
    assert_receive {:DOWN, ^mref, :process, ^pid, {:contract_violation, msg}}, 5_000
    assert msg =~ "protected memory"
  end
end
