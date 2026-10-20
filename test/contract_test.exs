defmodule MicroAgents.ContractTest do
  use ExUnit.Case, async: true
  alias MicroAgents.Validator

  defp v(src, mode \\ :default), do: Validator.validate(src, %{contract: mode})

  test "WRITE to hex address below 0x1000 is rejected" do
    assert {:error, {:contract, {0, {:write_below_floor, 0xFFF}}}} = v("WRITE 0xFFF\n")
    assert {:error, {:contract, {0, {:write_below_floor, 0}}}} = v("WRITE 0x0\n")
    assert {:ok, _} = v("WRITE 0x1000\n")
    assert {:ok, _} = v("WRITE 0x00001000\n")
  end

  test "RUN outside /opt/sovereign/bin/ is rejected" do
    assert {:error, {:contract, {0, {:run_outside_allowlist, "/usr/bin/rm"}}}} = v("RUN /usr/bin/rm --rf\n")
    assert {:error, {:contract, _}} = v("RUN /opt/sovereign/binx/a\n")
    assert {:error, {:contract, _}} = v("RUN /opt/sovereign/bin.sh\n")
    assert {:ok, _} = v("RUN /opt/sovereign/bin/agent\n")
    assert {:ok, _} = v("RUN /opt/sovereign/bin/a/b/c.py\n")
  end

  test "violation index points at the offending instruction" do
    assert {:error, {:contract, {1, {:write_below_floor, 0x10}}}} = v("WRITE 0x1000\nWRITE 0x10\n")
  end

  test "default mode mirrors Rust ExecutionContract::default() gaps" do
    assert {:ok, _} = v("RUN foo\n")
    assert {:ok, _} = v("RUN /opt/sovereign/bin\n")
    assert {:ok, _} = v("WRITE 0x2000 0x10\n")
  end

  test "strict mode mirrors --strict-contract" do
    assert {:error, {:contract, {0, {:run_target_not_path, {:ident, "foo"}}}}} = v("RUN foo\n", :strict)
    assert {:error, {:contract, {0, {:run_target_is_dir, _}}}} = v("RUN /opt/sovereign/bin\n", :strict)
    assert {:error, {:contract, {0, {:write_arg_below_floor, 0x10}}}} = v("WRITE 0x2000 0x10\n", :strict)
  end

  test "slice can only tighten: allowed_verbs and max_instructions" do
    assert {:error, {:contract, {0, {:verb_not_allowed, :free}}}} =
             Validator.validate("FREE x\n", %{allowed_verbs: [:read]})

    assert {:error, {:contract, {:too_many_instructions, 2, 1}}} =
             Validator.validate("READ a\nREAD b\n", %{max_instructions: 1})
  end
end
