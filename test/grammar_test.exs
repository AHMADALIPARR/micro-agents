defmodule MicroAgents.GrammarTest do
  use ExUnit.Case, async: true
  alias MicroAgents.Grammar

  @valid [
    {"WRITE 0x1000 --force 42\n", ["WRITE 0x1000 --force 42"]},
    {"READ /a/b.txt 0x10 -5\n", ["READ /a/b.txt 0x10 -5"]},
    {"READ 0x00ff 007 -0\n", ["READ 0xFF 7 0"]},
    {"FIX a\nOPEN b\nSET c 1 2 3\n", ["FIX a", "OPEN b", "SET c 1 2 3"]},
    {"ALLOC _x-1 --size 4096\n", ["ALLOC _x-1 --size 4096"]},
    {"RUN /opt/sovereign/bin/x.sh --dry-run\n", ["RUN /opt/sovereign/bin/x.sh --dry-run"]},
    {"OPEN /a/b_c-d.tar\n", ["OPEN /a/b_c-d.tar"]},
    {"READ x ---y\n", ["READ x ---y"]},
    {"READ x 9223372036854775807 -9223372036854775808 0xFFFFFFFFFFFFFFFF\n",
     ["READ x 9223372036854775807 -9223372036854775808 0xFFFFFFFFFFFFFFFF"]}
  ]

  test "valid programs parse and canonicalise" do
    for {src, canon} <- @valid do
      assert {:ok, instrs} = Grammar.parse(src), "expected accept: #{inspect(src)}"
      assert Enum.map(instrs, &Grammar.canonical/1) == canon
    end
  end

  test "AST shape" do
    assert {:ok, [{:write, {:addr, 0x1000}, [{:flag, "force"}, {:int, 42}]}]} =
             Grammar.parse("WRITE 0x1000 --force 42\n")
  end

  @invalid [
    {"", :empty_input},
    {"\n", :empty_line},
    {"READ x", :missing_final_newline},
    {"READ\n", :missing_target},
    {"READ \n", :trailing_space},
    {"READ  x\n", :double_space},
    {" READ x\n", :leading_space},
    {"READ\tx\n", :tab},
    {"READ x\r\n", :carriage_return},
    {"read x\n", :lowercase_verb},
    {"JUMP x\n", :unknown_verb},
    {"READ --flag\n", :invalid_target},
    {"READ -5\n", :invalid_target},
    {"SET c 1 2 3 4\n", :too_many_arguments},
    {"READ x 0x\n", :invalid_token},
    {"READ x 0xG\n", :invalid_token},
    {"READ x /a/b\n", :invalid_token},
    {"OPEN /a.b/c\n", :invalid_token},
    {"OPEN /a/b.c.d\n", :invalid_token},
    {"OPEN /a//b\n", :invalid_token},
    {"OPEN /\n", :invalid_token},
    {"FREE é\n", :non_ascii},
    {"READ x 9223372036854775808\n", :integer_overflow},
    {"READ x 0x1FFFFFFFFFFFFFFFF\n", :hex_overflow},
    {"TEST x\n\nTEST y\n", :empty_line}
  ]

  test "invalid programs are rejected with the expected kind" do
    for {src, kind} <- @invalid do
      assert {:error, {^kind, _off}} = Grammar.parse(src), "expected #{kind}: #{inspect(src)}"
    end
  end
end
