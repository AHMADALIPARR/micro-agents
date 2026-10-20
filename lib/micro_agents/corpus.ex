defmodule MicroAgents.Corpus do
  @moduledoc """
  Deterministic test corpus for cross-checking the native validator against the
  Rust CLI: hand-picked edge cases plus seeded random programs that mix valid and
  invalid tokens.
  """

  @handpicked [
    "WRITE 0x1000 --force 42\n",
    "WRITE 0xFFF\n",
    "WRITE 0x0\n",
    "WRITE 0x0fff\n",
    "WRITE 0x00001000\n",
    "WRITE 0x2000 0x10\n",
    "WRITE buf 0x10\n",
    "WRITE /dev/mem 0x0\n",
    "RUN /opt/sovereign/bin/agent\n",
    "RUN /opt/sovereign/bin/x.sh --dry-run 3\n",
    "RUN /opt/sovereign/bin\n",
    "RUN /opt/sovereign/binx/a\n",
    "RUN /opt/sovereign/bin.sh\n",
    "RUN /usr/bin/rm --rf\n",
    "RUN /opt/sovereign/bin/a/b/c.py\n",
    "RUN foo\n",
    "RUN 0x2000\n",
    "READ /a/b.txt 0x10 -5\n",
    "READ 0x00ff 007 -0\n",
    "FIX a\nOPEN b\nSET c 1 2 3\n",
    "SET c 1 2 3 4\n",
    "TEST x\n\nTEST y\n",
    "\n",
    "",
    "READ x",
    "READ\n",
    "READ \n",
    "READ  x\n",
    " READ x\n",
    "READ x \n",
    "READ\tx\n",
    "READ x\r\n",
    "read x\n",
    "Read x\n",
    "JUMP x\n",
    "READ --flag\n",
    "READ -5\n",
    "READ 5\n",
    "READ x /a/b\n",
    "READ x --\n",
    "READ x ---y\n",
    "READ x -\n",
    "READ x 0x\n",
    "READ x 0xG\n",
    "READ x 0x1FFFFFFFFFFFFFFFF\n",
    "READ x 0xFFFFFFFFFFFFFFFF\n",
    "READ x 9223372036854775807\n",
    "READ x 9223372036854775808\n",
    "READ x -9223372036854775808\n",
    "READ x -9223372036854775809\n",
    "OPEN /a.b/c\n",
    "OPEN /a/b.c.d\n",
    "OPEN /a/b.\n",
    "OPEN /a//b\n",
    "OPEN /\n",
    "OPEN /a/\n",
    "OPEN /a/b_c-d.tar\n",
    "OPEN /a/b.t_x\n",
    "ALLOC _x-1 --size 4096\n",
    "FREE 0xDEADBEEF\n",
    "FREE é\n",
    "FREE x\nFREE y",
    "WRITE 0x1000\nWRITE 0x10\n",
    "RUN /opt/sovereign/bin/ok\nRUN /tmp/evil\n"
  ]

  def handpicked, do: @handpicked

  @verbs ~w(FIX OPEN SET TEST RUN READ WRITE ALLOC FREE)
  @bad_verbs ~w(run Write JUMP)
  @good [
    "x", "_a", "Ab-9", "buf_1", "/opt/sovereign/bin/a", "/opt/sovereign/bin/t.sh",
    "/opt/sovereign/bin", "/usr/bin/x", "/a/b.txt", "0x0", "0xFFF", "0x1000", "0xdeadBEEF",
    "0x00ff", "--f", "--force-x", "-1", "42", "007", "-0", "9223372036854775807"
  ]
  @targets ["x", "_a", "Ab-9", "/opt/sovereign/bin/a", "/opt/sovereign/bin", "/usr/bin/x",
             "/a/b.txt", "0x0", "0xFFF", "0x1000", "0xdeadBEEF", "0x00ff", "/opt/sovereign/bin/t.sh"]
  @bad [
    "/a.b", "0x", "0xZZ", "--", "-", "1a", "a.b", "é", "x\t", "", "99999999999999999999",
    "0x10000000000000000", "/a//b", "/a/b.t_x"
  ]

  defp verb, do: if(:rand.uniform(12) == 1, do: Enum.random(@bad_verbs), else: Enum.random(@verbs))
  defp tok, do: if(:rand.uniform(10) == 1, do: Enum.random(@bad), else: Enum.random(@good))

  @doc "`n` seeded random programs (1..3 lines each)."
  def random(n, seed \\ 1337) do
    :rand.seed(:exsss, {seed, seed * 7, seed * 13})
    for _ <- 1..n, do: program()
  end

  defp program do
    lines = for _ <- 1..:rand.uniform(3), do: line()
    nl = if :rand.uniform(20) == 1, do: "", else: "\n"
    Enum.join(lines, "\n") <> nl
  end

  defp line do
    ntok = if :rand.uniform(15) == 1, do: Enum.random([0, 5]), else: :rand.uniform(4)
    toks = for(_ <- 1..ntok//1, do: tok())
    # bias the first word after the verb toward valid targets (no flags/integers)
    toks =
      case toks do
        [_ | rest] when ntok > 0 -> [if(:rand.uniform(8) == 1, do: tok(), else: Enum.random(@targets)) | rest]
        t -> t
      end

    Enum.join([verb() | toks], sep())
  end

  defp sep, do: if(:rand.uniform(60) == 1, do: "  ", else: " ")

  def all(n_random \\ 2000), do: @handpicked ++ random(n_random)
end
