defmodule MicroAgents.Grammar do
  @moduledoc """
  Pure-Elixir port of `/workspace/sovereign-cmd/grammar/commands.gbnf`.

      root ::= (instruction)+
      instruction ::= verb " " target ( " " argument_list )? "\\n"
      verb ::= "FIX" | "OPEN" | "SET" | "TEST" | "RUN" | "READ" | "WRITE" | "ALLOC" | "FREE"
      target ::= identifier | filepath | hex_address
      argument_list ::= argument ( " " argument )? ( " " argument )?
      argument ::= flag | identifier | integer | hex_address
      identifier  ::= [a-zA-Z_] [a-zA-Z0-9_-]*
      filepath    ::= "/" [a-zA-Z0-9_-]+ ( "/" [a-zA-Z0-9_-]+ )* ("." [a-zA-Z0-9]+)?
      flag        ::= "--" [a-zA-Z0-9_-]+
      integer     ::= "-"? [0-9]+
      hex_address ::= "0x" [0-9a-fA-F]+

  Strict mode (exactly what the grammar accepts) plus the two semantic checks the
  Rust crate also performs: integers must fit in i64 and hex addresses in u64.

  Parsing allocates only small terms (atoms, sub-binaries, small integers), so it is
  cheap enough to run inside a worker whose heap starts at a few hundred words.

  Result: `{:ok, [instruction]}` where an instruction is
  `{verb_atom, target, [argument]}` with
  `target   :: {:ident, binary} | {:path, binary} | {:addr, non_neg_integer}` and
  `argument :: {:flag, binary} | {:ident, binary} | {:int, integer} | {:addr, non_neg_integer}`,
  or `{:error, {kind_atom, byte_offset}}`. Kind atoms mirror `sovereign_cmd::ErrorKind`
  in snake_case.
  """

  @verbs %{
    "FIX" => :fix,
    "OPEN" => :open,
    "SET" => :set,
    "TEST" => :test,
    "RUN" => :run,
    "READ" => :read,
    "WRITE" => :write,
    "ALLOC" => :alloc,
    "FREE" => :free
  }
  @verb_names Map.new(@verbs, fn {k, v} -> {v, k} end)
  @i64_min -0x8000_0000_0000_0000
  @i64_max 0x7FFF_FFFF_FFFF_FFFF
  @u64_max 0xFFFF_FFFF_FFFF_FFFF

  @type target :: {:ident, binary} | {:path, binary} | {:addr, non_neg_integer}
  @type argument :: {:flag, binary} | {:ident, binary} | {:int, integer} | {:addr, non_neg_integer}
  @type instruction :: {atom, target, [argument]}

  defguardp is_word(c) when c in ?a..?z or c in ?A..?Z or c in ?0..?9 or c == ?_ or c == ?-
  defguardp is_alnum(c) when c in ?a..?z or c in ?A..?Z or c in ?0..?9
  defguardp is_hex(c) when c in ?0..?9 or c in ?a..?f or c in ?A..?F

  @spec parse(binary) :: {:ok, [instruction]} | {:error, {atom, non_neg_integer}}
  def parse(<<>>), do: {:error, {:empty_input, 0}}
  def parse(bin) when is_binary(bin), do: lines(bin, 0, [])
  def parse(_), do: {:error, {:not_a_binary, 0}}

  defp lines(<<>>, _off, acc), do: {:ok, :lists.reverse(acc)}

  defp lines(bin, off, acc) do
    case :binary.match(bin, "\n") do
      :nomatch ->
        case line(bin, off) do
          {:ok, _} -> {:error, {:missing_final_newline, off + byte_size(bin)}}
          err -> err
        end

      {pos, 1} ->
        <<l::binary-size(^pos), ?\n, rest::binary>> = bin

        case line(l, off) do
          {:ok, instr} -> lines(rest, off + pos + 1, [instr | acc])
          err -> err
        end
    end
  end

  # ---- one line (without its "\n") -----------------------------------------

  defp line(<<>>, off), do: {:error, {:empty_line, off}}
  defp line(<<?\s, _::binary>>, off), do: {:error, {:leading_space, off}}

  defp line(l, off) do
    with :ok <- scan_bytes(l, off),
         {:ok, words} <- split_words(l, off, 0, 0, []) do
      instruction(words)
    end
  end

  # Byte-level checks: non-ASCII, tab, CR (strict whitespace mode).
  defp scan_bytes(<<>>, _), do: :ok
  defp scan_bytes(<<c, _::binary>>, off) when c >= 0x80, do: {:error, {:non_ascii, off}}
  defp scan_bytes(<<?\t, _::binary>>, off), do: {:error, {:tab, off}}
  defp scan_bytes(<<?\r, _::binary>>, off), do: {:error, {:carriage_return, off}}
  defp scan_bytes(<<_, rest::binary>>, off), do: scan_bytes(rest, off + 1)

  # Split on single spaces, keeping byte offsets; reject "  " and trailing " ".
  defp split_words(l, base, start, i, acc) when i == byte_size(l) do
    {:ok, :lists.reverse([{binary_part(l, start, i - start), base + start} | acc])}
  end

  defp split_words(l, base, start, i, acc) do
    case :binary.at(l, i) do
      ?\s ->
        cond do
          i + 1 == byte_size(l) -> {:error, {:trailing_space, base + i}}
          :binary.at(l, i + 1) == ?\s -> {:error, {:double_space, base + i + 1}}
          true -> split_words(l, base, i + 1, i + 1, [{binary_part(l, start, i - start), base + start} | acc])
        end

      _ ->
        split_words(l, base, start, i + 1, acc)
    end
  end

  defp instruction([{vword, voff} | rest]) do
    case @verbs do
      %{^vword => verb} ->
        case rest do
          [] -> {:error, {:missing_target, voff + byte_size(vword)}}
          [_, _, _, _, {_, aoff} | _] -> {:error, {:too_many_arguments, aoff}}
          [{tw, toff} | args] ->
            with {:ok, target} <- target(tw, toff),
                 {:ok, args} <- arguments(args, []) do
              {:ok, {verb, target, args}}
            end
        end

      _ ->
        if Map.has_key?(@verbs, String.upcase(vword)),
          do: {:error, {:lowercase_verb, voff}},
          else: {:error, {:unknown_verb, voff}}
    end
  end

  defp target(w, off) do
    case classify(w) do
      {:flag, _} -> {:error, {:invalid_target, off}}
      {:int, _} -> {:error, {:invalid_target, off}}
      {:error, kind} -> {:error, {kind, off}}
      t -> {:ok, t}
    end
  end

  defp arguments([], acc), do: {:ok, :lists.reverse(acc)}

  defp arguments([{w, off} | rest], acc) do
    case classify(w) do
      {:path, _} -> {:error, {:invalid_token, off}}
      {:error, kind} -> {:error, {kind, off}}
      a -> arguments(rest, [a | acc])
    end
  end

  # ---- token classification --------------------------------------------------

  @doc false
  def classify(<<"--", name::binary>>) when name != <<>> do
    if all?(name, :word), do: {:flag, name}, else: {:error, :invalid_token}
  end

  def classify(<<"0x", hex::binary>>) when hex != <<>> do
    if all?(hex, :hex) do
      v = :erlang.binary_to_integer(hex, 16)
      if v <= @u64_max, do: {:addr, v}, else: {:error, :hex_overflow}
    else
      {:error, :invalid_token}
    end
  end

  def classify(<<?-, digits::binary>> = w) when digits != <<>> do
    if all?(digits, :digit), do: int(w), else: {:error, :invalid_token}
  end

  def classify(<<c, _::binary>> = w) when c in ?0..?9 do
    if all?(w, :digit), do: int(w), else: {:error, :invalid_token}
  end

  def classify(<<?/, _::binary>> = w) do
    if filepath?(w), do: {:path, w}, else: {:error, :invalid_token}
  end

  def classify(<<c, rest::binary>> = w) when c in ?a..?z or c in ?A..?Z or c == ?_ do
    if all?(rest, :word), do: {:ident, w}, else: {:error, :invalid_token}
  end

  def classify(_), do: {:error, :invalid_token}

  defp int(w) do
    v = :erlang.binary_to_integer(w)
    if v >= @i64_min and v <= @i64_max, do: {:int, v}, else: {:error, :integer_overflow}
  end

  defp all?(<<>>, _), do: true
  defp all?(<<c, r::binary>>, :word) when is_word(c), do: all?(r, :word)
  defp all?(<<c, r::binary>>, :hex) when is_hex(c), do: all?(r, :hex)
  defp all?(<<c, r::binary>>, :digit) when c in ?0..?9, do: all?(r, :digit)
  defp all?(<<c, r::binary>>, :alnum) when is_alnum(c), do: all?(r, :alnum)
  defp all?(_, _), do: false

  # "/" seg ("/" seg)* ("." ext)?   seg = [a-zA-Z0-9_-]+, ext = [a-zA-Z0-9]+
  defp filepath?(<<?/, rest::binary>>), do: seg(rest, 0)
  defp seg(<<c, r::binary>>, n) when is_word(c), do: seg(r, n + 1)
  defp seg(<<>>, n), do: n > 0
  defp seg(<<?/, r::binary>>, n) when n > 0, do: seg(r, 0)
  defp seg(<<?., ext::binary>>, n) when n > 0 and ext != <<>>, do: all?(ext, :alnum)
  defp seg(_, _), do: false

  # ---- canonical form ------------------------------------------------------------

  @doc """
  Canonical source form of one instruction, without the trailing newline, matching
  `impl Display for CommandAST` in the Rust crate (hex upper-case, no leading zeros).
  """
  @spec canonical(instruction) :: binary
  def canonical({verb, target, args}) do
    IO.iodata_to_binary([
      Map.fetch!(@verb_names, verb),
      ?\s,
      fmt(target) | Enum.map(args, &[?\s, fmt(&1)])
    ])
  end

  defp fmt({:ident, s}), do: s
  defp fmt({:path, s}), do: s
  defp fmt({:flag, s}), do: ["--", s]
  defp fmt({:int, i}), do: Integer.to_string(i)
  defp fmt({:addr, a}), do: ["0x", Integer.to_string(a, 16)]

  @doc "All verb atoms."
  def verbs, do: Map.values(@verbs)
end
