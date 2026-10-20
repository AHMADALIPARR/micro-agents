defmodule MicroAgents.LLM.Client do
  @moduledoc """
  Minimal llama-server HTTP client on OTP's `:httpc` (no Hex deps) and the
  built-in `JSON` module (Elixir 1.18).

  `completion/3` is non-streaming; `stream_completion/3` streams SSE and measures
  client-side time-to-first-token (request sent -> first non-empty `content`
  chunk).
  """

  def get(url) do
    case :httpc.request(:get, {String.to_charlist(url), []}, [timeout: 2_000, ssl: []], body_format: :binary) do
      {:ok, {{_, status, _}, _h, body}} -> {:ok, status, body}
      e -> e
    end
  end

  def completion(url, body, timeout \\ 120_000) do
    req = {String.to_charlist(url <> "/completion"), [], ~c"application/json", JSON.encode!(body)}

    case :httpc.request(:post, req, [timeout: timeout, ssl: []], body_format: :binary) do
      {:ok, {{_, 200, _}, _h, resp}} -> {:ok, JSON.decode!(resp)}
      {:ok, {{_, s, _}, _h, resp}} -> {:error, {s, resp}}
      e -> e
    end
  end

  @doc """
  Streaming completion. Returns `{:ok, %{content, ttft_us, total_us, timings, chunks}}`.
  """
  def stream_completion(url, body, timeout \\ 120_000) do
    body = Map.put(body, :stream, true)
    req = {String.to_charlist(url <> "/completion"), [], ~c"application/json", JSON.encode!(body)}
    t0 = System.monotonic_time(:microsecond)

    {:ok, ref} =
      :httpc.request(:post, req, [timeout: timeout, ssl: []], sync: false, stream: :self, body_format: :binary)

    loop(ref, t0, %{content: [], ttft_us: nil, timings: nil, chunks: 0, buf: "", last: nil}, timeout)
  end

  defp loop(ref, t0, acc, timeout) do
    receive do
      {:http, {^ref, :stream_start, _headers}} ->
        loop(ref, t0, acc, timeout)

      {:http, {^ref, :stream, data}} ->
        loop(ref, t0, parse_sse(acc.buf <> data, %{acc | buf: ""}, t0), timeout)

      {:http, {^ref, :stream_end, _headers}} ->
        acc = parse_sse(acc.buf <> "\n\n", %{acc | buf: ""}, t0)

        {:ok,
         %{
           content: IO.iodata_to_binary(acc.content),
           ttft_us: acc.ttft_us,
           total_us: System.monotonic_time(:microsecond) - t0,
           timings: acc.timings,
           chunks: acc.chunks,
           last: acc.last
         }}

      {:http, {^ref, {{_, status, _}, _h, body}}} ->
        {:error, {status, body}}

      {:http, {^ref, {:error, reason}}} ->
        {:error, reason}
    after
      timeout -> {:error, :timeout}
    end
  end

  defp parse_sse(data, acc, t0) do
    {events, rest} = split_events(data)

    acc =
      Enum.reduce(events, acc, fn ev, a ->
        case String.trim(ev) do
          "data: " <> json ->
            case JSON.decode(json) do
              {:ok, m} ->
                c = Map.get(m, "content", "")

                a =
                  if c != "" and is_nil(a.ttft_us),
                    do: %{a | ttft_us: System.monotonic_time(:microsecond) - t0},
                    else: a

                %{a | content: [a.content | c], chunks: a.chunks + 1, timings: Map.get(m, "timings", a.timings), last: m}

              _ ->
                a
            end

          _ ->
            a
        end
      end)

    %{acc | buf: rest}
  end

  defp split_events(data) do
    parts = String.split(data, "\n\n")
    {Enum.drop(parts, -1), List.last(parts)}
  end
end
