defmodule MicroAgents.MixProject do
  use Mix.Project

  def project do
    [
      app: :micro_agents,
      version: "0.1.0",
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      # No Hex dependencies: the project builds fully offline.
      deps: []
    ]
  end

  def application do
    [
      # :inets (httpc) is only used by the optional llama.cpp HTTP client.
      extra_applications: [:logger, :inets, :crypto],
      mod: {MicroAgents.Application, []}
    ]
  end
end
