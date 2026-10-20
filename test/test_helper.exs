rust? = MicroAgents.Validator.RustPort.available?()
exclude = [:llm] ++ if(rust?, do: [], else: [:rust])
ExUnit.start(exclude: exclude, capture_log: true)
