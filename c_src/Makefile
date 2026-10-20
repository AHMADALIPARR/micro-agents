# Builds the helper against an existing llama.cpp build (does NOT rebuild llama.cpp).
LLAMA_DIR ?= /workspace/llama.cpp
LLAMA_LIB ?= $(LLAMA_DIR)/build/bin
OUT ?= ../priv/seqcp_bench
CC ?= cc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Wno-unused-parameter

$(OUT): seqcp_bench.c
	mkdir -p $(dir $(OUT))
	$(CC) $(CFLAGS) -I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include $< \
	  -L$(LLAMA_LIB) -lllama -lggml -lggml-base -lm -Wl,-rpath,$(LLAMA_LIB) -o $@

clean:
	rm -f $(OUT)
.PHONY: clean
