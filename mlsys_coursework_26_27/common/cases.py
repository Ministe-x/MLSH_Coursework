"""Exactly 12 shared numerical-correctness/performance cases per part.

GEMM tuple order: M,N,K. Attention tuple order: B,H,S,D.
Each case is checked numerically once, then timed if it passes.
--suite correctness skips timing on these same 12 cases; it adds no shapes.
"""
GEMM_PERFORMANCE = [
    ("square_256", (256, 256, 256), "random"),
    ("square_512", (512, 512, 512), "random"),
    ("square_1024", (1024, 1024, 1024), "random"),
    ("wide", (256, 1024, 512), "random"),
    ("tall", (1024, 256, 512), "random"),
    ("odd", (257, 513, 129), "random"),
    # Added LLM cases; the six original cases above are unchanged.
    # Llama-3-8B-style widths: hidden=4096, FFN=14336, KV projection=1024.
    # M is flattened token count; N is output width; K is input width.
    ("llm_decode_q_proj", (1, 4096, 4096), "random"),
    ("llm_decode_ffn_up", (1, 14336, 4096), "random"),
    ("llm_prefill_q_proj", (128, 4096, 4096), "random"),
    ("llm_prefill_kv_proj", (512, 1024, 4096), "random"),
    ("llm_prefill_ffn_up", (256, 14336, 4096), "random"),
    ("llm_prefill_ffn_down", (256, 4096, 14336), "random"),
]
ATTENTION_PERFORMANCE = [
    ("s128_d64", (1, 4, 128, 64), "random"),
    ("s256_d64", (2, 4, 256, 64), "random"),
    ("s512_d64", (1, 4, 512, 64), "random"),
    ("s1024_d64", (1, 4, 1024, 64), "random"),
    ("s512_d128", (1, 2, 512, 128), "random"),
    ("odd", (1, 3, 257, 80), "random"),
    # Added full causal self-attention / prefill cases, not cached decode.
    # GPT-2-style MHA: 12 heads, D=64. Llama-2-7B-style MHA: 32 heads, D=128.
    ("llm_h12_s128_d64", (1, 12, 128, 64), "random"),
    ("llm_h12_s512_d64", (1, 12, 512, 64), "random"),
    ("llm_b2_h12_s512_d64", (2, 12, 512, 64), "random"),
    ("llm_h32_s512_d128", (1, 32, 512, 128), "random"),
    ("llm_h32_s1024_d128", (1, 32, 1024, 128), "random"),
    ("llm_h32_s2048_d128", (1, 32, 2048, 128), "random"),
]
