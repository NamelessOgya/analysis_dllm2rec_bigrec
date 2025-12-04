from vllm.sampling_params import BeamSearchParams

try:
    bsp = BeamSearchParams(
        beam_width=4,
        max_tokens=128,
        temperature=0
    )
    # Try adding the missing attribute
    bsp.truncate_prompt_tokens = None
    print("Monkey-patching SUCCESS")
    print(f"truncate_prompt_tokens: {bsp.truncate_prompt_tokens}")
except Exception as e:
    print(f"Monkey-patching FAILED: {e}")

try:
    class FixedBeamSearchParams(BeamSearchParams):
        truncate_prompt_tokens: int | None = None
    
    fbsp = FixedBeamSearchParams(
        beam_width=4,
        max_tokens=128,
        temperature=0,
        truncate_prompt_tokens=None
    )
    print("Subclassing SUCCESS")
    print(f"truncate_prompt_tokens: {fbsp.truncate_prompt_tokens}")
except Exception as e:
    print(f"Subclassing FAILED: {e}")
