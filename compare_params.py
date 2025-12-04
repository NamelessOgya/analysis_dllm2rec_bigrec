from vllm.sampling_params import BeamSearchParams, SamplingParams

# Create a dummy SamplingParams to see what attributes it has
sp = SamplingParams()
sp_attrs = set(dir(sp))

# Create BeamSearchParams
bsp = BeamSearchParams(beam_width=4, max_tokens=10, temperature=0)
bsp_attrs = set(dir(bsp))

# Find attributes in SamplingParams but not in BeamSearchParams
# These are potential candidates for missing attributes if LLM engine expects them
missing = sp_attrs - bsp_attrs
print("Attributes in SamplingParams but NOT in BeamSearchParams:")
for attr in sorted(missing):
    if not attr.startswith('_'):
        print(attr)
