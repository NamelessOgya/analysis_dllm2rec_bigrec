from vllm import SamplingParams

try:
    sp = SamplingParams(
        temperature=0,
        top_p=0.9,
        top_k=40,
        max_tokens=128,
        n=4,
    )
    print("SamplingParams instantiation SUCCESS")
except Exception as e:
    print(f"SamplingParams instantiation FAILED: {e}")
