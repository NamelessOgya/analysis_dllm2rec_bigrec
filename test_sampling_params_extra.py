from vllm import SamplingParams

try:
    sp = SamplingParams(
        n=4,
        temperature=0,
        extra_args={"use_beam_search": True}
    )
    print("SamplingParams instantiation with extra_args SUCCESS")
    print(sp)
except Exception as e:
    print(f"SamplingParams instantiation FAILED: {e}")
