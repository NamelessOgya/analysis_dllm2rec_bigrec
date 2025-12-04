import vllm
from vllm import SamplingParams
import inspect

with open("vllm_version_info.txt", "w") as f:
    f.write(f"VERSION: {vllm.__version__}\n")
    f.write(f"SIGNATURE: {inspect.signature(SamplingParams)}\n")
    f.write(f"DOC: {SamplingParams.__doc__}\n")
