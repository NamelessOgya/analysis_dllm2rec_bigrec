from vllm import SamplingParams
import sys

# Redirect stdout to file to capture help output
with open("vllm_help.txt", "w") as f:
    sys.stdout = f
    help(SamplingParams)
