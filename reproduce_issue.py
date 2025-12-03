
import torch
print(f"Torch version: {torch.__version__}")
try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
except AttributeError as e:
    print(f"Error importing transformers: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
