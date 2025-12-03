import fire
from typing import List

def train(
    train_data_path: List[str] = [""],
    val_data_path: List[str] = [""],
    lora_target_modules: List[str] = ["q_proj", "v_proj"],
    resume_from_checkpoint: str = None,
):
    print(f"train_data_path: {train_data_path} (type: {type(train_data_path)})")
    print(f"val_data_path: {val_data_path} (type: {type(val_data_path)})")
    print(f"lora_target_modules: {lora_target_modules} (type: {type(lora_target_modules)})")
    print(f"resume_from_checkpoint: {resume_from_checkpoint} (type: {type(resume_from_checkpoint)})")

if __name__ == "__main__":
    fire.Fire(train)
