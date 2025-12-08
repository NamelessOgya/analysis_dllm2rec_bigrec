import os
os.environ['LD_LIBRARY_PATH'] = 'YOUR_CONDA_ENV/lib'
import sys
from typing import List

import numpy as np 
import fire
import torch
import transformers
from datasets import load_dataset, concatenate_datasets
from transformers import EarlyStoppingCallback
from transformers import AutoModelForCausalLM, AutoTokenizer
"""
Unused imports:`
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (  # noqa: E402
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
# from transformers import LlamaForCausalLM, LlamaTokenizer  # noqa: F402


def train(
    # model/data params
    base_model: str = "",  # the only required argument
    train_data_path: List[str] = [""],
    val_data_path: List[str] = [""],
    output_dir: str = "./lora-alpaca",
    sample: int = -1,
    seed: int = 0,
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_file: str = None,
):
    print(
        f"Training Alpaca-LoRA model with params:\n"
        f"base_model: {base_model}\n"
        f"train_data_path: {train_data_path}\n"
        f"val_data_path: {val_data_path}\n"
        f"sample: {sample}\n"
        f"seed: {seed}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
        f"prompt_file: {prompt_file}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size
    # print(f"gradient_accumulation_steps: {gradient_accumulation_steps}")

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model
    os.environ["WANDB_DISABLED"] = "true"
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    # model.set_tau(tau)
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    except:
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(base_model)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = (
            tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        )
    tokenizer.padding_side = "right"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point, prompt_file)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""}, prompt_file)
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)


    train_data_list = []
    val_data_list = []

    for path in train_data_path:
        if path.endswith(".json"):
            train_data_list.append(load_dataset("json", data_files=path))
        else:
            train_data_list.append(load_dataset(path))

    for path in val_data_path:
        if path.endswith(".json"):
            val_data_list.append(load_dataset("json", data_files=path))
        else:
            val_data_list.append(load_dataset(path))

    for i in range(len(train_data_list)):
        train_data_list[i]["train"] = train_data_list[i]["train"].shuffle(seed=seed).select(range(sample)) if sample > -1 else train_data_list[i]["train"].shuffle(seed=seed)
        train_data_list[i]["train"] = train_data_list[i]["train"].shuffle(seed=seed)
        train_data_list[i] = train_data_list[i].map(lambda x: generate_and_tokenize_prompt(x))
    for i in range(len(val_data_list)):
        val_data_list[i] = val_data_list[i].map(lambda x: generate_and_tokenize_prompt(x))
    train_data = concatenate_datasets([_["train"] for _ in train_data_list])
    val_data = concatenate_datasets([_["train"] for _ in val_data_list])

    
    # train_data = train_data.shuffle(seed=42)[:sample] if sample > -1 else train_data
    # print(len(train_data))
    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    
 

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=8,
            optim="adamw_torch",
            eval_strategy="epoch",
            save_strategy="epoch",
            output_dir=output_dir,
            save_total_limit=None,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to=None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save best model with epoch info
    if trainer.state.best_model_checkpoint:
        best_checkpoint_path = trainer.state.best_model_checkpoint
        best_metric = trainer.state.best_metric 
        
        try:
            # Try to find epoch from log history
            # Checkpoint directory is usually "checkpoint-X"
            best_step = int(best_checkpoint_path.split('-')[-1])
            best_epoch = None
            
            # log_history is a list of dicts. We search for the entry corresponding to the best step.
            for entry in trainer.state.log_history:
                if entry.get("step") == best_step and "epoch" in entry:
                    best_epoch = entry["epoch"]
                    break
            
            if best_epoch is None:
                # If not found in log history, try to estimate or fallback
                best_epoch = "unknown"
            
            # Format epoch for directory name
            if isinstance(best_epoch, (int, float)):
                best_epoch_str = f"{int(best_epoch)}"
            else:
                best_epoch_str = str(best_epoch)
                
            best_model_dir = os.path.join(output_dir, f"best_model_epoch_{best_epoch_str}")
            
            if not os.path.exists(best_model_dir):
                os.makedirs(best_model_dir)

            print(f"Saving best model (step {best_step}, epoch {best_epoch_str}) to {best_model_dir}")
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            
            # Also save a summary json
            import json
            with open(os.path.join(output_dir, "training_summary.json"), "w") as f:
                json.dump({
                    "best_model_checkpoint": best_checkpoint_path,
                    "best_metric": best_metric,
                    "best_step": best_step,
                    "best_epoch": best_epoch,
                }, f, indent=4)
        except Exception as e:
            print(f"Error saving best model with epoch info: {e}")

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


def generate_prompt(data_point, prompt_file=None):
    # Read templates from files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if prompt_file:
        template_path = prompt_file
    else:
        template_path = os.path.join(script_dir, "templates", "prompt_template.txt")

    template_no_input_path = os.path.join(script_dir, "templates", "prompt_template_no_input.txt")
    
    if data_point["input"]:
        with open(template_path, "r") as f:
            template = f.read()
        return template.format(
            instruction=data_point["instruction"],
            input=data_point["input"],
            output=data_point["output"]
        )
    else:
        with open(template_no_input_path, "r") as f:
            template = f.read()
        return template.format(
            instruction=data_point["instruction"],
            output=data_point["output"]
        )


if __name__ == "__main__":
    fire.Fire(train)
