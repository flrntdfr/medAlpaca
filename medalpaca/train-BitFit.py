import os
import sys
from typing import Tuple, Union, List
import fire
import random
import numpy as np

import torch
import wandb

from datasets import load_dataset
from handler import DataHandler
from fastDP import PrivacyEngine
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    LlamaForCausalLM,
    LlamaTokenizer,
    Trainer,
    TrainingArguments,
)

# Seed for reproducibility
SEED = 54940


def train_with_fastdp(
    model,
    train_dataset,
    eval_dataset,
    tokenizer,
    per_device_batch_size,
    global_batch_size,
    num_epochs,
    learning_rate,
    output_dir,
    privacy_engine,
    use_wandb=False,
    eval_steps=200,
):
    """
    Manual training loop compatible with fastDP.
    
    This function implements a custom training loop that avoids the HuggingFace Trainer
    and Accelerate wrappers which are incompatible with fastDP's optimizer wrapping.
    """
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    
    # Create optimizer BEFORE attaching privacy engine
    # Using SGD instead of AdamW to reduce memory (AdamW needs 2x model size for momentum)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    # Attach privacy engine
    print("Attaching PrivacyEngine to optimizer...")
    privacy_engine.attach(optimizer)
    print("PrivacyEngine attached successfully!")
    
    # Remove non-tensor columns from datasets (only keep model inputs)
    # The Trainer does this automatically, but we need to do it manually
    model_input_columns = ['input_ids', 'attention_mask', 'labels']
    train_dataset = train_dataset.remove_columns(
        [col for col in train_dataset.column_names if col not in model_input_columns]
    )
    
    if eval_dataset:
        eval_dataset = eval_dataset.remove_columns(
            [col for col in eval_dataset.column_names if col not in model_input_columns]
        )
    
    # Create data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, 
        pad_to_multiple_of=8, 
        return_tensors="pt", 
        padding=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=per_device_batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    
    if eval_dataset:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=per_device_batch_size,
            collate_fn=data_collator,
        )
    
    # Calculate gradient accumulation steps
    gradient_accumulation_steps = global_batch_size // per_device_batch_size
    
    # Training loop
    model.train()
    global_step = 0
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")
        
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights after accumulation
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({
                "loss": f"{loss.item() * gradient_accumulation_steps:.4f}",
                "avg_loss": f"{epoch_loss / (step + 1) * gradient_accumulation_steps:.4f}"
            })
            
            # Log to wandb
            if use_wandb and (step + 1) % 10 == 0:
                wandb.log({
                    "train/loss": loss.item() * gradient_accumulation_steps,
                    "train/epoch": epoch,
                    "train/global_step": global_step,
                })
            
            # Evaluation
            if eval_dataset and global_step % eval_steps == 0 and global_step > 0:
                model.eval()
                eval_loss = 0
                eval_steps_count = 0
                
                with torch.no_grad():
                    for eval_batch in tqdm(eval_loader, desc="Evaluating", leave=False):
                        eval_batch = {k: v.to(model.device) for k, v in eval_batch.items()}
                        eval_outputs = model(**eval_batch)
                        eval_loss += eval_outputs.loss.item()
                        eval_steps_count += 1
                
                avg_eval_loss = eval_loss / eval_steps_count
                print(f"\nEvaluation at step {global_step}: loss = {avg_eval_loss:.4f}")
                
                if use_wandb:
                    wandb.log({
                        "eval/loss": avg_eval_loss,
                        "eval/global_step": global_step,
                    })
                
                model.train()
        
        print(f"Epoch {epoch + 1} completed. Average loss: {epoch_loss / len(train_loader) * gradient_accumulation_steps:.4f}")
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)


def main(
    model: str, # e.g. "decapoda-research/llama-7b-hf"
    val_set_size: Union[int, float] = 0.1,
    prompt_template: str = "prompts/medalpaca.json",
    model_max_length: int = 256,   # should not exceed 2048, as LLaMA is trained with this
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    data_path: str = "medical_meadow_small.json",
    train_in_8bit: bool = True,
    use_lora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    lora_target_modules: Tuple[str] = ("q_proj", "v_proj"),
    per_device_batch_size: int = 2,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    global_batch_size: int = 128,
    output_dir: str = "./output",
    save_total_limit: int = 3,
    eval_steps: int = 200,
    device_map: str = "auto",
    group_by_length: bool = False,
    wandb_run_name: str = "test",
    use_wandb: bool = False,
    wandb_project: str = "medalpaca",
    wandb_tags: Union[str, List[str]] = None,
    wandb_notes: str = None,
    optim: str = "adamw_torch",
    lr_scheduler_type: str = "cosine",
    fp16: bool = True,
    bf16: bool = False,
    gradient_checkpointing: bool = False,
    warmup_steps: int = 100,
    fsdp: str = "full_shard auto_wrap",
    fsdp_transformer_layer_cls_to_wrap: str = "LlamaDecoderLayer",
    # fastDP parameters
    target_epsilon: float = 2.0,
    clipping_fn: str = 'automatic',
    clipping_mode: str = 'MixOpt',
    clipping_style: str = 'all-layer',
    **kwargs
):
    """
    Trains a large language model using HuggingFace Transformers with custom configuration options.

    Args:
    model (str, optional):
        The model identifier on HuggingFace Model Hub.
    val_set_size (Union[int, float], optional):
        The proportion or number of samples to use for validation. Default is 0.1.
    prompt_template (str, optional):
        The path to the JSON file containing prompt templates. Default is "prompts/medalpaca.json".
    model_max_length (int, optional):
        The maximum length for model inputs. Default is 256.
    train_on_inputs (bool, optional):
        Whether to train on input tokens. Default is True.
    data_path (str, optional):
        The path to the dataset file. Default is "medical_meadow_small.json".
    train_in_8bit (bool, optional):
        Whether to use 8-bit training. Default is True.
    use_lora (bool, optional):
        Whether to use the Lora method. Default is True.
    lora_r (int, optional):
        The Lora method's reduction factor. Default is 8.
    lora_alpha (int, optional):
        The Lora method's alpha parameter. Default is 16.
    lora_dropout (float, optional):
        The dropout rate for Lora. Default is 0.1.
    lora_target_modules (List[str], optional):
        The target modules for Lora. Default is ["q_proj","v_proj"].
    per_device_batch_size (int, optional):
        The batch size per device. Default is 2.
    num_epochs (int, optional):
        The number of epochs for training. Default is 3.
    learning_rate (float, optional):
        The learning rate for the optimizer. Default is 2e-5.
    global_batch_size (int, optional):
        The number of samples the model needs to see until the weights get updated.
        Default is 128.
    output_dir (str, optional):
        The directory to save the model and outputs. Default is "./output".
    save_total_limit (int, optional):
        The maximum number of saved checkpoints. Default is 3.
    eval_steps (int, optional):
        The number of steps between evaluations. Default is 200.
    device_map (str, optional):
        The device placement strategy. Default is "auto".
    group_by_length (bool, optional):
        Whether to group samples by length for batch construction. Default is False.
    wandb_run_name (str, optional):
        The run name for Weights & Biases logging. Default is "test".
    use_wandb (bool, optional):
        Whether to use Weights & Biases for logging. Default is False.
    wandb_project (str, optional):
        The Weights & Biases project name. Default is "medalpaca".
    wandb_tags (Union[str, List[str]], optional):
        Tags to be added to the Weights & Biases run. Can be a comma-separated string or list of strings. Default is None.
    wandb_notes (str, optional):
        Notes to be added to the Weights & Biases run. Default is None.
    optim (str, optional):
        The optimizer to use. Default is "adamw_torch".
    lr_scheduler_type (str, optional):
        The learning rate scheduler type. Default is "cosine".
    fp16 (bool, optional):
        Whether to use mixed precision training (FP16). Default is True.
    bf16 (bool, optional):
        Whether to use mixed precision training (BF16). Default is False.
    gradient_checkpointing (bool, optional):
        Whether to use gradient checkpointing during training to reduce memory footprint
    warmup_steps (int, optional):
        The number of steps for warmup. Default is 200.
    fsdp (str, optional):
        Fully Sharded Data Parallel strategy. Only active with distributed training.
        Default is "full_shard auto_wrap"
    fsdp_transformer_layer_cls_to_wrap (optiona, str):
        The model layer to wrap for fsdp. Default is "LlamaDecoderLayer".
    target_epsilon (float, optional):
        Target epsilon for differential privacy. Default is 2.0.
    clipping_fn (str, optional):
        Clipping function for differential privacy. Default is 'automatic'.
    clipping_mode (str, optional):
        Clipping mode for differential privacy. Default is 'MixOpt'.
    clipping_style (str, optional):
        Clipping style for differential privacy. Default is 'all-layer'.
    **kwargs:
        additional arguments passed to the transformers.TrainingArguments"""

    # Set seeds for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    
    # Ensure deterministic behavior for fair GPU comparison
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Enable TensorFloat32 for better performance on Ampere+ GPUs (H100)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("TensorFloat32 enabled for accelerated FP32 operations")

    # Adapt arguments
    model_name = model
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    gradient_accumulation_steps = global_batch_size // per_device_batch_size
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
        if use_lora:
            # integer and mixed dtypes are not supported with fsdp
            fsdp, fsdp_transformer_layer_cls_to_wrap = "", None
    else:
        fsdp, fsdp_transformer_layer_cls_to_wrap = "", None

    # Initialize wandb with tags if use_wandb is True
    if use_wandb:
        # Parse wandb_tags - handle both string and list formats
        tags_list = []
        if wandb_tags:
            if isinstance(wandb_tags, str):
                # Split comma-separated string into list
                tags_list = [tag.strip() for tag in wandb_tags.split(',') if tag.strip()]
            elif isinstance(wandb_tags, list):
                tags_list = [str(tag).strip() for tag in wandb_tags if str(tag).strip()]
        
        # Initialize wandb run with tags and notes
        # The TrainingArguments with report_to="wandb" will handle the actual logging
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            tags=tags_list,
            notes=wandb_notes,
        )

    # perform some checks, to raise errors early
    if fp16 and bf16:
        raise ValueError("At most one of fp16 and bf16 can be True, but not both.")

    if train_in_8bit and not use_lora:
        raise ValueError("8bit training without LoRA is not supported")

    if use_lora and gradient_checkpointing:
        raise ValueError("gradient_checkpointing with LoRA training is not implemented")

    # init model
    if "llama" in model_name:
        # The LLaMA config on HF is not up to date with the library,
        # leading to errors when using AutoModelForCausalLM
        load_model = LlamaForCausalLM
    else:
        load_model = AutoModelForCausalLM

    # loading the model with torch_dtype=torch.float16 with only fp16 and no LoRA leads
    # to `ValueError: Attempting to unscale FP16 gradients.`

    model = load_model.from_pretrained(
        model_name,
        load_in_8bit=train_in_8bit,
        torch_dtype=torch.float16 if any([use_lora, bf16]) else torch.float32,
        device_map=device_map,
    )
    
    if train_in_8bit:
        model = prepare_model_for_kbit_training(model)

    if use_lora:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            inference_mode=False,
            # Add these new parameters
            #init_lora_weights=True,
            #use_rslora=True,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Explicitly disable gradient checkpointing for LoRA to prevent DDP conflicts
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_disable()
        if hasattr(model.config, 'use_gradient_checkpointing'):
            model.config.use_gradient_checkpointing = False
    
    # init tokenizer and tokenize function
    if "llama" in model_name.lower():
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    # load and tokenize data
    data_handler = DataHandler(
        tokenizer=tokenizer,
        prompt_template=prompt_template,
        model_max_length=model_max_length,
        train_on_inputs=train_on_inputs,
    )
    data = load_dataset("json", data_files=data_path)

    if val_set_size > 0:
        data = (
            data["train"]
            .train_test_split(test_size=val_set_size, shuffle=True, seed=SEED)
            .map(data_handler.generate_and_tokenize_prompt)
        )
    else:
        data = data.shuffle(seed=SEED).map(data_handler.generate_and_tokenize_prompt)

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    # Initialize PrivacyEngine for differential privacy
    print("=" * 60)
    print("Initializing fastDP PrivacyEngine...")
    print("=" * 60)
    
    sample_size = len(data["train"])
    
    privacy_engine = PrivacyEngine(
        model,
        batch_size=global_batch_size,
        sample_size=sample_size,
        epochs=num_epochs,
        target_epsilon=target_epsilon,
        clipping_fn=clipping_fn,
        clipping_mode=clipping_mode,
        origin_params=None,
        clipping_style=clipping_style,
    )
    
    print(f"PrivacyEngine configured:")
    print(f"  - Batch size: {global_batch_size}")
    print(f"  - Sample size: {sample_size}")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Target epsilon: {target_epsilon}")
    print(f"  - Clipping function: {clipping_fn}")
    print(f"  - Clipping mode: {clipping_mode}")
    print(f"  - Clipping style: {clipping_style}")
    print("=" * 60)

    # Configure model for training
    model.config.use_cache = False

    if use_lora:
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
        ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # Train with manual loop (fastDP is incompatible with HuggingFace Trainer/Accelerate)
    train_with_fastdp(
        model=model,
        train_dataset=data["train"],
        eval_dataset=data["test"] if val_set_size > 0 else None,
        tokenizer=tokenizer,
        per_device_batch_size=per_device_batch_size,
        global_batch_size=global_batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        output_dir=output_dir,
        privacy_engine=privacy_engine,
        use_wandb=use_wandb,
        eval_steps=eval_steps,
    )

    #model.save_pretrained(output_dir) # Commented for benchmarks

if __name__ == "__main__":
    fire.Fire(main)
