# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftConfig, prepare_model_for_kbit_training, get_peft_model

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer


class LlamaTrainer:
    def __init__(
        self,
        local_rank: int = -1,
        per_device_train_batch_size: int = 4,
        per_device_eval_batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        max_grad_norm: float = 0.3,
        weight_decay: float = 0.001,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_r: int = 64,
        max_seq_length: int = 1024,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        dataset: Optional[Dataset] = None,
        formatting_func: Optional[Callable] = None,
        dataset_text_field: Optional[str] = None,
        use_4bit: bool = True,
        use_nested_quant: bool = True,
        bnb_4bit_compute_dtype: str = "bfloat16",
        bnb_4bit_quant_type: str = "nf4",
        num_train_epochs: int = 3,
        fp16: bool = False,
        bf16: bool = True,
        tf32: bool = True,
        use_flash_attn: bool = False,
        packing: bool = True,
        gradient_checkpointing: bool = True,
        optim: str = "paged_adamw_32bit",
        lr_scheduler_type: str = "constant",
        max_steps: int = -1,
        warmup_ratio: float = 0.03,
        group_by_length: bool = False,
        save_steps: int = 100,
        logging_steps: int = 100,
        merge_and_push: bool = False,
        device_map: Union[Dict[int, int], List[int], str] = "auto",
        output_dir: Optional[Union[str, os.PathLike]] = None,
        **kwargs,
    ):
        if dataset is None:
            raise ValueError("A dataset (either a Dataset or a name) must be provided.")

        self.local_rank = local_rank
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.weight_decay = weight_decay
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_r = lora_r
        self.max_seq_length = max_seq_length
        self.model_name = model_name
        self.dataset = dataset
        self.dataset_text_field = dataset_text_field
        self.formatting_func = formatting_func
        self.use_4bit = use_4bit
        self.use_nested_quant = use_nested_quant
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.num_train_epochs = num_train_epochs
        self.fp16 = fp16
        self.bf16 = bf16
        self.tf32 = tf32
        self.use_flash_attn = use_flash_attn
        self.packing = packing
        self.gradient_checkpointing = gradient_checkpointing
        self.optim = optim
        self.lr_scheduler_type = lr_scheduler_type
        self.max_steps = max_steps
        self.warmup_ratio = warmup_ratio
        self.group_by_length = group_by_length
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        self.merge_and_push = merge_and_push
        self.device_map = device_map

        if output_dir is None:
            output_dir = Path("output") / f"{self.model_name}"
        self.output_dir = Path(output_dir)

    def train(self):
        self.training_arguments = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            optim=self.optim,
            save_steps=self.save_steps,
            logging_steps=self.logging_steps,
            learning_rate=self.learning_rate,
            fp16=self.fp16,
            bf16=self.bf16,
            tf32=self.tf32,
            max_grad_norm=self.max_grad_norm,
            max_steps=self.max_steps,
            warmup_ratio=self.warmup_ratio,
            group_by_length=self.group_by_length,
            lr_scheduler_type=self.lr_scheduler_type,
        )

        self.model, self.peft_config, self.tokenizer = self.create_and_prepare_model(
            self.model_name,
            self.lora_alpha,
            self.lora_dropout,
            self.lora_r,
            self.use_flash_attn,
            self.use_4bit,
            self.use_nested_quant,
            self.bnb_4bit_compute_dtype,
            self.bnb_4bit_quant_type,
            self.device_map,
        )
        self.model.config.use_cache = False

        if isinstance(self.dataset, str):
            self.dataset = load_dataset(self.dataset, split="train")

        # Fix weird overflow issue with fp16 training
        self.tokenizer.padding_side = "right"

        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset,
            peft_config=self.peft_config,
            max_seq_length=self.max_seq_length,
            tokenizer=self.tokenizer,
            args=self.training_arguments,
            dataset_text_field=self.dataset_text_field,
            formatting_func=self.formatting_func,
            packing=self.packing,
        )

        self.trainer.train()

        if self.merge_and_push:
            self.merge_and_save()

    def create_and_prepare_model(
        self,
        model_name: str,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_r: int = 64,
        use_flash_attn: bool = False,
        use_4bit: bool = True,
        use_nested_quant: bool = True,
        bnb_4bit_compute_dtype: str = "bfloat16",
        bnb_4bit_quant_type: str = "nf4",
        device_map: Union[Dict[int, int], List[int], str] = "auto",
    ) -> Tuple[PreTrainedModel, PeftConfig, PreTrainedTokenizer]:
        compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

        if use_flash_attn:
            if torch.cuda.get_device_capability()[0] >= 8:
                from utils.llama_patch import replace_attn_with_flash_attn

                print("Using flash attention")
                replace_attn_with_flash_attn()

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=use_nested_quant,
        )

        if compute_dtype == torch.float16 and use_4bit:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print(
                    "Your GPU supports bfloat16, you can accelerate training with the argument --bf16"
                )
                print("=" * 80)

        # Load the entire model on the GPU 0
        # switch to `device_map = "auto"` for multi-GPU
        # device_map = {"": 0}
        device_map = device_map

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            use_auth_token=True,
        )

        # check: https://github.com/huggingface/transformers/pull/24906
        model.config.pretraining_tp = 1

        # Validate that the model is using flash attention, by comparing doc strings
        if use_flash_attn:
            from utils.llama_patch import forward

            assert (
                model.model.layers[0].self_attn.forward.__doc__ == forward.__doc__
            ), "Model is not using flash attention"

        peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_r,
            bias="none",
            task_type="CAUSAL_LM",
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        # prepare model for training
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
        return model, peft_config, tokenizer

    def merge_and_save(self):
        final_checkpoint_dir = self.output_dir / "final_checkpoints"
        self.trainer.model.save_pretrained(final_checkpoint_dir)

        # Free memory for merging weights
        del self.model
        torch.cuda.empty_cache()

        from peft import AutoPeftModelForCausalLM

        self.model = AutoPeftModelForCausalLM.from_pretrained(
            final_checkpoint_dir, low_cpu_mem_usage=True
        )
        self.model = self.model.merge_and_unload()

        output_merged_dir = os.path.join(
            final_checkpoint_dir, "final_merged_checkpoint"
        )
        self.model.save_pretrained(output_merged_dir, safe_serialization=True)
        self.tokenizer.save_pretrained(output_merged_dir)
