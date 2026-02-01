#!/usr/bin/env python3
"""
Fine-tune a language model using LoRA adapters for Enterprise-to-Casual task.
Handles hardware detection (CPU/MPS/CUDA) and graceful fallbacks.
"""

import os
import json
import torch
from typing import Tuple

print("🚀 Starting fine-tuning script...\n")


def get_device() -> Tuple[str, bool]:
    """Detect available hardware acceleration."""

    if torch.cuda.is_available():
        device = "cuda"
        has_accelerator = True
        print(f"✅ Device: CUDA GPU ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = "mps"
        has_accelerator = True
        print("✅ Device: Apple Silicon MPS")
    else:
        device = "cpu"
        has_accelerator = False
        print("⚠️  Device: CPU (training will be slower)")

    return device, has_accelerator


def select_model(device: str, has_accelerator: bool) -> str:
    """Select appropriate model based on hardware."""

    # For CPU/MPS, use smaller model
    if device in ["cpu", "mps"]:
        model_name = "facebook/opt-350m"
        print(f"📦 Model: {model_name} (optimized for {device.upper()})")
    else:
        # If CUDA with enough memory, could use larger model
        model_name = "facebook/opt-350m"
        print(f"📦 Model: {model_name}")

    return model_name


def create_mock_adapters(model_name: str, output_dir: str, error_msg: str) -> None:
    """Create mock adapter config for API testing when training fails."""

    print(f"\n⚠️  Training failed: {error_msg}")
    print("📦 Creating mock adapters for API testing...\n")

    os.makedirs(output_dir, exist_ok=True)

    mock_config = {
        "adapter_type": "lora",
        "base_model_name_or_path": model_name,
        "r": 16,
        "lora_alpha": 32,
        "mock": True,
        "message": "Training failed - using mock adapters for API testing",
        "error": error_msg
    }

    config_path = os.path.join(output_dir, "adapter_config.json")
    with open(config_path, "w") as f:
        json.dump(mock_config, f, indent=2)

    print(f"✅ Mock adapter config created at: {config_path}")
    print("   API will use rule-based generation for testing\n")


def train_model():
    """Main training function with error handling."""

    # Detect hardware
    device, has_accelerator = get_device()

    # Select model
    model_name = select_model(device, has_accelerator)

    # Training config
    output_dir = "./fine_tuned_adapters"
    data_path = "data/train.jsonl"

    # Check if data exists
    if not os.path.exists(data_path):
        create_mock_adapters(
            model_name,
            output_dir,
            f"Training data not found at {data_path}. Run 'make data' first."
        )
        return

    print("\n📊 Training Configuration:")
    print(f"  Data: {data_path}")
    print(f"  Output: {output_dir}")
    print(f"  LoRA rank: 16")
    print(f"  Learning rate: 2e-4")
    print(f"  Max steps: 60")
    print()

    try:
        # Import training dependencies
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import load_dataset

        print("✅ Dependencies loaded successfully\n")

        # Load dataset
        print("📁 Loading dataset...")
        dataset = load_dataset('json', data_files=data_path, split='train')
        print(f"✅ Loaded {len(dataset)} training examples\n")

        # Format dataset
        def format_prompt(example):
            return {
                "text": f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
            }

        dataset = dataset.map(format_prompt)
        print("✅ Dataset formatted\n")

        # Load tokenizer
        print("📦 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("✅ Tokenizer loaded\n")

        # Tokenize dataset
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
                padding="max_length"
            )

        print("🔤 Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        print("✅ Dataset tokenized\n")

        # Load base model
        print("📦 Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="auto" if has_accelerator else None
        )
        print("✅ Base model loaded\n")

        # Configure LoRA
        print("🔧 Configuring LoRA adapters...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print("✅ LoRA adapters configured\n")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            max_steps=60,
            logging_steps=10,
            save_strategy="steps",
            save_steps=60,
            save_total_limit=1,
            fp16=False,  # Disable for CPU/MPS compatibility
            report_to="none",
            remove_unused_columns=False
        )

        # Create custom data collator
        from transformers import DataCollatorForLanguageModeling

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        # Create trainer
        print("🎓 Initializing trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator
        )
        print("✅ Trainer initialized\n")

        # Train
        print("🚀 Starting training (this may take 10-20 minutes on CPU)...\n")
        print("=" * 60)
        trainer.train()
        print("=" * 60)
        print()

        # Save model
        print("💾 Saving fine-tuned adapters...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"✅ Adapters saved to: {output_dir}\n")

        print("🎉 Training completed successfully!\n")

    except ImportError as e:
        create_mock_adapters(
            model_name,
            output_dir,
            f"Missing dependency: {str(e)}"
        )

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            create_mock_adapters(
                model_name,
                output_dir,
                "Out of memory. Try reducing batch size or using a smaller model."
            )
        else:
            create_mock_adapters(
                model_name,
                output_dir,
                f"Runtime error: {str(e)}"
            )

    except Exception as e:
        create_mock_adapters(
            model_name,
            output_dir,
            f"Unexpected error: {str(e)}"
        )


if __name__ == "__main__":
    train_model()
