import json
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
import os

def load_jsonl_dataset(file_path):
    """Load and prepare dataset from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def format_chat_template(example):
    """Format the conversation for training"""
    # Assuming your JSONL has 'instruction' and 'output' fields
    # Adjust field names based on your actual data structure
    if 'instruction' in example and 'output' in example:
        conversation = f"<|user|>\n{example['instruction']}<|end|>\n<|assistant|>\n{example['output']}<|end|>"
    elif 'messages' in example:
        # If using chat format
        conversation = ""
        for message in example['conversations']:
            role = message['from']
            content = message['value']
            if role == 'human':
                conversation += f"<|user|>\n{content}<|end|>\n"
            elif role == 'gpt':
                conversation += f"<|assistant|>\n{content}<|end|>"
    else:
        raise ValueError("Dataset format not recognized. Expected 'instruction'+'output' or 'messages' format")
    
    return {"text": conversation}

def main():
    # Configuration
    model_name = "unsloth/Phi-3-mini-4k-instruct-bnb-4bit"
    jsonl_file_path = "your_dataset.jsonl"  # Update with your file path
    output_dir = "./phi3-finetuned"
    max_seq_length = 2048
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )
    
    # Configure LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    # Load and prepare dataset
    print("Loading dataset...")
    raw_data = load_jsonl_dataset(jsonl_file_path)
    dataset = Dataset.from_list(raw_data)
    
    # Format dataset
    formatted_dataset = dataset.map(format_chat_template, remove_columns=dataset.column_names)
    
    # Split dataset (optional)
    if len(formatted_dataset) > 100:
        train_test_split = formatted_dataset.train_test_split(test_size=0.1)
        train_dataset = train_test_split['train']
        eval_dataset = train_test_split['test']
    else:
        train_dataset = formatted_dataset
        eval_dataset = None
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        save_steps=500,
        save_total_limit=2,
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=500 if eval_dataset else None,
        load_best_model_at_end=True if eval_dataset else False,
        report_to=None,  # Disable wandb logging
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8),
        args=training_args,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    print("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save to GGUF format (optional)
    print("Saving to GGUF format...")
    model.save_pretrained_gguf(output_dir, tokenizer, quantization_method="q4_k_m")
    
    print(f"Training completed! Model saved to {output_dir}")

if __name__ == "__main__":
    main()