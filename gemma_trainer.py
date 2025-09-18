import json
import torch
from datasets import Dataset, load_dataset
import os
import subprocess

# --- Colab Specific Setup ---

def check_gpu_compatibility():
    """
    Checks for a compatible NVIDIA GPU and exits if one is not found.
    """
    if not torch.cuda.is_available():
        print("\n\n" + "="*80)
        print("❌ ERROR: No NVIDIA GPU detected. Unsloth requires a GPU.")
        print("Please change your runtime type to a GPU.")
        print("Go to 'Runtime' -> 'Change runtime type' -> 'Hardware accelerator' and select 'T4 GPU' or another GPU.")
        print("="*80 + "\n\n")
        os.kill(os.getpid(), 9)
    else:
        print("✅ Compatible NVIDIA GPU detected.")

def check_and_install_dependencies():
    """
    Checks if unsloth is installed. If not, it installs all required dependencies
    and then stops execution to allow for a manual runtime restart.
    """
    try:
        import unsloth
        print("✅ Dependencies are already installed.")
    except ImportError:
        print("Installing necessary packages. This may take a few minutes...")

        try:
            # Using subprocess to handle installation quietly
            # Unsloth's Gemma-3 support requires newer versions of dependencies
            subprocess.run(["pip", "install", "unsloth[colab-new]"], check=True, capture_output=True)
            subprocess.run(["pip", "install", "--no-deps", "trl", "peft", "accelerate", "bitsandbytes"], check=True, capture_output=True)
            print("✅ Installation successful.")

        except subprocess.CalledProcessError as e:
            print(f"❌ An error occurred during installation: {e}")
            print(e.stderr.decode())
            # Exit if installation fails
            exit()

        # Stop execution to force user to restart runtime
        print("\n\n" + "="*80)
        print("IMPORTANT: Dependencies installed. You MUST restart the runtime now.")
        print("Go to 'Runtime' -> 'Restart runtime' in the menu, and then run this cell again.")
        print("="*80 + "\n\n")
        os.kill(os.getpid(), 9)

def mount_google_drive():
    """Mounts Google Drive to the Colab environment."""
    print("Mounting Google Drive...")
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted successfully at /content/drive.")
    except ImportError:
        print("This script is designed to run in Google Colab. Could not find google.colab library.")
    except Exception as e:
        print(f"An error occurred while mounting Google Drive: {e}")
        exit()


# Now that setup is done, we can import the heavy libraries
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments

# --- Core Functions ---

def load_jsonl_dataset(file_path):
    """Load and prepare dataset from JSONL file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}. Please ensure the file exists in your Google Drive.")
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def format_conversation_structure(example):
    """
    Formats an example from various known structures into a unified
    list of messages format. This format is compatible with apply_chat_template.
    """
    messages = []
    if 'instruction' in example and 'output' in example:
        # Alpaca-style format
        messages.append({"role": "user", "content": example['instruction']})
        messages.append({"role": "assistant", "content": example['output']})
    elif 'conversations' in example:
        # ShareGPT-style format
        for message in example['conversations']:
            role = "user" if message.get('from') == 'human' else "assistant"
            content = message.get('value', '')
            messages.append({"role": role, "content": content})
    # Add more format checks here if needed
    else:
        raise ValueError("Dataset format not recognized. Expected 'instruction'+'output' or 'conversations' format.")

    return {"messages": messages}

def main():
    # --- Configuration ---
    # ⭐️ MODIFIED: Updated model name to Gemma-3 270M
    model_name = "unsloth/gemma-3-270m-it-bnb-4bit"
    # ⚠️ IMPORTANT: Update this path to point to your training data in Google Drive
    jsonl_file_path = "./trainingA.jsonl"
    # ⭐️ MODIFIED: Updated output directory for the new model
    output_dir = "./gemma3-270m-finetuned"
    max_seq_length = 2048

    # --- Model and Tokenizer Loading ---
    print("Loading model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,      # Auto-detect
        load_in_4bit=True, # This enables QLoRA quantization
    )

    # --- ⭐️ CRITICAL FOR GEMMA-3: Apply the correct chat template ---
    print("Applying Gemma-3 chat template...")
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma3",
    )

    # --- LoRA Configuration ---
    print("Configuring LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        # ⭐️ MODIFIED: Increased rank and alpha for potentially better performance
        r=128,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # --- Dataset Preparation ---
    print(f"Loading and preparing dataset from {jsonl_file_path}...")
    raw_data = load_jsonl_dataset(jsonl_file_path)
    dataset = Dataset.from_list(raw_data)

    # 1. First, map to create the 'messages' column with a standardized conversation structure
    structured_dataset = dataset.map(format_conversation_structure, remove_columns=list(dataset.features))

    # --- Dataset Splitting ---
    if len(structured_dataset) > 100:
        train_test_split = structured_dataset.train_test_split(test_size=0.1)
        train_dataset = train_test_split['train']
        eval_dataset = train_test_split['test']
        print(f"Dataset split into {len(train_dataset)} training and {len(eval_dataset)} evaluation samples.")
    else:
        train_dataset = structured_dataset
        eval_dataset = None
        print(f"Using the full dataset of {len(train_dataset)} samples for training.")


    # 2. Define a formatting function that uses the tokenizer to create the final training string.
    def formatting_prompts_func(example):
        # The .removeprefix('<bos>') is a recommended practice from the Unsloth Gemma-3 notebook
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False).removeprefix('<bos>')
        return {"text": text}

    # 3. Map this function to create the final 'text' column that the SFTTrainer will use
    train_dataset = train_dataset.map(formatting_prompts_func)
    if eval_dataset:
        eval_dataset = eval_dataset.map(formatting_prompts_func)

    # --- Training Arguments ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=8,
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
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=500 if eval_dataset else None,
        load_best_model_at_end=True if eval_dataset else False,
        report_to="none",  # Disable wandb logging
    )

    # --- Initialize Trainer ---
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text", # **CRITICAL**: Point to the new 'text' column
        max_seq_length=max_seq_length,
        args=training_args,
        dataset_num_proc=2, # Enable multiprocessing for faster data processing
    )

    # --- Start Training ---
    print("Starting training... 🚀")
    trainer.train()

    # --- Save Final Model ---
    print(f"Saving final LoRA adapters to {output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    # --- Save to GGUF Format (Optional) ---
    print("Attempting to save model in GGUF format...")
    try:
        # ⭐️ MODIFIED: Using `quantization_type` as per the latest Unsloth API for GGUF
        model.save_pretrained_gguf(output_dir, tokenizer, quantization_type="Q8_0")
        print("✅ GGUF model (Q8_0) saved successfully.")
    except Exception as e:
        print(f"Could not save GGUF model. This might be due to model compatibility or library versions. Error: {e}")

    print(f"\n🎉 Training completed! Model and tokenizer saved to {output_dir}")

if __name__ == "__main__":
    # Start the main training process
    main()