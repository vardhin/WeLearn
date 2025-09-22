# --- Standard Library Imports ---
import json
import torch
import gc
from datasets import Dataset

# --- Hugging Face Library Imports ---
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- Configuration ---
# ⚠️ UPDATE THESE PATHS to match your local file system
# The base model identifier from Hugging Face
base_model_name = "unsloth/gemma-3-270m-it"
# The local path to your saved adapter (the result of your fine-tuning)
adapter_path = "/home/vardhin/gemma-merged/merged_adapter" # Example: "C:/Users/YourUser/Desktop/my_project/merged_adapter"
# The local path to your dataset for testing
jsonl_file_path = "./trainingB.jsonl" # Example: "C:/Users/YourUser/Desktop/my_project/trainingB.jsonl"

NUM_TEST_QUESTIONS = 5

# --- 0. Confirm CPU Execution ---
print("="*80)
print("⚙️ Running on CPU. This will be very slow and use a lot of RAM.")
print("="*80)

# --- 1. Load Dataset and Create a Sample ---
print("\nLoading dataset and preparing sample...")
try:
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    dataset = Dataset.from_list(data)
    sample_dataset = dataset.select(range(NUM_TEST_QUESTIONS))
    print(f"✅ Loaded {len(dataset)} records. Using {NUM_TEST_QUESTIONS} for testing.")
except FileNotFoundError:
    print(f"❌ ERROR: Dataset not found at '{jsonl_file_path}'. Please check the file path.")
    exit()

# --- 2. Load Base Model on CPU ---
# NOTE: We are not using 4-bit quantization as it requires a GPU.
# The model will be loaded in full precision (float32).
print(f"\nLoading base model: {base_model_name}...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float32, # Explicitly use float32 for CPU
    device_map="cpu",
)
print("✅ Base model loaded.")


# --- 3. Generate and Print Responses from BASE MODEL ---
print(f"\n--- Generating responses from BASE MODEL ({base_model_name}) ---")
for item in sample_dataset:
    prompt = item.get('instruction', '') or item['conversations'][0]['value']
    original_output = item.get('output', '') or item['conversations'][1]['value']

    if not prompt:
        print("Skipping item with no valid prompt.")
        continue

    # Format the prompt for inference
    messages = [{"role": "user", "content": prompt}]
    # Input tensors are automatically placed on the CPU
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)

    # Generate the response
    outputs = model.generate(inputs, max_new_tokens=128, use_cache=True, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)[0]

    # Print the comparison
    print(f"\n❓ PROMPT:\n{prompt}")
    print("-------------------------------------------------")
    print(f"🤖 BASE MODEL RESPONSE:\n{response}")
    print("-------------------------------------------------")
    print(f"✅ ORIGINAL RESPONSE:\n{original_output}")
    print("=================================================")

# --- 4. IMPORTANT: Clear Base Model from Memory ---
print("\nClearing base model from memory...")
del model
gc.collect()
print("✅ Memory cleared.")


# --- 5. Load the Fine-Tuned Model (Base Model + PEFT Adapter) on CPU ---
print(f"\nLoading base model again to apply the adapter...")
# We must load the base model again first
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float32,
    device_map="cpu",
)

print(f"Applying PEFT adapter from: {adapter_path}...")
# Now, we apply the LoRA adapter to the base model
ft_model = PeftModel.from_pretrained(model, adapter_path)
print("✅ Fine-tuned model loaded.")


# --- 6. Generate and Print Responses from FINE-TUNED MODEL ---
print("\n--- Generating responses from FINE-TUNED MODEL ---")
for item in sample_dataset: # Using the SAME sample_dataset for a fair comparison
    prompt = item.get('instruction', '') or item['conversations'][0]['value']
    original_output = item.get('output', '') or item['conversations'][1]['value']

    if not prompt:
        print("Skipping item with no valid prompt.")
        continue

    # Format the prompt for inference
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)

    # Generate the response from the fine-tuned model
    outputs = ft_model.generate(inputs, max_new_tokens=128, use_cache=True, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)[0]

    # Print the comparison
    print(f"\n❓ PROMPT:\n{prompt}")
    print("-------------------------------------------------")
    print(f"🚀 FINE-TUNED MODEL RESPONSE:\n{response}")
    print("-------------------------------------------------")
    print(f"✅ ORIGINAL RESPONSE:\n{original_output}")
    print("=================================================")


# --- 7. Final Cleanup ---
print("\nClearing fine-tuned model from memory...")
del ft_model
del tokenizer
gc.collect()
print("✅ All done. Memory cleared.")