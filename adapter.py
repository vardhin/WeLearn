import torch
from unsloth import FastLanguageModel
import os

# ==============================================================================
# 🔧 1. CONFIGURATION
# ==============================================================================
# The base model from Hugging Face. Must match the model your adapter was trained on.
base_model_name = "unsloth/gemma-3-270m-it-bnb-4bit" 

# ⚠️ CHANGE THIS to the folder path of your downloaded adapter on your PC.
# Example for Windows: "C:/Users/YourUser/Desktop/my_gemma_adapter"
# Example for Mac/Linux: "/home/user/adapters/my_gemma_adapter"
local_adapter_path = "/home/vardhin/gemma3-270m-finetuned-A" 


# ==============================================================================
# ⚙️ 2. SCRIPT LOGIC (No changes needed below this line)
# ==============================================================================

def check_environment():
    """Checks for a compatible GPU and valid adapter path."""
    # Check for GPU
    if not torch.cuda.is_available():
        print("❌ ERROR: No NVIDIA GPU detected. This script requires a GPU to run.")
        exit()
    print("✅ NVIDIA GPU detected.")

    # Check if the adapter path is valid
    if not os.path.isdir(local_adapter_path):
        print(f"❌ ERROR: The adapter path '{local_adapter_path}' does not exist or is not a directory.")
        print("Please update the 'local_adapter_path' variable in the script.")
        exit()
    print(f"✅ Adapter folder found at: {local_adapter_path}")

def main():
    """Main function to load the model and run the chat interface."""
    check_environment()
    
    # --- Load the Base Model and Tokenizer ---
    print(f"\n🧠 Loading base model: '{base_model_name}'...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    print("✅ Base model loaded successfully.")

    # --- Attach the Local Adapter ---
    print(f"🔌 Attaching adapter from '{local_adapter_path}'...")
    model.load_adapter(local_adapter_path)
    print("✅ Adapter attached successfully. The model is now finetuned!")

    # --- Interactive Chat Loop ---
    print("\n\n💬 Model is ready. You can now start chatting.")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("-" * 50)

    while True:
        # Get user input
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("🤖 Goodbye!")
            break

        # Format the prompt using the model's chat template
        messages = [{"role": "user", "content": user_input}]
        inputs = tokenizer.apply_chat_template(
            messages, 
            return_tensors="pt", 
            add_generation_prompt=True
        ).to("cuda")

        # Generate the response
        outputs = model.generate(
            inputs, 
            max_new_tokens=256, 
            use_cache=True, 
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)[0]

        # Print the model's response
        print(f"🤖 Model: {response}")
        print("-" * 50)


if __name__ == "__main__":
    main()