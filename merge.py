import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import os
import argparse

def merge_lora_adapters(base_model_path, adapter1_path, adapter2_path, output_path, merge_method="average"):
    """
    Merge two QLoRA adapters into a single adapter.
    
    Args:
        base_model_path: Path to the base Gemma model
        adapter1_path: Path to first LoRA adapter
        adapter2_path: Path to second LoRA adapter
        output_path: Path to save merged adapter
        merge_method: "average" or "concat" (default: "average")
    """
    
    # Validate adapter paths
    print("Validating adapter paths...")
    for i, adapter_path in enumerate([adapter1_path, adapter2_path], 1):
        if not os.path.exists(adapter_path):
            raise ValueError(f"Adapter {i} path does not exist: {adapter_path}")
        
        config_file = os.path.join(adapter_path, "adapter_config.json")
        if not os.path.exists(config_file):
            raise ValueError(f"Adapter {i} missing adapter_config.json: {adapter_path}")
        
        print(f"✅ Adapter {i} validated: {adapter_path}")
    
    print("Loading base model...")
    # Load the base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="cpu",  # Use CPU since no GPU available
        low_cpu_mem_usage=True
    )
    
    print("Loading first adapter...")
    # Load first adapter
    model_with_adapter1 = PeftModel.from_pretrained(
        base_model,
        adapter1_path,
        adapter_name="adapter1"
    )
    
    print("Loading second adapter...")
    # Load second adapter
    model_with_adapter1.load_adapter(adapter2_path, adapter_name="adapter2")
    
    if merge_method == "average":
        print("Merging adapters using average method...")
        # Average the adapters
        adapter_weights = [0.5, 0.5]  # Equal weights
        model_with_adapter1.add_weighted_adapter(
            ["adapter1", "adapter2"],
            adapter_weights,
            "merged_adapter"
        )
    elif merge_method == "concat":
        print("Merging adapters using concatenation method...")
        # This is more complex - you might want to use average instead
        model_with_adapter1.add_weighted_adapter(
            ["adapter1", "adapter2"],
            [0.5, 0.5],
            "merged_adapter"
        )
    
    # Set the merged adapter as active
    model_with_adapter1.set_adapter("merged_adapter")
    
    # Create output directory structure
    merged_output_path = os.path.join(output_path, "merged_adapter")
    os.makedirs(merged_output_path, exist_ok=True)
    
    print(f"Saving merged adapter to {merged_output_path}...")
    
    # Save only the merged adapter (not the full model)
    model_with_adapter1.save_pretrained(merged_output_path, selected_adapters=["merged_adapter"])
    
    # Also save tokenizer to the main output path
    tokenizer.save_pretrained(output_path)
    
    # Verify files were saved
    saved_files = os.listdir(merged_output_path)
    print(f"Saved files in merged_adapter: {saved_files}")
    
    print("Merge completed successfully!")

def alternative_merge_method(base_model_path, adapter1_path, adapter2_path, output_path):
    """
    Alternative method: Load adapters separately and manually merge weights
    """
    print("Using alternative merge method...")
    
    # Validate adapter paths
    print("Validating adapter paths...")
    for i, adapter_path in enumerate([adapter1_path, adapter2_path], 1):
        if not os.path.exists(adapter_path):
            raise ValueError(f"Adapter {i} path does not exist: {adapter_path}")
        
        config_file = os.path.join(adapter_path, "adapter_config.json")
        if not os.path.exists(config_file):
            raise ValueError(f"Adapter {i} missing adapter_config.json: {adapter_path}")
        
        print(f"✅ Adapter {i} validated: {adapter_path}")
    
    # Load configs
    config1 = PeftConfig.from_pretrained(adapter1_path)
    config2 = PeftConfig.from_pretrained(adapter2_path)
    
    # Load base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    
    # Load first adapter and merge to base
    print("Merging first adapter...")
    model1 = PeftModel.from_pretrained(base_model, adapter1_path)
    merged_model1 = model1.merge_and_unload()
    
    # Load second adapter on the already merged model
    print("Merging second adapter...")
    model2 = PeftModel.from_pretrained(merged_model1, adapter2_path)
    final_merged_model = model2.merge_and_unload()
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save the final merged model
    print(f"Saving final merged model to {output_path}...")
    final_merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    # Verify files were saved
    saved_files = os.listdir(output_path)
    print(f"Saved files: {saved_files}")
    
    print("Alternative merge completed!")

def simple_weight_averaging_merge(base_model_path, adapter1_path, adapter2_path, output_path):
    """
    Simple method: Create a new adapter with averaged weights
    """
    print("Using simple weight averaging merge method...")
    
    # Validate paths
    for i, path in enumerate([adapter1_path, adapter2_path], 1):
        if not os.path.exists(os.path.join(path, "adapter_config.json")):
            raise ValueError(f"Invalid adapter {i} path: {path}")
    
    import torch
    import safetensors.torch as st
    import json
    
    # Load adapter configs
    with open(os.path.join(adapter1_path, "adapter_config.json"), 'r') as f:
        config1 = json.load(f)
    
    with open(os.path.join(adapter2_path, "adapter_config.json"), 'r') as f:
        config2 = json.load(f)
    
    # Check if configs are compatible
    if config1.get('target_modules') != config2.get('target_modules'):
        print("Warning: Adapters have different target modules")
    
    # Load weights from both adapters
    print("Loading adapter weights...")
    
    # Try to find weight files
    def find_weight_file(adapter_path):
        files = os.listdir(adapter_path)
        safetensor_files = [f for f in files if f.endswith('.safetensors')]
        bin_files = [f for f in files if f.endswith('.bin')]
        
        if safetensor_files:
            return os.path.join(adapter_path, safetensor_files[0]), 'safetensors'
        elif bin_files:
            return os.path.join(adapter_path, bin_files[0]), 'torch'
        else:
            raise ValueError(f"No weight files found in {adapter_path}")
    
    weight_file1, format1 = find_weight_file(adapter1_path)
    weight_file2, format2 = find_weight_file(adapter2_path)
    
    # Load weights
    if format1 == 'safetensors':
        weights1 = st.load_file(weight_file1)
    else:
        weights1 = torch.load(weight_file1, map_location='cpu')
    
    if format2 == 'safetensors':
        weights2 = st.load_file(weight_file2)
    else:
        weights2 = torch.load(weight_file2, map_location='cpu')
    
    # Average weights
    print("Averaging weights...")
    merged_weights = {}
    all_keys = set(weights1.keys()) | set(weights2.keys())
    
    for key in all_keys:
        if key in weights1 and key in weights2:
            merged_weights[key] = (weights1[key] + weights2[key]) / 2.0
        elif key in weights1:
            merged_weights[key] = weights1[key]
        else:
            merged_weights[key] = weights2[key]
    
    # Create output directory
    merged_output_path = os.path.join(output_path, "merged_adapter")
    os.makedirs(merged_output_path, exist_ok=True)
    
    # Save merged weights
    print(f"Saving merged weights to {merged_output_path}...")
    output_weight_file = os.path.join(merged_output_path, "adapter_model.safetensors")
    st.save_file(merged_weights, output_weight_file)
    
    # Save config (use config from first adapter as base)
    output_config_file = os.path.join(merged_output_path, "adapter_config.json")
    with open(output_config_file, 'w') as f:
        json.dump(config1, f, indent=2)
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)
    
    # Verify
    saved_files = os.listdir(merged_output_path)
    print(f"Saved files in merged_adapter: {saved_files}")
    
    print("Simple weight averaging merge completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge two QLoRA adapters")
    parser.add_argument("--base_model", type=str, default="google/gemma-3-270m-it", 
                       help="Base model path or HuggingFace model name")
    parser.add_argument("--adapter1", type=str, required=True, 
                       help="Path to first LoRA adapter")
    parser.add_argument("--adapter2", type=str, required=True, 
                       help="Path to second LoRA adapter")
    parser.add_argument("--output", type=str, required=True, 
                       help="Output path for merged adapter")
    parser.add_argument("--method", type=str, choices=["average", "sequential", "simple"], 
                       default="simple", help="Merge method")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    try:
        if args.method == "average":
            merge_lora_adapters(
                args.base_model, 
                args.adapter1, 
                args.adapter2, 
                args.output, 
                "average"
            )
        elif args.method == "sequential":
            alternative_merge_method(
                args.base_model,
                args.adapter1,
                args.adapter2,
                args.output
            )
        elif args.method == "simple":
            simple_weight_averaging_merge(
                args.base_model,
                args.adapter1,
                args.adapter2,
                args.output
            )
    except Exception as e:
        print(f"Error during merge: {e}")
        print("Trying simple weight averaging method...")
        try:
            simple_weight_averaging_merge(
                args.base_model,
                args.adapter1,
                args.adapter2,
                args.output
            )
        except Exception as e2:
            print(f"All merge methods failed. Last error: {e2}")
            raise