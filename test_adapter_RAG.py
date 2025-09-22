# --- Standard Library Imports ---
import json
import torch
import gc
import os

# --- Hugging Face Library Imports ---
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import Dataset

# --- RAG Library Imports (LangChain) ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# --- Configuration ---
# ⚠️ UPDATE THESE PATHS to match your local file system
# Model and Data Paths
base_model_name = "unsloth/gemma-3-270m-it"
adapter_path = "/home/vardhin/gemma-merged/merged_adapter" 
jsonl_file_path = "./trainingB.jsonl" 

# RAG Configuration
# Create a folder named "knowledge_base" and place your .txt files inside it
KNOWLEDGE_BASE_PATH = "./collaboration_expert_text" 
# This is a popular, lightweight embedding model that runs well on CPU
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" 
#EMBEDDING_MODEL_NAME = "google/embedding-gemma-2b"
NUM_RETRIEVED_DOCS = 3 # Number of relevant text chunks to retrieve

# Testing Configuration
NUM_TEST_QUESTIONS = 5

# --- 0. Confirm CPU Execution ---
print("="*80)
print("⚙️ Running on CPU. This will be very slow and use a lot of RAM.")
print("="*80)

# --- 1. Load Fine-Tuned Model ---
# We only need to load the fine-tuned model once for both tests.
print(f"\nLoading base model: {base_model_name}...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float32,
    device_map="cpu",
)

print(f"Applying PEFT adapter from: {adapter_path}...")
ft_model = PeftModel.from_pretrained(model, adapter_path)
print("✅ Fine-tuned model loaded and ready.")


# --- 2. Setup the RAG Pipeline ---
def setup_rag_pipeline(folder_path, embedding_model_name):
    """
    Loads documents, splits them, creates embeddings, and sets up a retriever.
    """
    print("\nSetting up the RAG pipeline...")
    try:
        # Load documents from the specified folder (for .txt files)
        loader = DirectoryLoader(folder_path, glob="**/*.txt", show_progress=True)
        documents = loader.load()
        if not documents:
            print(f"❌ ERROR: No .txt files found in '{folder_path}'. Please check the path and file types.")
            return None
        print(f"✅ Loaded {len(documents)} documents.")

        # Split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        print(f"✅ Split documents into {len(texts)} chunks.")

        # Create embeddings
        print(f"✅ Loading embedding model: '{embedding_model_name}'...")
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

        # Create a FAISS vector store (an in-memory index)
        print("✅ Creating FAISS vector store...")
        vector_store = FAISS.from_documents(texts, embeddings)
        print("✅ RAG pipeline is ready.")
        
        # Return the retriever object
        return vector_store.as_retriever(search_kwargs={"k": NUM_RETRIEVED_DOCS})

    except Exception as e:
        print(f"❌ An error occurred during RAG setup: {e}")
        return None

# Create the retriever
retriever = setup_rag_pipeline(KNOWLEDGE_BASE_PATH, EMBEDDING_MODEL_NAME)
if not retriever:
    exit()


# --- 3. Load Test Dataset ---
print("\nLoading test dataset...")
try:
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    dataset = Dataset.from_list(data)
    sample_dataset = dataset.select(range(NUM_TEST_QUESTIONS))
    print(f"✅ Loaded {len(dataset)} records. Using {NUM_TEST_QUESTIONS} for testing.")
except FileNotFoundError:
    print(f"❌ ERROR: Dataset not found at '{jsonl_file_path}'. Please check the file path.")
    exit()


# --- 4. Run Comparison: Fine-Tuned vs. Fine-Tuned + RAG ---
print("\n--- Starting Comparison ---")
for item in sample_dataset:
    question = item.get('instruction', '') or item['conversations'][0]['value']
    original_output = item.get('output', '') or item['conversations'][1]['value']

    if not question:
        continue
    
    print(f"\n\n==================== QUESTION ====================")
    print(f"❓ {question}")
    print("==================================================")

    # --- Test 1: Fine-Tuned Model WITHOUT RAG ---
    messages_without_rag = [{"role": "user", "content": question}]
    inputs_without_rag = tokenizer.apply_chat_template(messages_without_rag, return_tensors="pt", add_generation_prompt=True)
    outputs_without_rag = ft_model.generate(inputs_without_rag, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)
    response_without_rag = tokenizer.batch_decode(outputs_without_rag[:, inputs_without_rag.shape[1]:], skip_special_tokens=True)[0]

    # --- Test 2: Fine-Tuned Model WITH RAG ---
    # 1. Retrieve relevant context
    retrieved_docs = retriever.get_relevant_documents(question)
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # 2. Create the augmented prompt
    rag_prompt_template = f"""Based on the context below, please provide a detailed and accurate answer to the user's question.

### Context:
{context_text}

### Question:
{question}

### Answer:
"""
    messages_with_rag = [{"role": "user", "content": rag_prompt_template}]
    inputs_with_rag = tokenizer.apply_chat_template(messages_with_rag, return_tensors="pt", add_generation_prompt=True)
    outputs_with_rag = ft_model.generate(inputs_with_rag, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)
    response_with_rag = tokenizer.batch_decode(outputs_with_rag[:, inputs_with_rag.shape[1]:], skip_special_tokens=True)[0]


    # --- Print Results ---
    print("\n🧠 RESPONSE (Fine-Tuned Model WITHOUT RAG):")
    print(response_without_rag)
    print("--------------------------------------------------")
    print("\n🚀 RESPONSE (Fine-Tuned Model WITH RAG):")
    print(response_with_rag)
    print("--------------------------------------------------")
    print("\n✅ ORIGINAL RESPONSE (from dataset):")
    print(original_output)
    print("==================================================")


# --- 5. Final Cleanup ---
print("\nClearing model from memory...")
del model
del ft_model
del tokenizer
del retriever
gc.collect()
print("✅ All done. Memory cleared.")