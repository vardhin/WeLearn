import sqlite3
import json
import os
from typing import List, Dict, Optional

def get_training_data_from_db(database_path: str) -> List[tuple]:
    """Get question and answer data from specified database"""
    try:
        if not os.path.exists(database_path):
            print(f"Database file not found: {database_path}")
            return []
            
        with sqlite3.connect(database_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT question, answer
                FROM training_data
                ORDER BY created_at DESC
            """)
            
            results = cursor.fetchall()
        return results
    
    except Exception as e:
        print(f"Error getting training data: {e}")
        return []

def get_training_data_count_from_db(database_path: str) -> int:
    """Get total count of training data entries from specified database"""
    try:
        with sqlite3.connect(database_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM training_data")
            count = cursor.fetchone()[0]
        return count
    
    except Exception as e:
        print(f"Error getting training data count: {e}")
        return 0

def get_models_stats_from_db(database_path: str) -> List[tuple]:
    """Get statistics about training data by model from specified database"""
    try:
        with sqlite3.connect(database_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT generated_model, COUNT(*)
                FROM training_data
                GROUP BY generated_model
                ORDER BY COUNT(*) DESC
            """)
            
            results = cursor.fetchall()
        return results
    
    except Exception as e:
        print(f"Error getting model stats: {e}")
        return []

def create_unsloth_dataset(database_path: str, output_format: str = "jsonl", output_file: str = "training_dataset.jsonl") -> None:
    """
    Convert training data from SQLite database to Unsloth-compatible format
    
    Args:
        database_path: Path to the SQLite database file
        output_format: Format for output ("jsonl", "json", "alpaca", "sharegpt", "phi3")
        output_file: Output file name
    """
    
    # Get all training data from database
    training_data = get_training_data_from_db(database_path)
    
    if not training_data:
        print("No training data found in database")
        return
    
    # Convert to specified format
    if output_format.lower() == "alpaca":
        dataset = create_alpaca_format(training_data)
        save_alpaca_dataset(dataset, output_file)
    elif output_format.lower() == "json":
        dataset = create_conversation_format(training_data)
        save_json_dataset(dataset, output_file)
    elif output_format.lower() == "sharegpt":
        dataset = create_sharegpt_format(training_data)
        save_jsonl_dataset(dataset, output_file)
    elif output_format.lower() == "phi3":
        dataset = create_phi3_unsloth_format(training_data)
        save_jsonl_dataset(dataset, output_file)
    else:  # default to jsonl
        dataset = create_conversation_format(training_data)
        save_jsonl_dataset(dataset, output_file)
    
    print(f"Dataset created: {output_file}")
    print(f"Total samples: {len(dataset)}")
    print(f"Format used: {output_format}")

def create_conversation_format(training_data: List[tuple]) -> List[Dict]:
    """
    Create conversation format suitable for Unsloth (ChatML/Messages format)
    """
    dataset = []
    
    for row in training_data:
        question, answer = row
        
        # Create conversation format
        conversation = {
            "messages": [
                {
                    "role": "user",
                    "content": question
                },
                {
                    "role": "assistant", 
                    "content": answer
                }
            ]
        }
        
        dataset.append(conversation)
    
    return dataset

def create_alpaca_format(training_data: List[tuple]) -> List[Dict]:
    """
    Create Alpaca format for instruction tuning
    """
    dataset = []
    
    for row in training_data:
        question, answer = row
        
        # Create Alpaca format
        alpaca_entry = {
            "instruction": question,
            "input": "",  # Empty input for simple Q&A
            "output": answer
        }
        
        dataset.append(alpaca_entry)
    
    return dataset

def create_sharegpt_format(training_data: List[tuple]) -> List[Dict]:
    """
    Create ShareGPT format for conversation training
    """
    dataset = []
    
    for row in training_data:
        question, answer = row
        
        # Create ShareGPT format
        sharegpt_entry = {
            "conversations": [
                {
                    "from": "human",
                    "value": question
                },
                {
                    "from": "gpt",
                    "value": answer
                }
            ]
        }
        
        dataset.append(sharegpt_entry)
    
    return dataset

def create_phi3_unsloth_format(training_data: List[tuple]) -> List[Dict]:
    """
    Create Phi-3 format specifically for Unsloth training
    Uses ShareGPT style with "conversations" field as shown in Unsloth docs
    """
    dataset = []
    
    for row in training_data:
        question, answer = row
        
        # Phi-3 Unsloth format - exactly like ShareGPT but specifically for Phi-3
        phi3_entry = {
            "conversations": [
                {
                    "from": "human",
                    "value": question
                },
                {
                    "from": "gpt", 
                    "value": answer
                }
            ]
        }
        
        dataset.append(phi3_entry)
    
    return dataset

def save_jsonl_dataset(dataset: List[Dict], output_file: str) -> None:
    """Save dataset in JSONL format (one JSON object per line)"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in dataset:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

def save_json_dataset(dataset: List[Dict], output_file: str) -> None:
    """Save dataset in JSON format"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

def save_alpaca_dataset(dataset: List[Dict], output_file: str) -> None:
    """Save dataset in Alpaca JSON format"""
    if not output_file.endswith('.json'):
        output_file = output_file.replace('.jsonl', '.json')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

def filter_by_model(database_path: str, model_name: str, output_file: str = None, output_format: str = "jsonl") -> None:
    """
    Create dataset filtered by specific model
    """
    if output_file is None:
        output_file = f"training_dataset_{model_name.replace('/', '_')}.jsonl"
    
    try:
        with sqlite3.connect(database_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT question, answer
                FROM training_data
                WHERE generated_model = ?
                ORDER BY created_at DESC
            """, (model_name,))
            
            training_data = cursor.fetchall()
    except Exception as e:
        print(f"Error filtering by model: {e}")
        return
    
    if not training_data:
        print(f"No training data found for model: {model_name}")
        return
    
    # Use specified format for filtered data
    if output_format.lower() == "phi3":
        dataset = create_phi3_unsloth_format(training_data)
    elif output_format.lower() == "sharegpt":
        dataset = create_sharegpt_format(training_data)
    elif output_format.lower() == "alpaca":
        dataset = create_alpaca_format(training_data)
    else:
        dataset = create_conversation_format(training_data)
    
    save_jsonl_dataset(dataset, output_file)
    
    print(f"Filtered dataset created: {output_file}")
    print(f"Total samples for {model_name}: {len(dataset)}")

def create_train_test_split(database_path: str, train_ratio: float = 0.8, output_prefix: str = "dataset", output_format: str = "jsonl") -> None:
    """
    Create train/test split of the dataset
    """
    import random
    
    training_data = get_training_data_from_db(database_path)
    
    if not training_data:
        print("No training data found")
        return
    
    # Shuffle the data
    random.shuffle(training_data)
    
    # Calculate split point
    split_point = int(len(training_data) * train_ratio)
    
    train_data = training_data[:split_point]
    test_data = training_data[split_point:]
    
    # Create datasets based on format
    if output_format.lower() == "phi3":
        train_dataset = create_phi3_unsloth_format(train_data)
        test_dataset = create_phi3_unsloth_format(test_data)
    elif output_format.lower() == "sharegpt":
        train_dataset = create_sharegpt_format(train_data)
        test_dataset = create_sharegpt_format(test_data)
    elif output_format.lower() == "alpaca":
        train_dataset = create_alpaca_format(train_data)
        test_dataset = create_alpaca_format(test_data)
    else:
        train_dataset = create_conversation_format(train_data)
        test_dataset = create_conversation_format(test_data)
    
    # Save datasets
    train_file = f"{output_prefix}_train.jsonl"
    test_file = f"{output_prefix}_test.jsonl"
    
    save_jsonl_dataset(train_dataset, train_file)
    save_jsonl_dataset(test_dataset, test_file)
    
    print(f"Train dataset: {train_file} ({len(train_dataset)} samples)")
    print(f"Test dataset: {test_file} ({len(test_dataset)} samples)")

def print_dataset_stats(database_path: str) -> None:
    """Print statistics about the dataset"""
    total_count = get_training_data_count_from_db(database_path)
    models_stats = get_models_stats_from_db(database_path)
    
    print(f"Dataset Statistics for: {database_path}")
    print(f"Total samples: {total_count}")
    print(f"Models breakdown:")
    
    for model, count in models_stats:
        percentage = (count / total_count) * 100 if total_count > 0 else 0
        print(f"  {model}: {count} samples ({percentage:.1f}%)")

def validate_database(database_path: str) -> bool:
    """Validate that the database exists and has the correct structure"""
    if not os.path.exists(database_path):
        print(f"Error: Database file not found: {database_path}")
        return False
    
    try:
        with sqlite3.connect(database_path) as conn:
            cursor = conn.cursor()
            
            # Check if training_data table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='training_data'
            """)
            
            if not cursor.fetchone():
                print(f"Error: 'training_data' table not found in database: {database_path}")
                return False
                
            # Check table structure
            cursor.execute("PRAGMA table_info(training_data)")
            columns = [row[1] for row in cursor.fetchall()]
            
            required_columns = ['question', 'answer']
            missing_columns = [col for col in required_columns if col not in columns]
            
            if missing_columns:
                print(f"Error: Missing required columns in database: {missing_columns}")
                return False
                
    except Exception as e:
        print(f"Error validating database: {e}")
        return False
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert SQLite training data to Unsloth dataset format")
    parser.add_argument("--database", "-d", required=True,
                       help="Path to SQLite database file")
    parser.add_argument("--format", choices=["jsonl", "json", "alpaca", "sharegpt", "phi3"], 
                       default="jsonl", help="Output format")
    parser.add_argument("--output", default="training_dataset.jsonl", 
                       help="Output file name")
    parser.add_argument("--model", help="Filter by specific model")
    parser.add_argument("--split", action="store_true", 
                       help="Create train/test split")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                       help="Train ratio for split (default: 0.8)")
    parser.add_argument("--stats", action="store_true",
                       help="Show dataset statistics")
    
    args = parser.parse_args()
    
    # Validate database first
    if not validate_database(args.database):
        exit(1)
    
    if args.stats:
        print_dataset_stats(args.database)
    elif args.split:
        create_train_test_split(args.database, args.train_ratio, "dataset", args.format)
    elif args.model:
        filter_by_model(args.database, args.model, args.output, args.format)
    else:
        create_unsloth_dataset(args.database, args.format, args.output)