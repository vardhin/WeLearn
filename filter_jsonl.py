import json
import argparse
import os
from typing import Dict, Any

def filter_file_markers(input_file: str, output_file: str = None) -> None:
    """
    Remove rows containing __FILE_MARKER__ from a JSONL file
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file (optional, defaults to input_file_filtered.jsonl)
    """
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return
    
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_filtered.jsonl"
    
    total_rows = 0
    filtered_rows = 0
    removed_rows = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue
                
                total_rows += 1
                
                try:
                    # Parse JSON line
                    data = json.loads(line)
                    
                    # Check if __FILE_MARKER__ exists anywhere in the JSON data
                    if contains_file_marker(data):
                        removed_rows += 1
                        print(f"Removed row {line_num}: Contains __FILE_MARKER__")
                    else:
                        # Write the clean row to output file
                        json.dump(data, outfile, ensure_ascii=False)
                        outfile.write('\n')
                        filtered_rows += 1
                        
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num}: {e}")
                    continue
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return
    
    print(f"\nFiltering complete!")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Total rows processed: {total_rows}")
    print(f"Rows kept: {filtered_rows}")
    print(f"Rows removed: {removed_rows}")
    print(f"Removal rate: {(removed_rows / total_rows * 100):.1f}%" if total_rows > 0 else "0.0%")

def contains_file_marker(data: Dict[str, Any]) -> bool:
    """
    Recursively check if __FILE_MARKER__ exists anywhere in the data structure
    
    Args:
        data: Dictionary or other data structure to search
        
    Returns:
        True if __FILE_MARKER__ is found, False otherwise
    """
    
    def search_recursive(obj):
        if isinstance(obj, str):
            return "__FILE_MARKER__" in obj
        elif isinstance(obj, dict):
            return any(search_recursive(value) for value in obj.values())
        elif isinstance(obj, list):
            return any(search_recursive(item) for item in obj)
        else:
            return False
    
    return search_recursive(data)

def preview_file_markers(input_file: str, max_preview: int = 5) -> None:
    """
    Preview rows that contain __FILE_MARKER__ without filtering
    
    Args:
        input_file: Path to input JSONL file
        max_preview: Maximum number of examples to show
    """
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return
    
    found_count = 0
    total_rows = 0
    
    print(f"Previewing rows with __FILE_MARKER__ from: {input_file}")
    print("=" * 60)
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue
                
                total_rows += 1
                
                try:
                    data = json.loads(line)
                    
                    if contains_file_marker(data):
                        found_count += 1
                        
                        if found_count <= max_preview:
                            print(f"\nRow {line_num}:")
                            print(json.dumps(data, indent=2, ensure_ascii=False)[:500] + "..." if len(json.dumps(data)) > 500 else json.dumps(data, indent=2, ensure_ascii=False))
                            print("-" * 40)
                        
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num}: {e}")
                    continue
    
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    print(f"\nSummary:")
    print(f"Total rows: {total_rows}")
    print(f"Rows with __FILE_MARKER__: {found_count}")
    print(f"Percentage: {(found_count / total_rows * 100):.1f}%" if total_rows > 0 else "0.0%")
    
    if found_count > max_preview:
        print(f"Showing first {max_preview} examples (found {found_count} total)")

def batch_filter_files(input_dir: str, output_dir: str = None) -> None:
    """
    Filter all JSONL files in a directory
    
    Args:
        input_dir: Directory containing JSONL files
        output_dir: Output directory (optional, defaults to input_dir/filtered)
    """
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return
    
    if output_dir is None:
        output_dir = os.path.join(input_dir, "filtered")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all JSONL files
    jsonl_files = [f for f in os.listdir(input_dir) if f.endswith('.jsonl')]
    
    if not jsonl_files:
        print(f"No JSONL files found in: {input_dir}")
        return
    
    print(f"Found {len(jsonl_files)} JSONL files to process")
    print(f"Output directory: {output_dir}")
    print("-" * 50)
    
    for filename in jsonl_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        print(f"\nProcessing: {filename}")
        filter_file_markers(input_path, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter JSONL files to remove rows containing __FILE_MARKER__")
    parser.add_argument("input_file", help="Input JSONL file path")
    parser.add_argument("--output", "-o", help="Output file path (optional)")
    parser.add_argument("--preview", "-p", action="store_true", 
                       help="Preview rows with __FILE_MARKER__ without filtering")
    parser.add_argument("--max-preview", type=int, default=5,
                       help="Maximum number of preview examples (default: 5)")
    parser.add_argument("--batch", "-b", action="store_true",
                       help="Process all JSONL files in input directory")
    parser.add_argument("--output-dir", help="Output directory for batch processing")
    
    args = parser.parse_args()
    
    if args.batch:
        batch_filter_files(args.input_file, args.output_dir)
    elif args.preview:
        preview_file_markers(args.input_file, args.max_preview)
    else:
        filter_file_markers(args.input_file, args.output)