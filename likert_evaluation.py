import pandas as pd
import requests
import json
import time
import os
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

class LikertEvaluator:
    def __init__(self):
        self.gemini_api_key = None
        self.selected_model = "gemini-2.5-flash"  # Default model
        self.batch_size = 10  # Process 10 questions at a time
        
        # Load environment variables
        load_dotenv()
        
    def get_gemini_api_key(self) -> Optional[str]:
        """Get Gemini API key from .env or user input"""
        # Try to get from .env file first
        api_key = os.getenv('GEMINI_API_KEY')
        
        if api_key:
            print("✓ Found Gemini API key in .env file")
            return api_key
        else:
            print("✗ Gemini API key not found in .env file")
            api_key = input("Enter your Gemini API key: ").strip()
            if not api_key:
                print("API key is required for Gemini models.")
                return None
            return api_key
    
    def test_gemini_api_connection(self) -> bool:
        """Test if Gemini API is accessible"""
        print("🔍 Testing Gemini API connection...")
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.selected_model}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.gemini_api_key
        }
        
        test_data = {
            "contents": [{
                "parts": [{
                    "text": "Say hello"
                }]
            }]
        }
        
        try:
            response = requests.post(url, headers=headers, json=test_data, timeout=30)
            if response.status_code == 200:
                print("✅ API connection successful!")
                return True
            else:
                print(f"❌ API test failed with status {response.status_code}: {response.text}")
                return False
        except Exception as e:
            print(f"❌ API test failed: {e}")
            return False
    
    def load_excel_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load Excel file and validate columns"""
        try:
            df = pd.read_excel(file_path)
            
            # Check required columns
            required_columns = ['question', 'ai-answer', 'original-answer']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"❌ Missing required columns: {missing_columns}")
                print(f"📋 Available columns: {list(df.columns)}")
                return None
            
            print(f"✅ Excel file loaded successfully")
            print(f"📊 Total rows: {len(df)}")
            print(f"📋 Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading Excel file: {e}")
            return None
    
    def create_evaluation_prompt(self, questions_batch: List[Dict]) -> str:
        """Create prompt for evaluating a batch of questions"""
        prompt = """You are an expert evaluator tasked with scoring AI-generated answers compared to original answers using a Likert scale.

Please evaluate each AI answer on how well it matches the quality, accuracy, and completeness of the original answer.

Scoring criteria (1-5 scale):
1 = Very Poor: AI answer is completely wrong, irrelevant, or missing key information
2 = Poor: AI answer has some relevant content but significant errors or omissions
3 = Average: AI answer is partially correct but lacks depth or has minor errors
4 = Good: AI answer is mostly accurate and complete with minor improvements needed
5 = Excellent: AI answer matches or exceeds the quality of the original answer

For each question, provide ONLY the numeric score (1-5) in this exact format:
Q1: [score]
Q2: [score]
Q3: [score]
... and so on

Here are the questions with their AI answers and original answers to evaluate:

"""
        
        for i, item in enumerate(questions_batch, 1):
            prompt += f"""
Question {i}: {item['question']}

AI Answer {i}: {item['ai_answer']}

Original Answer {i}: {item['original_answer']}

---
"""
        
        prompt += "\nPlease provide scores for each question (Q1: [score], Q2: [score], etc.):"
        
        return prompt
    
    def call_gemini_api(self, prompt: str) -> Optional[str]:
        """Call Gemini API to get evaluation scores"""
        try:
            print(f"  🔄 Making API call for evaluation...")
            
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.selected_model}:generateContent"
            
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.gemini_api_key
            }
            
            data = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }]
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=180)
            
            print(f"  📊 Response status: {response.status_code}")
            
            response.raise_for_status()
            
            result = response.json()
            if "candidates" in result and result["candidates"]:
                if "content" in result["candidates"][0] and "parts" in result["candidates"][0]["content"]:
                    response_text = result["candidates"][0]["content"]["parts"][0]["text"]
                    print(f"  ✅ API call successful")
                    return response_text
                else:
                    print(f"  ⚠️ Response missing expected content structure")
            else:
                print(f"  ⚠️ No candidates in response")
                
        except requests.exceptions.HTTPError as e:
            print(f"  🚫 HTTP error {e.response.status_code}: {e.response.text[:200]}")
        except requests.exceptions.Timeout:
            print(f"  ⏰ Request timed out")
        except Exception as e:
            print(f"  🚫 Unexpected error: {e}")

        return None
    
    def parse_scores(self, response: str, batch_size: int) -> List[Optional[int]]:
        """Parse scores from API response"""
        scores = []
        
        if not response:
            return [None] * batch_size
        
        lines = response.strip().split('\n')
        
        for i in range(1, batch_size + 1):
            score = None
            
            # Look for pattern like "Q1: 4" or "Question 1: 4"
            for line in lines:
                line = line.strip()
                if f"Q{i}:" in line or f"Question {i}:" in line:
                    # Extract number after colon
                    parts = line.split(':')
                    if len(parts) > 1:
                        try:
                            score_text = parts[1].strip()
                            # Extract first number found
                            score_num = int(''.join(filter(str.isdigit, score_text)))
                            if 1 <= score_num <= 5:
                                score = score_num
                                break
                        except (ValueError, IndexError):
                            continue
            
            scores.append(score)
        
        return scores
    
    def evaluate_excel_file(self, excel_path: str, output_path: str = None):
        """Main method to evaluate Excel file"""
        print("=== Likert Scale Evaluation ===")
        
        # Get API key
        api_key = self.get_gemini_api_key()
        if not api_key:
            return False
        self.gemini_api_key = api_key
        
        # Test API connection
        if not self.test_gemini_api_connection():
            print("⚠️ API test failed. Continuing anyway...")
        
        # Load Excel file
        df = self.load_excel_file(excel_path)
        if df is None:
            return False
        
        # Check if evaluation column already exists
        if 'likert_score' in df.columns:
            print("⚠️ 'likert_score' column already exists")
            overwrite = input("Overwrite existing scores? (y/n): ").strip().lower()
            if overwrite not in ['y', 'yes']:
                print("🚫 Evaluation cancelled.")
                return False
        
        # Initialize scores column
        df['likert_score'] = None
        
        # Process in batches
        total_rows = len(df)
        processed_rows = 0
        failed_rows = 0
        
        print(f"\n🚀 Starting evaluation...")
        print(f"📊 Total rows: {total_rows}")
        print(f"📦 Batch size: {self.batch_size}")
        print(f"🔄 Estimated batches: {(total_rows + self.batch_size - 1) // self.batch_size}")
        
        for batch_start in range(0, total_rows, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_rows)
            batch_num = (batch_start // self.batch_size) + 1
            
            print(f"\n{'='*60}")
            print(f"📦 Processing Batch {batch_num} (rows {batch_start + 1}-{batch_end})")
            
            # Prepare batch data
            questions_batch = []
            for idx in range(batch_start, batch_end):
                row = df.iloc[idx]
                questions_batch.append({
                    'question': str(row['question']),
                    'ai_answer': str(row['ai-answer']),
                    'original_answer': str(row['original-answer'])
                })
            
            # Create prompt and get evaluation
            prompt = self.create_evaluation_prompt(questions_batch)
            response = self.call_gemini_api(prompt)
            
            if response:
                # Parse scores
                scores = self.parse_scores(response, len(questions_batch))
                
                # Update DataFrame
                for i, score in enumerate(scores):
                    row_idx = batch_start + i
                    if score is not None:
                        df.at[row_idx, 'likert_score'] = score
                        processed_rows += 1
                    else:
                        failed_rows += 1
                        print(f"    ⚠️ Failed to parse score for row {row_idx + 1}")
                
                success_count = sum(1 for s in scores if s is not None)
                print(f"  ✅ Successfully evaluated {success_count}/{len(questions_batch)} items in this batch")
                
            else:
                print(f"  ❌ Failed to get evaluation for batch {batch_num}")
                failed_rows += len(questions_batch)
            
            # Add delay between batches to avoid rate limiting
            if batch_start + self.batch_size < total_rows:
                print(f"  ⏸️ Waiting 2 seconds before next batch...")
                time.sleep(2)
            
            # Progress update
            total_processed = batch_end
            print(f"  📈 Progress: {total_processed}/{total_rows} rows processed")
        
        # Save results
        if output_path is None:
            # Create output filename based on input filename
            base_name = os.path.splitext(excel_path)[0]
            output_path = f"{base_name}_evaluated.xlsx"
        
        try:
            df.to_excel(output_path, index=False)
            print(f"\n✅ Results saved to: {output_path}")
        except Exception as e:
            print(f"\n❌ Error saving file: {e}")
            return False
        
        # Final summary
        print(f"\n{'='*60}")
        print(f"🎯 Final Summary")
        print(f"{'='*60}")
        print(f"📊 Total rows: {total_rows}")
        print(f"✅ Successfully evaluated: {processed_rows}")
        print(f"❌ Failed evaluations: {failed_rows}")
        
        if total_rows > 0:
            success_rate = (processed_rows / total_rows) * 100
            print(f"📈 Success rate: {success_rate:.1f}%")
        
        print(f"💾 Output file: {output_path}")
        print("\n🎉 Evaluation complete!")
        
        return True

def main():
    evaluator = LikertEvaluator()
    
    # Get Excel file path from user
    excel_path = "./base_model_evaluation_output.xlsx"
    
    if not os.path.exists(excel_path):
        print(f"❌ File not found: {excel_path}")
        return
    
    # Optional: Get output path
    output_path = input("Enter output file path (or press Enter for auto-generated): ").strip()
    if not output_path:
        output_path = None
    
    # Run evaluation
    evaluator.evaluate_excel_file(excel_path, output_path)

if __name__ == "__main__":
    main()