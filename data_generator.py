import uuid
import json
import re
import time
import os
import hashlib
from typing import List, Dict, Optional, Tuple
import requests
import subprocess
from dotenv import load_dotenv

import training_data_store

class DataGenerator:
    def __init__(self):
        self.selected_model = None
        self.model_type = None  # 'gemini' or 'ollama'
        self.gemini_api_key = None
        self.processed_files = set()  # Track files being processed in current session
        self.text_files_folder = "collaboration_expert_text"
        
        # Load environment variables
        load_dotenv()
        
    def get_file_title_hash(self, title: str) -> str:
        """Generate a hash from the file title to use as UID"""
        return hashlib.md5(title.encode('utf-8')).hexdigest()
        
    def get_text_files(self) -> List[Tuple[str, str, str]]:
        """Get all text files from the field_expert_text folder"""
        text_files = []
        
        if not os.path.exists(self.text_files_folder):
            print(f"Error: Folder '{self.text_files_folder}' not found.")
            return []
        
        try:
            for filename in os.listdir(self.text_files_folder):
                if filename.endswith('.txt'):
                    file_path = os.path.join(self.text_files_folder, filename)
                    
                    # Read file content
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                        
                        if content:  # Only add non-empty files
                            # Use filename without extension as title
                            title = os.path.splitext(filename)[0]
                            # Use file path as unique identifier
                            uid = file_path
                            text_files.append((uid, title, content))
                    
                    except Exception as e:
                        print(f"Error reading file {filename}: {e}")
                        continue
            
            print(f"Found {len(text_files)} text files in '{self.text_files_folder}' folder")
            return text_files
            
        except Exception as e:
            print(f"Error accessing folder '{self.text_files_folder}': {e}")
            return []
        
    def get_gemini_api_key(self) -> Optional[str]:
        """Get Gemini API key from .env or user input"""
        # Try to get from .env file first
        api_key = os.getenv('GEMINI_API_KEY')
        
        if api_key:
            print("✓ Found Gemini API key in .env file")
            return api_key
        else:
            print("✗ Gemini API key not found in .env file")
            print("\nTo avoid entering the API key each time, add it to your .env file:")
            print("Format: GEMINI_API_KEY=your_api_key_here")
            print("Example: GEMINI_API_KEY=AIzaSyBxxxxxxxxxxxxxxxxxxxxxxx")
            print("\nAlternatively, you can enter it now:")
            
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
        
    def get_gemini_models(self) -> List[str]:
        """Get available Gemini models from API"""
        if not self.gemini_api_key:
            return []
        
        url = "https://generativelanguage.googleapis.com/v1beta/models"
        headers = {
            "x-goog-api-key": self.gemini_api_key
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            models = []
            
            if "models" in result:
                for model in result["models"]:
                    model_name = model.get("name", "")
                    # Extract just the model name (remove "models/" prefix)
                    if model_name.startswith("models/"):
                        model_name = model_name[7:]
                    
                    # Filter for text generation models only
                    supported_methods = model.get("supportedGenerationMethods", [])
                    if "generateContent" in supported_methods:
                        models.append(model_name)
            
            return sorted(models)  # Sort alphabetically
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching Gemini models: {e}")
            # Fallback to hardcoded models if API fails
            return [
                "gemini-2.0-flash",
                "gemini-1.5-flash",
                "gemini-1.5-pro", 
                "gemini-1.0-pro"
            ]
        except KeyError as e:
            print(f"Unexpected API response format: {e}")
            # Fallback to hardcoded models
            return [
                "gemini-2.0-flash",
                "gemini-1.5-flash",
                "gemini-1.5-pro",
                "gemini-1.0-pro"
            ]
    
    def get_ollama_models(self) -> List[str]:
        """Get available Ollama models"""
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                models = []
                for line in lines:
                    if line.strip():
                        model_name = line.split()[0]
                        models.append(model_name)
                return models
            else:
                print("Error getting Ollama models:", result.stderr)
                return []
        except FileNotFoundError:
            print("Ollama not found. Please install Ollama first.")
            return []
    
    def select_model(self):
        """Interactive model selection"""
        print("\n=== Model Selection ===")
        print("1. Gemini API (Online)")
        print("2. Ollama (Offline)")
        
        choice = input("Select model type (1 or 2): ")


        if choice == "1":
            self.model_type = "gemini"
            
            # Get API key from .env or user input
            api_key = self.get_gemini_api_key()
            if not api_key:
                return False
            self.gemini_api_key = api_key
            
            print("\nFetching available Gemini models...")
            gemini_models = self.get_gemini_models()
            
            if not gemini_models:
                print("No Gemini models available.")
                return False
                
            print("\nAvailable Gemini models:")
            for i, model in enumerate(gemini_models, 1):
                print(f"{i}. {model}")
            
            model_choice = input(f"Select model (1-{len(gemini_models)}): ").strip()
            try:
                model_idx = int(model_choice) - 1
                if 0 <= model_idx < len(gemini_models):
                    self.selected_model = gemini_models[model_idx]
                    print(f"Selected: {self.selected_model}")
                    
                    # Test API connection
                    if not self.test_gemini_api_connection():
                        print("⚠️ API test failed. You can still continue, but expect errors.")
                        continue_anyway = input("Continue anyway? (y/n): ").strip().lower()
                        if continue_anyway not in ['y', 'yes']:
                            return False
                    
                    return True
                else:
                    print("Invalid model selection.")
                    return False
            except ValueError:
                print("Invalid input.")
                return False
                
        elif choice == "2":
            self.model_type = "ollama"
            ollama_models = self.get_ollama_models()
            
            if not ollama_models:
                print("No Ollama models found.")
                return False
                
            print("\nAvailable Ollama models:")
            for i, model in enumerate(ollama_models, 1):
                print(f"{i}. {model}")
            
            model_choice = input(f"Select model (1-{len(ollama_models)}): ").strip()
            try:
                model_idx = int(model_choice) - 1
                if 0 <= model_idx < len(ollama_models):
                    self.selected_model = ollama_models[model_idx]
                    print(f"Selected: {self.selected_model}")
                    return True
                else:
                    print("Invalid model selection.")
                    return False
            except ValueError:
                print("Invalid input.")
                return False
                
        else:
            print("Invalid choice.")
            return False
    
    def is_file_already_processed(self, title: str) -> bool:
        """Check if file has already been processed by checking title hash in training data"""
        try:
            # Generate hash from title
            title_hash = self.get_file_title_hash(title)
            
            # Check if this hash exists in training data store
            existing_data = training_data_store.get_training_data_by_uid(title_hash)
            
            if existing_data:
                print(f"  📋 File already processed (found hash: {title_hash[:8]}...)")
                return True
            else:
                return False
                
        except Exception as e:
            print(f"  ⚠️ Error checking if file was processed: {e}")
            return False

    def process_text_file(self, text_file: Tuple) -> bool:
        """Process a single text file and generate training data"""
        file_path, title, content = text_file
        
        print(f"\n📄 Processing file: {title}")
        print(f"   📏 Content length: {len(content)} characters")
        
        # Check if already processed using title hash
        if self.is_file_already_processed(title):
            print(f"⏭️ File '{title}' already processed. Skipping.")
            return True
        
        # Generate Q&A pairs
        print("  🤖 Generating Q&A pairs...")
        response = self.generate_qa_pairs(content)
        if not response:
            print(f"  ❌ Failed to generate Q&A pairs for file: {title}")
            return False
        
        # Parse Q&A pairs
        qa_pairs = self.parse_qa_response(response)
        
        if not qa_pairs:
            print(f"  ❌ No valid Q&A pairs found in response for file: {title}")
            print(f"  📋 Raw response preview: {response[:200]}...")
            return False
        
        print(f"  ✅ Found {len(qa_pairs)} Q&A pairs")
        
        # Generate title hash to use as identifier
        title_hash = self.get_file_title_hash(title)
        print(f"  🔑 File hash: {title_hash[:8]}...")
        
        # Store each Q&A pair in training data store with title hash as UID
        success_count = 0
        for i, (question, answer) in enumerate(qa_pairs):
            # Create unique UID for each Q&A pair by combining title hash with index
            qa_uid = f"{title_hash}_{i:03d}"
            
            success = training_data_store.create_training_data(
                question=question,
                answer=answer,
                uid=qa_uid,
                generated_model=self.selected_model
            )
            
            if success:
                success_count += 1
                print(f"    ✅ Stored Q&A pair {success_count}")
            else:
                print(f"    ⚠️ Failed to store Q&A pair (question may already exist)")
        
        # Store a marker entry with just the title hash to indicate file was processed
        marker_success = training_data_store.create_training_data(
            question=f"__FILE_MARKER__:{title}",
            answer=f"File processed successfully. Generated {success_count} Q&A pairs.",
            uid=title_hash,
            generated_model=self.selected_model
        )
        
        if marker_success:
            print(f"  📌 Stored file processing marker")
        
        print(f"  📊 Successfully stored {success_count}/{len(qa_pairs)} Q&A pairs")
        
        return success_count > 0
    
    def run(self):
        """Main execution method"""
        print("=== Training Data Generator ===")
        print(f"📁 Reading text files from: {self.text_files_folder}/")
        
        # Model selection
        if not self.select_model():
            print("Model selection failed. Exiting.")
            return
        
        # Get text files to process
        print(f"\n🤖 Selected model: {self.selected_model} ({self.model_type})")
        
        # Get all text files
        all_text_files = self.get_text_files()
        
        if not all_text_files:
            print(f"❌ No text files found in the '{self.text_files_folder}' folder.")
            print(f"📝 Please make sure the folder exists and contains .txt files.")
            return
        
        print(f"📊 Total text files to process: {len(all_text_files)}")
        
        # Show file list (first 10 files only to avoid spam)
        print("\n📋 Found files (showing first 10):")
        for i, (file_path, title, content) in enumerate(all_text_files[:10], 1):
            content_preview = content[:100] + "..." if len(content) > 100 else content
            print(f"  {i}. {title} ({len(content)} chars)")
        
        if len(all_text_files) > 10:
            print(f"  ... and {len(all_text_files) - 10} more files")
        
        # Ask user if they want to process all files
        confirm = input(f"\n❓ Process all {len(all_text_files)} text files? (y/n): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("🚫 Processing cancelled.")
            return
        
        # Ask about continuation on errors
        print("\n⚙️ Processing Options:")
        print("1. Stop on first error")
        print("2. Continue processing on errors (recommended)")
        error_choice = input("Select option (1 or 2): ").strip()
        continue_on_error = error_choice == "2"
        
        # Process text files
        processed_count = 0
        skipped_count = 0
        failed_count = 0
        
        print(f"\n🚀 Starting processing...")
        if continue_on_error:
            print("   ✅ Will continue processing even if some files fail")
        else:
            print("   ⚠️ Will stop on first error")
        
        for i, text_file in enumerate(all_text_files, 1):
            print(f"\n{'='*60}")
            print(f"📄 File {i}/{len(all_text_files)}")
            
            try:
                if self.process_text_file(text_file):
                    processed_count += 1
                else:
                    failed_count += 1
                    if not continue_on_error:
                        print(f"\n❌ Stopping due to error processing file {i}")
                        break
            except KeyboardInterrupt:
                print(f"\n\n⚠️ Processing interrupted by user at file {i}/{len(all_text_files)}")
                print(f"📊 Processed so far: {processed_count} successful, {failed_count} failed")
                user_choice = input("Continue processing? (y/n): ").strip().lower()
                if user_choice not in ['y', 'yes']:
                    break
            except Exception as e:
                print(f"❌ Unexpected error processing file: {e}")
                failed_count += 1
                if not continue_on_error:
                    print(f"\n❌ Stopping due to unexpected error at file {i}")
                    break
            
            # Show progress every 5 files
            if i % 5 == 0:
                print(f"\n📈 Progress Update:")
                print(f"   📊 Checked: {i}/{len(all_text_files)} files")
                print(f"   ✅ Processed: {processed_count}")
                print(f"   ⏭️  Skipped: {skipped_count}")
                print(f"   ❌ Failed: {failed_count}")
        
        print(f"\n{'='*60}")
        print(f"🎯 Final Summary")
        print(f"{'='*60}")
        print(f"📊 Total files checked: {len(all_text_files)}")
        print(f"✅ Successfully processed: {processed_count}")
        print(f"⏭️ Skipped (already processed): {skipped_count}")
        print(f"❌ Failed to process: {failed_count}")
        
        if (len(all_text_files)-skipped_count) > 0:
            success_rate = (processed_count/(len(all_text_files)-skipped_count)*100)
            print(f"📈 Success rate: {success_rate:.1f}%")
        
        print("\n🎉 Training data generation complete!")
        
        # Show database stats
        total_qa_pairs = training_data_store.get_training_data_count()
        print(f"💾 Total Q&A pairs in database: {total_qa_pairs}")

    def generate_qa_pairs(self, content: str) -> Optional[str]:
        """Generate Q&A pairs from content using the selected model"""
        if self.model_type == "gemini":
            return self._generate_qa_pairs_gemini(content)
        elif self.model_type == "ollama":
            return self._generate_qa_pairs_ollama(content)
        else:
            print(f"❌ Unknown model type: {self.model_type}")
            return None

    def _generate_qa_pairs_gemini(self, content: str) -> Optional[str]:
        """Generate Q&A pairs using Gemini API"""
        prompt = """Generate question-answer pairs from the following text content. Create comprehensive questions that test understanding of the key concepts, methods, and findings.

Format your response as follows:
Q1: [Question 1]
A1: [Answer 1]

Q2: [Question 2]
A2: [Answer 2]

And so on...

Guidelines:
- Create 8-12 question-answer pairs
- Focus on key concepts, methods, results, and implications
- Make questions clear and specific
- Provide detailed, accurate answers
- Cover different aspects of the content
- Include both factual and conceptual questions"""

        return self.call_gemini_api(prompt, content)

    def _generate_qa_pairs_ollama(self, content: str) -> Optional[str]:
        """Generate Q&A pairs using Ollama"""
        prompt = """Generate question-answer pairs from the following text content. Create comprehensive questions that test understanding of the key concepts, methods, and findings.

Format your response as follows:
Q1: [Question 1]
A1: [Answer 1]

Q2: [Question 2]
A2: [Answer 2]

And so on...

Guidelines:
- Create 8-12 question-answer pairs
- Focus on key concepts, methods, results, and implications
- Make questions clear and specific
- Provide detailed, accurate answers
- Cover different aspects of the content
- Include both factual and conceptual questions"""

        return self.call_ollama_api(prompt, content)

    def call_gemini_api(self, prompt: str, content: str) -> Optional[str]:
        """Call Gemini API with single attempt and 15k char limit"""
        
        # Truncate content if too long (15k character limit)
        max_content_length = 15000
        if len(content) > max_content_length:
            print(f"  📝 Content too long ({len(content)} chars), truncating to {max_content_length} chars")
            content = content[:max_content_length] + "\n... (content truncated)"
        
        try:
            print(f"  🔄 Making API call...")
            
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.selected_model}:generateContent"
            
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.gemini_api_key
            }
            
            full_prompt = f"{prompt}\n\nText Content:\n{content}"
            
            data = {
                "contents": [{
                    "parts": [{
                        "text": full_prompt
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
                    print(f"  📋 Response structure: {json.dumps(result, indent=2)[:200]}...")
            else:
                print(f"  ⚠️ No candidates in response")
                print(f"  📋 Response: {json.dumps(result, indent=2)[:200]}...")
                
        except requests.exceptions.HTTPError as e:
            error_details = ""
            try:
                error_json = e.response.json()
                error_details = f" - {error_json.get('error', {}).get('message', 'No details')}"
            except:
                error_details = f" - {e.response.text[:100]}"
            
            if e.response.status_code == 503:
                print(f"  🚫 Service unavailable (503){error_details}")
            elif e.response.status_code == 429:
                print(f"  🚫 Rate limited (429){error_details}")
            elif e.response.status_code == 400:
                print(f"  🚫 Bad request (400){error_details}")
                if "SAFETY" in str(error_details).upper() or "BLOCKED" in str(error_details).upper():
                    print(f"  ⚠️ Content blocked by safety filters")
            else:
                print(f"  🚫 HTTP error {e.response.status_code}{error_details}")
            
        except requests.exceptions.Timeout:
            print(f"  ⏰ Request timed out")
            
        except requests.exceptions.RequestException as e:
            print(f"  🚫 Request error: {e}")
            
        except KeyError as e:
            print(f"  🚫 Unexpected response format: {e}")
        
        except Exception as e:
            print(f"  🚫 Unexpected error: {e}")

        return None

    def call_ollama_api(self, prompt: str, content: str) -> Optional[str]:
        """Call Ollama API with content truncation"""
        
        # Truncate content if too long
        max_content_length = 15000
        if len(content) > max_content_length:
            print(f"  📝 Content too long ({len(content)} chars), truncating to {max_content_length} chars")
            content = content[:max_content_length] + "\n... (content truncated)"
        
        try:
            print(f"  🔄 Making Ollama API call...")
            
            full_prompt = f"{prompt}\n\nText Content:\n{content}"
            
            data = {
                "model": self.selected_model,
                "prompt": full_prompt,
                "stream": False
            }
            
            response = requests.post("http://localhost:11434/api/generate", json=data, timeout=300)
            
            print(f"  📊 Response status: {response.status_code}")
            
            response.raise_for_status()
            
            result = response.json()
            if "response" in result:
                print(f"  ✅ Ollama API call successful")
                return result["response"]
            else:
                print(f"  ⚠️ No response in Ollama result")
                print(f"  📋 Response: {json.dumps(result, indent=2)[:200]}...")
                
        except requests.exceptions.RequestException as e:
            print(f"  🚫 Ollama API error: {e}")
        except Exception as e:
            print(f"  🚫 Unexpected error: {e}")

        return None

    def parse_qa_response(self, response: str) -> List[Tuple[str, str]]:
        """Parse Q&A pairs from AI response"""
        if not response:
            return []
        
        qa_pairs = []
        
        # Try to find Q/A pattern
        # Look for patterns like "Q1:", "Q2:", etc. followed by "A1:", "A2:", etc.
        lines = response.split('\n')
        current_question = None
        current_answer = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line starts with Q followed by number and colon
            q_match = re.match(r'^Q\d+\s*:\s*(.+)', line, re.IGNORECASE)
            if q_match:
                # Save previous Q&A pair if exists
                if current_question and current_answer:
                    qa_pairs.append((current_question.strip(), current_answer.strip()))
                
                current_question = q_match.group(1)
                current_answer = None
                continue
            
            # Check if line starts with A followed by number and colon
            a_match = re.match(r'^A\d+\s*:\s*(.+)', line, re.IGNORECASE)
            if a_match:
                current_answer = a_match.group(1)
                continue
            
            # If we have a current question but no answer yet, this might be continuation of question
            if current_question and not current_answer:
                current_question += " " + line
            # If we have both question and answer, this might be continuation of answer
            elif current_question and current_answer:
                current_answer += " " + line
        
        # Don't forget the last pair
        if current_question and current_answer:
            qa_pairs.append((current_question.strip(), current_answer.strip()))
        
        # If the above didn't work, try a simpler approach
        if not qa_pairs:
            # Look for any question marks followed by text
            potential_questions = re.findall(r'([^.!?]*\?[^Q]*?)(?=Q\d+|$)', response, re.DOTALL | re.IGNORECASE)
            for i, q_text in enumerate(potential_questions):
                lines = q_text.strip().split('\n')
                question = ""
                answer = ""
                found_answer = False
                
                for line in lines:
                    line = line.strip()
                    if '?' in line and not found_answer:
                        question += line + " "
                    elif line and found_answer:
                        answer += line + " "
                    elif line and question:
                        found_answer = True
                        answer += line + " "
                
                if question.strip() and answer.strip():
                    qa_pairs.append((question.strip(), answer.strip()))
        
        # Clean up the pairs
        cleaned_pairs = []
        for question, answer in qa_pairs:
            # Remove Q1:, A1: prefixes if they exist
            question = re.sub(r'^Q\d+\s*:\s*', '', question, flags=re.IGNORECASE).strip()
            answer = re.sub(r'^A\d+\s*:\s*', '', answer, flags=re.IGNORECASE).strip()
            
            # Only keep pairs with substantial content
            if len(question) > 10 and len(answer) > 10:
                cleaned_pairs.append((question, answer))
        
        return cleaned_pairs

def main():
    generator = DataGenerator()
    generator.run()

if __name__ == "__main__":
    main()