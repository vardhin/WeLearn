import pandas as pd
import os
from typing import Optional

class ScoreAnalyzer:
    def __init__(self):
        self.df = None
        self.score_column = 'likert_score'  # 4th column name
    
    def load_excel_file(self, file_path: str) -> bool:
        """Load Excel file and validate it has the score column"""
        try:
            self.df = pd.read_excel(file_path)
            
            print(f"✅ Excel file loaded successfully")
            print(f"📊 Total rows: {len(self.df)}")
            print(f"📋 Columns: {list(self.df.columns)}")
            
            # Check if score column exists
            if self.score_column not in self.df.columns:
                print(f"❌ Score column '{self.score_column}' not found")
                print(f"📋 Available columns: {list(self.df.columns)}")
                
                # Try to find score column by position (4th column)
                if len(self.df.columns) >= 4:
                    self.score_column = self.df.columns[3]  # 4th column (0-indexed)
                    print(f"🔄 Using 4th column '{self.score_column}' as score column")
                else:
                    print(f"❌ File doesn't have enough columns (need at least 4)")
                    return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading Excel file: {e}")
            return False
    
    def analyze_scores(self) -> bool:
        """Calculate and display score statistics"""
        if self.df is None:
            print("❌ No data loaded. Please load an Excel file first.")
            return False
        
        try:
            # Get the score column
            scores = self.df[self.score_column]
            
            # Remove any NaN/None values and convert to numeric
            valid_scores = pd.to_numeric(scores, errors='coerce').dropna()
            
            if len(valid_scores) == 0:
                print(f"❌ No valid numeric scores found in column '{self.score_column}'")
                return False
            
            # Calculate statistics
            total_rows = len(self.df)
            valid_rows = len(valid_scores)
            missing_rows = total_rows - valid_rows
            
            score_sum = valid_scores.sum()
            score_average = valid_scores.mean()
            score_min = valid_scores.min()
            score_max = valid_scores.max()
            score_std = valid_scores.std()
            
            # Display results
            print(f"\n{'='*60}")
            print(f"📊 SCORE ANALYSIS RESULTS")
            print(f"{'='*60}")
            print(f"📁 File: {self.score_column}")
            print(f"📈 Total rows in file: {total_rows}")
            print(f"✅ Rows with valid scores: {valid_rows}")
            print(f"❌ Rows with missing scores: {missing_rows}")
            
            if missing_rows > 0:
                completion_rate = (valid_rows / total_rows) * 100
                print(f"📊 Completion rate: {completion_rate:.1f}%")
            
            print(f"\n🎯 SCORE STATISTICS:")
            print(f"➕ Sum of all scores: {score_sum:.2f}")
            print(f"📊 Average score: {score_average:.2f}")
            print(f"⬇️  Minimum score: {score_min:.2f}")
            print(f"⬆️  Maximum score: {score_max:.2f}")
            print(f"📏 Standard deviation: {score_std:.2f}")
            
            # Score distribution
            print(f"\n📈 SCORE DISTRIBUTION:")
            score_counts = valid_scores.value_counts().sort_index()
            for score, count in score_counts.items():
                percentage = (count / valid_rows) * 100
                print(f"   Score {int(score)}: {count} times ({percentage:.1f}%)")
            
            # Quality assessment
            print(f"\n🎭 QUALITY ASSESSMENT:")
            if score_average >= 4.0:
                quality = "Excellent"
                emoji = "🌟"
            elif score_average >= 3.5:
                quality = "Good"
                emoji = "👍"
            elif score_average >= 3.0:
                quality = "Average"
                emoji = "😐"
            elif score_average >= 2.0:
                quality = "Below Average"
                emoji = "👎"
            else:
                quality = "Poor"
                emoji = "💔"
            
            print(f"{emoji} Overall Quality: {quality} (Average: {score_average:.2f}/5.0)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error analyzing scores: {e}")
            return False
    
    def save_summary_report(self, output_path: str = None) -> bool:
        """Save a summary report to a text file"""
        if self.df is None:
            print("❌ No data loaded.")
            return False
        
        try:
            scores = pd.to_numeric(self.df[self.score_column], errors='coerce').dropna()
            
            if len(scores) == 0:
                print("❌ No valid scores to create report.")
                return False
            
            # Generate report content
            report_content = f"""LIKERT SCORE ANALYSIS REPORT
{'='*50}

File Analysis:
- Total rows: {len(self.df)}
- Valid scores: {len(scores)}
- Missing scores: {len(self.df) - len(scores)}
- Completion rate: {(len(scores) / len(self.df)) * 100:.1f}%

Score Statistics:
- Sum of all scores: {scores.sum():.2f}
- Average score: {scores.mean():.2f}
- Minimum score: {scores.min():.2f}
- Maximum score: {scores.max():.2f}
- Standard deviation: {scores.std():.2f}

Score Distribution:
"""
            
            score_counts = scores.value_counts().sort_index()
            for score, count in score_counts.items():
                percentage = (count / len(scores)) * 100
                report_content += f"- Score {int(score)}: {count} times ({percentage:.1f}%)\n"
            
            # Save report
            if output_path is None:
                output_path = "score_analysis_report.txt"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            print(f"\n📄 Summary report saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving report: {e}")
            return False

def main():
    analyzer = ScoreAnalyzer()
    
    # Default file path (you can change this)
    default_file = "./base_likert.xlsx"
    
    # Get Excel file path from user
    print("=== Score Analyzer ===")
    excel_path = input(f"Enter Excel file path (or press Enter for '{default_file}'): ").strip()
    
    if not excel_path:
        excel_path = default_file
    
    if not os.path.exists(excel_path):
        print(f"❌ File not found: {excel_path}")
        return
    
    # Load and analyze the file
    if analyzer.load_excel_file(excel_path):
        if analyzer.analyze_scores():
            # Ask if user wants to save a report
            save_report = input("\n💾 Save summary report to file? (y/n): ").strip().lower()
            if save_report in ['y', 'yes']:
                report_path = input("Enter report file path (or press Enter for default): ").strip()
                if not report_path:
                    report_path = None
                analyzer.save_summary_report(report_path)
        else:
            print("❌ Failed to analyze scores.")
    else:
        print("❌ Failed to load Excel file.")

if __name__ == "__main__":
    main()