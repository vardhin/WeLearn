import fitz  # PyMuPDF
import os

def extract_text_from_pdfs(pdf_folder, text_folder):
    if not os.path.exists(text_folder):
        os.makedirs(text_folder)

    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            text_path = os.path.join(text_folder, pdf_file.replace('.pdf', '.txt'))

            try:
                doc = fitz.open(pdf_path)
                full_text = ""
                for page in doc:
                    full_text += page.get_text()
                doc.close()

                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(full_text)
                print(f"Successfully extracted: {pdf_file}")
            except Exception as e:
                print(f"Could not process {pdf_file}: {e}")

# --- RUN THE EXTRACTION ---
extract_text_from_pdfs("./field_expert", "./field_expert_text")
extract_text_from_pdfs("./collaboration_expert", "./collaboration_expert_text")