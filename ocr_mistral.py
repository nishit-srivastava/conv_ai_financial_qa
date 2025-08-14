
import requests, os

API_KEY = os.getenv("MISTRAL_API_KEY")
PDF_FILE = "data/raw_pdfs/financials_2023.pdf"

url = "https://api.mistral.ai/v1/ocr"  # Example endpoint, check actual API docs
headers = {"Authorization": f"Bearer {API_KEY}"}
files = {'file': open(PDF_FILE, 'rb')}
data = {"output_format": "markdown", "language": "en"}

response = requests.post(url, headers=headers, files=files, data=data)

if response.status_code == 200:
    ocr_result = response.json().get("text", "")
    with open("data/ocr_output/financials_2023.md", "w", encoding="utf-8") as f:
        f.write(ocr_result)
    print("OCR complete. Output saved.")
else:
    print(f"Error: {response.status_code}, {response.text}")
