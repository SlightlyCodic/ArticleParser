import streamlit as st
import requests
import fitz  # PyMuPDF
import pandas as pd
import io
import PyPDF2
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import openai
import gspread
from google.oauth2.service_account import Credentials
import json

# Write the service account file
service_account_info = dict(st.secrets["gcp_service_account"])
try:
    with open("service_account.json", "w") as f:
        json.dump(service_account_info, f)
except Exception as e:
    st.error(f"Error creating service account file: {e}")

def extract_text_with_ocr(uploaded_file):
    # Convert PDF pages to images
    images = convert_from_bytes(uploaded_file.read())
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img)
    return text

def extract_text_pypdf2(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def append_to_gsheet(article_name, challenges, interventions, url_link, sheet_name="Sheet1"):
    # Define the scope
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_file("service_account.json", scopes=scope)
    client = gspread.authorize(creds)

    # Open the Google Sheet (replace with your sheet name or URL)
    sheet = client.open("Challenges and Limitation").worksheet(sheet_name)
    # Append the row
    sheet.append_row([article_name, challenges, interventions, url_link])

# --- CONFIG ---
HF_API_KEY = st.secrets["HF_API_KEY"]
MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # or your preferred instruct model
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Article Summarizer", layout="centered")
st.title("📄 PDF Article Summarizer")
st.caption("Extract Challenges & Interventions using Hugging Face LLM or OpenAI ChatGPT")

# --- Upload PDF ---
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# --- Model Selection ---
model_choice = st.radio(
    "Choose LLM backend:",
    ("Hugging Face (Mixtral)", "OpenAI ChatGPT")
)

if uploaded_file is not None:
    article_name = st.text_input("Article Name", value=uploaded_file.name)

    if st.button("Extract Information"):
        # Try normal extraction first
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""

        # If no text, try OCR
        if len(text.strip()) == 0:
            st.warning("No text extracted with PyPDF2. Trying OCR (may take longer)...")
            uploaded_file.seek(0)  # Reset file pointer
            text = extract_text_with_ocr(uploaded_file)

        if len(text.strip()) == 0:
            st.error("No text extracted. This PDF may be encrypted or not readable.")
            st.stop()

        # --- Step 2: Create prompt for LLM ---
        hf_prompt = f"""Read the following academic article and extract the main challenges and interventions as bullet points.

Challenges:
- (list each challenge as a bullet point)

Interventions:
- (list each intervention as a bullet point)

Article:
{text[:1000]}
"""

        gpt_prompt_template = """Read the following academic article and extract:

1. List the main challenges discussed in the article as bullet points.
2. List the interventions or solutions proposed as bullet points.

Return the result in this format:

Challenges:\n- ...\n- ...\nInterventions:\n- ...\n- ...

Article:
{text}
"""

        # --- Step 3: Send to selected LLM API ---
        with st.spinner(f"Processing with {model_choice}..."):
            result_text = None
            if model_choice == "Hugging Face (Mixtral)":
                prompt = hf_prompt
                response = requests.post(
                    f"https://api-inference.huggingface.co/models/{MODEL_NAME}",
                    headers={
                        "Authorization": f"Bearer {HF_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={"inputs": prompt}
                )
                if response.status_code != 200:
                    st.error(f"Model Error: {response.text}")
                    st.stop()
                output = response.json()
                result_text = output[0].get("generated_text", "No output")
            elif model_choice == "OpenAI ChatGPT":
                openai.api_key = OPENAI_API_KEY
                # Use GPT-4-turbo and split text into chunks if needed
                max_tokens_per_chunk = 12000  # ~16k context, leave room for prompt/response
                import tiktoken
                enc = tiktoken.encoding_for_model("gpt-4-turbo")
                text_tokens = enc.encode(text)
                chunk_size = max_tokens_per_chunk
                chunks = [text_tokens[i:i+chunk_size] for i in range(0, len(text_tokens), chunk_size)]
                results = []
                for idx, chunk in enumerate(chunks):
                    chunk_text = enc.decode(chunk)
                    prompt = gpt_prompt_template.replace("{text}", chunk_text)
                    try:
                        completion = openai.ChatCompletion.create(
                            model="gpt-4-turbo",
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant for extracting information from academic articles."},
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=2048,
                            temperature=0.2
                        )
                        results.append(completion.choices[0].message.content)
                    except Exception as e:
                        st.error(f"OpenAI API Error (chunk {idx+1}): {e}")
                        st.stop()
                result_text = "\n".join(results)

        # --- Step 4: Parse Challenges & Interventions ---
        import re
        challenges = re.search(r"Challenges:\s*((?:- .*(?:\n|$))+)", result_text)
        interventions = re.search(r"Intervention[s]?:\s*((?:- .*(?:\n|$))+)", result_text)

        challenges_text = challenges.group(1).strip() if challenges else "Not found"
        interventions_text = interventions.group(1).strip() if interventions else "Not found"

        st.subheader("✅ Extracted Information")
        st.markdown(f"**Challenges**: {challenges_text}")
        st.markdown(f"**Interventions**: {interventions_text}")

        # Append to Google Sheets
        try:
            append_to_gsheet(article_name, challenges_text, interventions_text, "N/A (uploaded file)")
            st.success("Data appended to Google Sheet!")
        except Exception as e:
            st.warning(f"Could not append to Google Sheet: {e}")
