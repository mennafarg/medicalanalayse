from flask import Flask, request, jsonify
import requests
import fitz
import json
from groq import Groq
import os

app = Flask(__name__)

client = Groq(api_key="grok_api")


# =========================
# استخراج النص من PDF
# =========================
def extract_text_from_pdf(url):
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception("Failed to download PDF")

    pdf_bytes = response.content
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    text = ""
    for page in doc:
        text += page.get_text()

    doc.close()
    return text


# =========================
# تحليل النص طبياً
# =========================
def analyze_medical_case(medical_text):

    prompt = f"""
Analyze the medical case and return JSON.

Case:
{medical_text}
"""

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        response_format={"type": "json_object"}
    )

    return json.loads(chat_completion.choices[0].message.content)


# =========================
# API Endpoint
# =========================
@app.route("/analyze", methods=["POST"])
def analyze():

    data = request.json
    file_url = data.get("file_url")

    if not file_url:
        return jsonify({"error": "file_url is required"}), 400

    try:
        text = extract_text_from_pdf(file_url)
        result = analyze_medical_case(text)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# تشغيل السيرفر
if __name__ == "__main__":
    app.run(debug=True, port=5000)
