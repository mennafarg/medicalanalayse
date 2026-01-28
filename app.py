import os
import time
import uuid
import json
from flask import Flask, request, jsonify
import google.generativeai as genai

app = Flask(__name__)


genai.configure(api_key=os.environ["GEMINI_API_KEY"])
MODEL_NAME = "gemini-2.5-flash"


SESSIONS = {}


def upload_to_gemini(file_storage):
    tmp_path = f"/tmp/{uuid.uuid4()}.pdf"
    file_storage.save(tmp_path)

    f = genai.upload_file(tmp_path)
    while f.state.name == "PROCESSING":
        time.sleep(2)
        f = genai.get_file(f.name)

    os.remove(tmp_path)
    return f


@app.route("/analyze", methods=["POST"])
def analyze():

    medical = request.files.get("medical")
    lab = request.files.get("lab")
    radiology = request.files.get("radiology")

    if not medical or not lab or not radiology:
        return jsonify({"error": "All files are required"}), 400

    medical_f = upload_to_gemini(medical)
    lab_f = upload_to_gemini(lab)
    radiology_f = upload_to_gemini(radiology)

    analysis_prompt = """
You are a senior clinical decision-support AI.

Return ONLY valid JSON in this format:
{
  "high_priority_alerts": [],
  "low_priority_alerts": [],
  "current_diagnoses": [],
  "critical_allergies": [],
  "ai_insight": [],
  "ai_summary": [],
  "likely_diagnoses": {
    "high_likelihood": [],
    "possible": [],
    "low_likelihood": []
  }
}
"""

    model = genai.GenerativeModel(
        MODEL_NAME,
        generation_config={
            "temperature": 0.1,
            "response_mime_type": "application/json"
        }
    )

    response = model.generate_content(
        [analysis_prompt, medical_f, lab_f, radiology_f]
    )

    analysis_json = json.loads(response.text)

    # نعمل session للمريض
    session_id = str(uuid.uuid4())

    chat_model = genai.GenerativeModel(
        MODEL_NAME,
        generation_config={"temperature": 0}
    )

    chat = chat_model.start_chat(
        history=[{
            "role": "user",
            "parts": [medical_f, lab_f, radiology_f]
        }]
    )

    SESSIONS[session_id] = chat

    return jsonify({
        "session_id": session_id,
        "analysis": analysis_json
    })

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    session_id = data.get("session_id")
    question = data.get("question")

    if not session_id or not question:
        return jsonify({"error": "session_id and question are required"}), 400

    chat = SESSIONS.get(session_id)
    if not chat:
        return jsonify({"error": "Invalid session_id"}), 400

    response = chat.send_message(question)

    return jsonify({
        "answer": response.text
    })


@app.route("/", methods=["GET"])
def health():
    return "API is running ✅"
