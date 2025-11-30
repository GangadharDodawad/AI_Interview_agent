# AI_Interview_agent

# 🎙️ Helix — AI-Powered Mock Interviewer

Helix is an intelligent mock interview system built using **Streamlit** and **Google Gemini AI**.  
It simulates a real hiring manager — reading your resume, analyzing the job description, asking dynamic technical & behavioral questions, and generating a final evaluation report.

---

## 🚀 Key Features

- 📄 Reads & understands resume (PDF) and job description
- 🤖 Adaptive questioning based on user responses
- 🧠 Evaluates both **technical and communication** skills
- 👨‍💻 Includes a live **Python coding challenge**
- 📊 Generates a downloadable **interview report (PDF)**
- 🎤 Optional voice input + text-to-speech response

---

## 🛠️ Tech Stack

| Component | Tech Used |
|----------|-----------|
| UI | Streamlit |
| AI Model | Gemini (via `langchain-google-genai`) |
| Workflow | LangGraph + LangChain |
| File Handling | PyPDF2, FPDF |
| Audio | edge-tts, streamlit-mic-recorder |

---
