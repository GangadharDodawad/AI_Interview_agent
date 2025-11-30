import os
import re
import json
import asyncio
import streamlit as st
import base64
import edge_tts
from io import BytesIO
from fpdf import FPDF

# Fix for Streamlit's asyncio loop issue
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

import PyPDF2
from typing import TypedDict, List, Annotated, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    BaseMessage,
)
import operator
from streamlit_mic_recorder import speech_to_text

# --- Page Config ---
st.set_page_config(page_title="Helix: AI Interviewer", layout="wide")

# --- Light global styling via CSS ---
st.markdown(
    """
<style>
.stApp {
    background: linear-gradient(135deg, #eef2ff 0%, #f9fafb 45%, #e0f2fe 100%);
    color: #0f172a;
}
section[data-testid="stSidebar"] {
    background: #f9fafb;
    border-right: 1px solid #e5e7eb;
}
section[data-testid="stSidebar"] * {
    color: #0f172a !important;
}
.stButton > button {
    border-radius: 999px;
    border: 1px solid #38bdf8;
    background: linear-gradient(90deg, #0ea5e9, #22c55e);
    color: white;
    font-weight: 600;
    padding: 0.4rem 1.2rem;
    cursor: pointer;
    transition: all 0.2s ease-in-out;
}
.stButton > button:hover {
    box-shadow: 0 0 12px rgba(56, 189, 248, 0.7);
    transform: translateY(-1px) scale(1.01);
}
.block-container {
    padding-top: 0.8rem;
}
[data-testid="stChatMessage"] {
    border-radius: 0.75rem;
    padding: 0.1rem 0.1rem;
    margin-bottom: 0.2rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# --- Helper: PDF Text Extraction ---
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""


# --- Helper: Text To Speech ---
async def text_to_speech(text):
    """Converts text to audio using Edge TTS (Male Voice)."""
    try:
        communicate = edge_tts.Communicate(text, "en-US-ChristopherNeural")
        mp3_fp = BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_fp.write(chunk["data"])

        mp3_fp.seek(0)
        b64 = base64.b64encode(mp3_fp.read()).decode()
        md = f"""
            <audio controls autoplay style="display:none">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True)
    except Exception:
        # Don't break the UI if TTS fails
        pass


# --- Helper: PDF Report Generation ---
def generate_pdf_report(state):
    """Generates a PDF report from the interview state."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt="Interview Report: Candidate Evaluation", ln=1, align="C")
    pdf.ln(10)

    evals = state.get("evaluations", [])
    tech_scores = [e["tech_score"] for e in evals]
    comm_scores = [e["comm_score"] for e in evals]
    avg_tech = sum(tech_scores) / len(tech_scores) if tech_scores else 0
    avg_comm = sum(comm_scores) / len(comm_scores) if comm_scores else 0

    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, txt=f"Final Technical Score: {avg_tech:.1f}/10", ln=1)
    pdf.cell(200, 10, txt=f"Final Communication Score: {avg_comm:.1f}/10", ln=1)

    rec = "No Hire"
    if avg_tech > 7:
        rec = "Strong Hire"
    elif avg_tech > 5:
        rec = "Hire"
    pdf.cell(200, 10, txt=f"Recommendation: {rec}", ln=1)
    pdf.ln(10)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, txt="Identified Weaknesses / Gaps:", ln=1)
    pdf.set_font("Arial", size=11)
    for area in state.get("weak_areas", []):
        pdf.cell(200, 8, txt=f"- {area}", ln=1)
    pdf.ln(10)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, txt="Interview Transcript & Notes:", ln=1)
    pdf.set_font("Arial", size=10)

    for i, e in enumerate(evals):
        q = e["question"].encode("latin-1", "replace").decode("latin-1")
        a = e["answer"].encode("latin-1", "replace").decode("latin-1")

        pdf.multi_cell(0, 8, txt=f"Q{i+1}: {q}")
        pdf.set_text_color(100, 100, 100)
        pdf.multi_cell(0, 8, txt=f"Candidate: {a}")
        pdf.set_text_color(0, 0, 0)
        pdf.cell(
            0,
            8,
            txt=f"Score: Tech {e['tech_score']}/10, Comm {e['comm_score']}/10",
            ln=1,
        )
        pdf.ln(5)

    return pdf.output(dest="S").encode("latin-1")


# ============================================================
# Candidate Name Extraction
# ============================================================

def extract_candidate_name(resume_text: str, api_key: str) -> str:
    """
    Try to extract the candidate's name from resume text.
    1) Use simple regex heuristics.
    2) If that fails, use the LLM.
    """
    if not resume_text or not resume_text.strip():
        return "Candidate"

    # Simple regex-based attempts
    patterns = [
        r"Name[:\- ]+([A-Za-z][A-Za-z .'-]{2,50})",  # Name: John Doe
        r"^([A-Za-z][A-Za-z .'-]{2,50})\s*\n",       # first line = name
    ]

    for pattern in patterns:
        match = re.search(pattern, resume_text, flags=re.MULTILINE)
        if match:
            name = match.group(1).strip()
            if 3 <= len(name) <= 60:
                return name

    # Fallback: LLM
    if not api_key:
        return "Candidate"

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.1,
            google_api_key=api_key,
        )

        prompt = f"""
        You are given the raw text of a resume.

        Task:
        - Extract ONLY the full name of the person who owns this resume.
        - Respond with ONLY the name, no extra words.

        Resume:
        {resume_text}
        """

        result = llm.invoke(prompt)
        name = result.content.strip()
        if 3 <= len(name) <= 60:
            return name
    except Exception as e:
        print("[extract_candidate_name] Error:", e)

    return "Candidate"


# --- State Definition ---
class InterviewState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    job_description: str
    resume_text: str
    weak_areas: List[str]
    topic: str
    interaction_count: int
    evaluations: List[dict]
    red_flags: List[str]
    interview_complete: bool
    mode: str
    candidate_name: str


# --- Context Builder ---
def context_builder(state: InterviewState):
    api_key = os.environ.get("GOOGLE_API_KEY")
    default_weak = ["Algorithms & Data Structures", "System Design", "Clean Code / Best Practices"]

    candidate_name = state.get("candidate_name") or extract_candidate_name(
        state.get("resume_text", ""), api_key or ""
    )

    if not api_key:
        return {
            "weak_areas": default_weak,
            "mode": "chat",
            "candidate_name": candidate_name,
            "messages": [
                SystemMessage(
                    content=(
                        "Resume Context Loaded, but GOOGLE_API_KEY is missing, "
                        "using default weak areas. Resume Content (truncated): "
                        f"{state['resume_text'][:1000]}..."
                    )
                )
            ],
        }

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            google_api_key=api_key,
        )

        prompt = f"""
        Role: Technical Recruiter.
        JD: {state['job_description']}
        Resume: {state['resume_text']}

        Task: Identify the top 3 critical technical gaps/weaknesses in the resume compared to the JD.
        Return ONLY a comma-separated list of these 3 areas.
        """
        response = llm.invoke(prompt)
        weak_areas = [area.strip() for area in response.content.split(",") if area.strip()] or default_weak

        return {
            "weak_areas": weak_areas,
            "mode": "chat",
            "candidate_name": candidate_name,
            "messages": [
                SystemMessage(
                    content=f"Resume Context Loaded for {candidate_name}. Resume Content: {state['resume_text'][:2000]}..."
                )
            ],
        }
    except Exception as e:
        print("[context_builder] Error:", e)
        return {
            "weak_areas": default_weak,
            "mode": "chat",
            "candidate_name": candidate_name,
            "messages": [
                SystemMessage(
                    content=f"Error analyzing resume, falling back to defaults. Error: {e}. Resume: {state['resume_text'][:1000]}..."
                )
            ],
        }


# --- Helper: extract cleaner question line ---
def extract_question_text(ai_msg: str) -> str:
    """
    Try to get the actual question sentence from the AI message.
    Prefer the last sentence ending with '?'.
    If none, fall back to the last non-empty sentence, else whole text.
    """
    if not ai_msg:
        return ""

    sentences = re.split(r'(?<=[?.!])\s+', ai_msg.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    for s in reversed(sentences):
        if s.endswith("?"):
            return s

    if sentences:
        return sentences[-1]

    return ai_msg.strip()


# --- Evaluator (JSON-based scoring) ---
def evaluator(state: InterviewState):
    messages = state["messages"]
    if not messages or isinstance(messages[-1], AIMessage):
        return {}

    last_user_msg = messages[-1].content
    last_ai_msg = messages[-2].content if len(messages) > 1 else "Introduction"
    current_mode = state.get("mode", "chat")

    # Don't evaluate simple introductions strictly
    if state.get("interaction_count", 0) == 0:
        return {"interaction_count": 1, "evaluations": []}

    question_text = extract_question_text(last_ai_msg)

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return {
            "evaluations": [
                {
                    "question": question_text,
                    "answer": last_user_msg,
                    "tech_score": 5,
                    "comm_score": 5,
                }
            ],
            "interaction_count": state.get("interaction_count", 0) + 1,
            "red_flags": [],
        }

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=api_key,
    )

    if current_mode == "code":
        mode_desc = "coding solution"
        extra = f"Problem statement:\n{last_ai_msg}\n\nCode submitted:\n{last_user_msg}\n"
    else:
        mode_desc = "verbal answer"
        extra = f"Question asked:\n{last_ai_msg}\n\nAnswer:\n{last_user_msg}\n"

    prompt = f"""
You are scoring a candidate's {mode_desc} in an interview.

{extra}

Score the candidate on:
- Technical correctness, depth, and clarity (tech score from 0 to 10).
- Communication clarity and structure (comm score from 0 to 10).
Also note any red flags if present (for example: copying, dishonesty, extremely weak basics). If there are none, use the string "none".

Now respond with ONLY a valid JSON object, no other text.
The JSON must be exactly in this format:

{{
  "tech": <integer between 0 and 10>,
  "comm": <integer between 0 and 10>,
  "red_flags": "<text or 'none'>"
}}
"""
    response = llm.invoke(prompt)

    tech_score, comm_score = 5, 5
    red_flag = None

    try:
        data = json.loads(response.content.strip())
        tech_score = int(data.get("tech", 5))
        comm_score = int(data.get("comm", 5))
        rf = str(data.get("red_flags", "")).strip()
        if rf and rf.lower() not in ("none", "no", "nil"):
            red_flag = rf
    except Exception as e:
        print("[evaluator] JSON parse error:", e, "RAW:", response.content)

    return {
        "evaluations": [
            {
                "question": question_text,
                "answer": last_user_msg,
                "tech_score": tech_score,
                "comm_score": comm_score,
            }
        ],
        "interaction_count": state.get("interaction_count", 0) + 1,
        "red_flags": [red_flag] if red_flag else [],
    }


# ============================================================
# Question Generator (STRICT, grounded, verifies answers)
# ============================================================

def question_generator(state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        messages: List[BaseMessage] = state.get("messages", [])
        interaction_count: int = state.get("interaction_count", 0)
        resume_text: str = state.get("resume_text", "Not provided")
        job_description: str = state.get("job_description", "Not provided")
        weak_areas_raw = state.get("weak_areas", [])
        current_mode: str = state.get("mode", "chat")

        if isinstance(weak_areas_raw, list):
            weak_areas_str = ", ".join(weak_areas_raw) if weak_areas_raw else "Not specified"
        else:
            weak_areas_str = str(weak_areas_raw)

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return {
                "messages": [
                    AIMessage(
                        content="Server error: GOOGLE_API_KEY is not set. Please enter your Gemini API key in the sidebar."
                    )
                ],
                "mode": "chat",
            }

        if not state.get("candidate_name"):
            state["candidate_name"] = extract_candidate_name(resume_text, api_key)
        candidate_name: str = state["candidate_name"]
        first_name = candidate_name.split()[0] if " " in candidate_name else candidate_name

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.2,
            google_api_key=api_key,
        )

        # Get last user + AI
        last_user_msg = None
        last_ai_msg = None
        for m in reversed(messages):
            if isinstance(m, HumanMessage) and last_user_msg is None:
                last_user_msg = m.content
            elif isinstance(m, AIMessage) and last_ai_msg is None:
                last_ai_msg = m.content
            if last_user_msg is not None and last_ai_msg is not None:
                break

        if last_user_msg is None:
            last_user_msg = ""
        if last_ai_msg is None:
            last_ai_msg = "Introduction / No specific prior question."

        def recent_history(n: int = 8) -> str:
            tail = messages[-n:]
            lines = []
            for msg in tail:
                role = "User" if isinstance(msg, HumanMessage) else "AI"
                lines.append(f"{role}: {msg.content}")
            return "\n".join(lines)

        history_text = recent_history(8)

        grounding_block = f"""
        VERY IMPORTANT RULES (DO NOT IGNORE):
        - Base your feedback and next question ONLY on:
          the Job Description, the resume text, the chat history, and the candidate's last answer.
        - If some detail is NOT present in these, explicitly say:
          "I don't have that detail from your resume or answers" instead of guessing.
        - Do NOT invent company names, project names, or tools/technologies not present in resume or chat history.
        - Ask at most ONE main question and optionally ONE short sub-question.
        - No random small talk; stay focused and professional.
        """

        # PHASE 0: INTRO
        if interaction_count == 0:
            prompt = f"""
            You are 'Helix', a senior Engineering Hiring Manager at a tech company.

            You are about to conduct a technical + behavioral interview with {candidate_name}.
            Use their first name "{first_name}" at least once.

            Job Description (summary):
            {job_description}

            {grounding_block}

            Task:
            - Introduce yourself briefly as the hiring manager.
            - In 2–3 short sentences, outline the structure:
              quick intro from the candidate, resume/project deep dive,
              technical fundamentals plus a small coding exercise, and behavioral questions with time for their questions.
            - Then ask {first_name} to introduce themselves, focusing on:
              their current role/education, the main technologies they are comfortable with, and the kind of roles they are targeting.
            - Do NOT ask a technical question yet.

            Keep it 4–6 sentences total.
            """
            response = llm.invoke(prompt)
            return {"messages": [response], "mode": "chat"}

        # PHASE 1: RESUME DEEP DIVE
        elif interaction_count == 1:
            prompt = f"""
            You are 'Helix', a senior Engineering Hiring Manager interviewing {candidate_name}.
            Use their first name "{first_name}" naturally.

            Your previous prompt:
            {last_ai_msg}

            {first_name}'s introduction:
            {last_user_msg}

            Resume text:
            {resume_text}

            Job description:
            {job_description}

            {grounding_block}

            Task:
            - In 1–2 sentences, acknowledge {first_name}'s intro and link it to something in their resume.
            - Choose ONE specific project, internship, or role from the resume that fits the job description.
            - Ask 1–2 focused questions about that exact item, such as:
              their responsibilities, tech stack and why, or measurable impact.
            - Refer to that project/role using wording from the resume so it feels grounded and personal.

            4–7 sentences total.
            """
            response = llm.invoke(prompt)
            return {"messages": [response], "mode": "chat"}

        # PHASE 2–3: TECHNICAL FUNDAMENTALS
        elif interaction_count in [2, 3]:
            word_count = len(last_user_msg.split()) if last_user_msg else 0

            prompt = f"""
            You are 'Helix', a senior Engineering Hiring Manager.

            Last question you asked:
            "{last_ai_msg}"

            {first_name}'s answer:
            "{last_user_msg}"

            Job description:
            {job_description}

            Focus / weak areas:
            {weak_areas_str}

            Recent chat history:
            {history_text}

            {grounding_block}

            Task:
            - First, give 2–3 sentences of feedback on their answer:
              clearly mention which parts are correct, which are partially correct, and what is missing or slightly off.
            - Then ask ONE technical follow-up that stays in the same topic area:
              for example, deeper explanation, complexity analysis, trade-offs, or a concrete example.
            - Prefer questions linked to the listed focus/weak areas if relevant.

            Keep it 4–7 sentences total.
            """

            if word_count < 8:
                prompt += """
                Since their answer was very short, explicitly ask them to elaborate with a concrete example or clearer explanation.
                """

            response = llm.invoke(prompt)
            return {"messages": [response], "mode": "chat"}

        # PHASE 4: CODING CHALLENGE
        elif interaction_count == 4:
            prompt = f"""
            You are 'Helix', a hiring manager interviewing {candidate_name}.
            Use their first name "{first_name}" once.

            Job description:
            {job_description}

            Focus / weak areas:
            {weak_areas_str}

            Recent chat history:
            {history_text}

            {grounding_block}

            Task:
            - Propose ONE small coding problem in Python solvable in about 15–20 minutes.
            - Focus on basic logic and simple data structures like arrays, strings, or hash maps,
              or a tiny OOP design, preferably connected to backend or data if the JD suggests that.
            - Explain the task and input/output format clearly in a few sentences.
            - Tell {first_name} to write only the code (no long explanation).

            Do NOT include the solution.
            """
            response = llm.invoke(prompt)
            return {"messages": [response], "mode": "code"}

        # PHASE 5: CODE DISCUSSION
        elif interaction_count == 5:
            prompt = f"""
            You are 'Helix', the hiring manager.

            Previous coding problem:
            {last_ai_msg}

            {first_name}'s code:
            {last_user_msg}

            {grounding_block}

            Task:
            - In 2–3 sentences, give direct feedback:
              say whether the logic is correct, mention any obvious bugs or missing edge cases,
              and describe the rough time and space complexity.
            - Then ask 1–2 follow-up questions about how they could improve performance,
              handle a specific edge case, or refactor for readability and testability.

            Keep it 4–7 sentences total.
            """
            response = llm.invoke(prompt)
            return {"messages": [response], "mode": "chat"}

        # PHASE 6–7: BEHAVIORAL
        elif interaction_count in [6, 7]:
            prompt = f"""
            You are 'Helix', a hiring manager in the behavioral / team fit phase
            with candidate {candidate_name}. Use their first name "{first_name}" once.

            Last prompt:
            {last_ai_msg}

            {first_name}'s last answer:
            {last_user_msg}

            Job description:
            {job_description}

            Recent chat history:
            {history_text}

            {grounding_block}

            Task:
            - In 1–2 sentences, briefly react to their answer
              (what was good, or what could use more detail like impact/results).
            - Then ask ONE behavioral question that follows naturally from their answer and the role,
              about handling failure or ambiguity, teamwork and communication, ownership, or learning new tech.
            - Encourage them to give a specific example with situation, what they had to do,
              what actions they took, and the result.

            3–6 sentences total.
            """
            response = llm.invoke(prompt)
            return {"messages": [response], "mode": "chat"}

        # FINAL PHASE
        else:
            word_count = len(last_user_msg.split()) if last_user_msg else 0

            prompt = f"""
            You are 'Helix', a senior Engineering Hiring Manager.

            You are in the final phase of the interview with {candidate_name}.
            Use their first name "{first_name}" once.

            Last question you asked:
            {last_ai_msg}

            {first_name}'s last answer:
            {last_user_msg}

            Job description:
            {job_description}

            Focus / weak areas:
            {weak_areas_str}

            Recent chat history:
            {history_text}

            {grounding_block}

            Task:
            - In 1–2 sentences, comment on their last answer and mention what they did well
              or what could be a bit clearer.
            - Then either ask one final higher-level technical/design question that fits the earlier topics,
              OR a reflective question about how {first_name} learns, handles feedback, or plans to grow.
            - Finally, explicitly invite them to ask any questions about the role, team, or company.

            4–7 sentences total.
            """

            if current_mode == "chat" and word_count < 6:
                prompt += """
            Since their last answer was very short, your new question should invite them
            to elaborate with clearer reasoning or a concrete example before you close.
            """

            response = llm.invoke(prompt)
            return {"messages": [response], "mode": "chat"}

    except Exception as e:
        print("[question_generator] Fatal error:", e)
        return {
            "messages": [
                AIMessage(
                    content=f"Oops, something went wrong while generating the next interview question: {e}"
                )
            ],
            "mode": "chat",
        }


# --- Termination & Report ---
def termination_check(state: InterviewState):
    if state.get("interaction_count", 0) >= 8:
        return {"interview_complete": True}
    return {"interview_complete": False}


def report_generator(state: InterviewState):
    return {"messages": [AIMessage(content="Interview complete. Generating report...")]}


# --- LangGraph Build ---
def build_graph():
    workflow = StateGraph(InterviewState)
    workflow.add_node("context_builder", context_builder)
    workflow.add_node("question_generator", question_generator)
    workflow.add_node("evaluator", evaluator)
    workflow.add_node("termination_check", termination_check)
    workflow.add_node("report_generator", report_generator)

    workflow.set_entry_point("context_builder")
    workflow.add_edge("context_builder", "question_generator")
    workflow.add_edge("evaluator", "termination_check")

    def should_continue(state):
        return "report_generator" if state.get("interview_complete") else "question_generator"

    workflow.add_conditional_edges(
        "termination_check", should_continue, ["report_generator", "question_generator"]
    )
    workflow.add_edge("report_generator", END)
    workflow.add_edge("question_generator", END)
    return workflow.compile()


# --- Handle Input (shared for chat & code) ---
def handle_input(user_content):
    new_msg = HumanMessage(content=user_content)
    st.session_state.messages.append(new_msg)
    full_state = st.session_state.graph_state
    full_state["messages"] = st.session_state.messages

    with st.spinner("Analyzing..."):
        eval_updates = evaluator(full_state)
        full_state.update(eval_updates)
        term_updates = termination_check(full_state)
        full_state.update(term_updates)

        if full_state["interview_complete"]:
            report_generator(full_state)
            st.session_state.messages.append(
                AIMessage(content="Interview Concluded. Please download your report below.")
            )
        else:
            q_updates = question_generator(full_state)
            full_state["messages"].extend(q_updates["messages"])
            full_state["mode"] = q_updates.get("mode", "chat")

    st.session_state.graph_state = full_state
    st.rerun()


# --- Streamlit UI ---
def main():
    # --- TOP HEADER ---
    st.markdown(
        """
        <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
            <h1 style="margin-bottom:0.2rem;">🎙️ Helix: AI Interviewer</h1>
            <p style="color:#444; font-size:0.95rem;">
                Simulated 1:1 interview with a senior engineering hiring manager.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- SIDEBAR CONFIG ---
    with st.sidebar:
        st.subheader("⚙️ Configuration")
        api_key = st.text_input("Gemini API Key", type="password")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key

        jd = st.text_area("Job Description", height=120, placeholder="Paste the JD here...")
        uploaded_file = st.file_uploader("Resume (PDF)", type="pdf")

        # Live summary if running
        if "graph_state" in st.session_state:
            gs = st.session_state.graph_state
            interaction = gs.get("interaction_count", 0)
            candidate_name = gs.get("candidate_name", "Candidate")
            weak_areas = gs.get("weak_areas", [])

            st.markdown("---")
            st.subheader("👤 Candidate")
            st.write(f"**Name:** {candidate_name}")
            st.write(f"**Turns so far:** {interaction}")

            total_turns = 8
            progress = min(interaction / total_turns, 1.0)
            st.progress(progress)

            if weak_areas:
                st.markdown("**Focus Areas:**")
                for w in weak_areas:
                    st.markdown(f"- 🔍 {w}")

        start_btn = st.button("🚀 Start Interview", use_container_width=True)

        if start_btn:
            if not api_key or not jd or not uploaded_file:
                st.warning("Please provide API key, Job Description, and a Resume PDF.")
            else:
                resume_text = extract_text_from_pdf(uploaded_file)

                if not resume_text.strip():
                    st.error(
                        "Could not extract text from PDF. Is it a scanned image? Please upload a text-based PDF."
                    )
                else:
                    st.session_state.messages = []
                    st.session_state.app = build_graph()
                    candidate_name = extract_candidate_name(resume_text, api_key)
                    initial_state: InterviewState = {
                        "messages": [],
                        "job_description": jd,
                        "resume_text": resume_text,
                        "weak_areas": [],
                        "topic": "",
                        "interaction_count": 0,
                        "evaluations": [],
                        "red_flags": [],
                        "interview_complete": False,
                        "mode": "chat",
                        "candidate_name": candidate_name,
                    }
                    events = st.session_state.app.invoke(initial_state)
                    st.session_state.graph_state = events
                    st.session_state.messages = events["messages"]
                    st.rerun()

    # --- MAIN AREA ---
    # Chat history
    if "messages" in st.session_state:
        for msg in st.session_state.messages:
            if isinstance(msg, AIMessage):
                with st.chat_message("assistant"):
                    st.write(msg.content)
            elif isinstance(msg, HumanMessage):
                with st.chat_message("user"):
                    if "def " in msg.content or "import " in msg.content:
                        st.code(msg.content, language="python")
                    else:
                        st.write(msg.content)

    # Input Area (while interview running)
    if "graph_state" in st.session_state and not st.session_state.graph_state.get(
        "interview_complete", False
    ):
        # Audio playback of last AI message
        last_msg = st.session_state.messages[-1]
        if isinstance(last_msg, AIMessage):
            msg_hash = hash(last_msg.content)
            if (
                "last_played_hash" not in st.session_state
                or st.session_state.last_played_hash != msg_hash
            ):
                asyncio.run(text_to_speech(last_msg.content))
                st.session_state.last_played_hash = msg_hash

        current_mode = st.session_state.graph_state.get("mode", "chat")

        # Phase label
        interaction = st.session_state.graph_state.get("interaction_count", 0)
        if interaction == 0:
            phase = "Introduction"
        elif interaction == 1:
            phase = "Resume Deep Dive"
        elif interaction in [2, 3]:
            phase = "Technical Fundamentals"
        elif interaction == 4:
            phase = "Coding Challenge"
        elif interaction == 5:
            phase = "Code Review & Optimization"
        elif interaction in [6, 7]:
            phase = "Behavioral & Team Fit"
        else:
            phase = "Final Questions"

        st.markdown(
            f"<p style='color:#555; font-size:0.9rem; margin-top:0.5rem;'>🧭 Current Phase: <b>{phase}</b></p>",
            unsafe_allow_html=True,
        )

        # Coding phase
        if current_mode == "code":
            st.subheader("🧪 Coding Challenge")
            st.info("Write your Python solution below and click Submit.")

            code_input = st.text_area(
                "Your Python solution",
                value="# Write your Python code here...",
                height=250,
                key="code_input",
            )

            submit_code = st.button("✅ Submit Solution", use_container_width=True)
            if submit_code:
                if len(code_input.strip()) < 10:
                    st.warning("Please write a more complete answer before submitting.")
                else:
                    handle_input(code_input)

        # Verbal / text phase
        else:
            st.markdown("---")
            st.markdown("### 💬 Your Response")
            col_voice, col_text = st.columns([1, 3])
            with col_voice:
                st.write("🎤 **Speak:**")
                voice_text = speech_to_text(
                    language="en",
                    start_prompt="REC",
                    stop_prompt="STOP",
                    just_once=True,
                    key="STT",
                )
            with col_text:
                text_input = st.chat_input("Type your answer...")

            if voice_text:
                handle_input(voice_text)
            elif text_input:
                handle_input(text_input)

    # --- SUMMARY & REPORT AT BOTTOM ---
    if "graph_state" in st.session_state and st.session_state.graph_state.get(
        "interview_complete", False
    ):
        st.markdown("---")
        st.subheader("📊 Summary & Report")

        state = st.session_state.graph_state
        evals = state["evaluations"]
        avg_tech = (
            sum(e["tech_score"] for e in evals) / len(evals)
            if evals
            else 0
        )

        st.metric("Final Technical Score", f"{avg_tech:.1f}/10")

        if evals:
            st.markdown("**Recent Answers:**")
            for e in evals[-3:]:
                q = e["question"]
                if len(q) > 70:
                    q = q[:70] + "..."
                st.markdown(
                    f"- Q: _{q}_ → Tech: **{e['tech_score']}/10**, Comm: **{e['comm_score']}/10**"
                )

        pdf_bytes = generate_pdf_report(state)
        st.download_button(
            label="📄 Download Interview Report (PDF)",
            data=pdf_bytes,
            file_name="interview_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
