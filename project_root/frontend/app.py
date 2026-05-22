import json
import urllib.request
import urllib.error

import streamlit as st

API_URL = "http://localhost:8000/validate"

st.set_page_config(page_title="Citation Validator", layout="centered")

with open("frontend/styles/minimal.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Citation Validator")

citation = st.text_area(
    "Paste a raw citation below",
    height=150,
    placeholder="e.g. Smith, J. (2020). Machine learning. Journal of AI.",
)

col1, _ = st.columns([1, 4])
with col1:
    submitted = st.button("Validate", type="primary", use_container_width=True)

if submitted:
    if not citation.strip():
        st.warning("Please enter a citation.")
        st.stop()

    payload = json.dumps({"citation": citation.strip()}).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError:
        st.error("Could not reach the API at localhost:8000. Is the server running?")
        st.stop()

    st.session_state["result"] = data
    st.session_state["input_citation"] = citation.strip()
    st.rerun()

if "result" in st.session_state:
    result = st.session_state["result"]
    label = result.get("label", "")
    confidence = result.get("confidence", 0.0)
    reason = result.get("reason", "")
    source = result.get("source", "unknown")

    source_label = {
        "db_heuristic": "Local Database Match",
        "llm_deepseek": "DeepSeek LLM Verified",
        "live_lookup": "Semantic Scholar API",
    }.get(source, source)

    if label == "VALID":
        color = "#2e7d32"
        bg = "#e8f5e9"
    elif label == "PARTIALLY_VALID":
        color = "#e65100"
        bg = "#fff3e0"
    else:
        color = "#c62828"
        bg = "#ffebee"

    st.markdown(
        f'<div class="result-box" style="background:{bg};padding:1.5rem;'
        f'border-radius:8px;border-left:5px solid {color};">'
        f'<div style="font-size:2rem;font-weight:700;color:{color};">{label}</div>'
        f'<div style="font-size:1.2rem;margin-top:0.5rem;">'
        f"Confidence: <strong>{confidence:.2f}</strong></div>"
        f'<div style="margin-top:0.25rem;color:#888;">Source: {source_label}</div>'
        f'<div style="margin-top:0.5rem;color:#555;">{reason}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    live = result.get("live_match")
    if live:
        st.markdown("### Live Match (Semantic Scholar)")
        st.markdown(
            f"**{live.get('title', '')}**  "
            f"({live.get('year', '')})<br>"
            f"{live.get('authors', '')}<br>"
            f"*{live.get('venue', '')}*",
            unsafe_allow_html=True,
        )

    matches = result.get("top_matches", [])
    if matches:
        st.subheader("Top DB Matches")
        for i, m in enumerate(matches[:3], 1):
            st.markdown(
                f"**{i}.** {m.get('title', '')} ({m.get('year', '')})  "
                f"— fuzzy:{m.get('fuzzy_score', 0):.0f} "
                f"semantic:{m.get('semantic_score', 0):.2f} "
                f"final:{m.get('final_score', 0):.2f}"
            )

    st.page_link("pages/results.py", label="View detailed results →")
