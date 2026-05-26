import json
import urllib.request
import urllib.error

import streamlit as st

API_URL = "http://localhost:8000/validate_batch"

st.set_page_config(page_title="Citation Validator", layout="centered")

with open("frontend/styles/minimal.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Citation Validator")

citation_text = st.text_area(
    "Paste raw citations below (one per line)",
    height=200,
    placeholder="Smith, J. (2020). Machine learning. Journal of AI.\nDoe, A. (2019). Deep learning. NeurIPS.\nMiller, R. (2021). Attention mechanisms. ACL.",
)

col1, _ = st.columns([1, 4])
with col1:
    submitted = st.button("Validate", type="primary", use_container_width=True)

if submitted:
    raw_lines = citation_text.strip().split("\n")
    citations = [line.strip() for line in raw_lines if line.strip()]

    if not citations:
        st.warning("Please enter at least one citation.")
        st.stop()

    payload = json.dumps({"citations": citations}).encode("utf-8")
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

    if "error" in data:
        st.error(data["error"])
        st.stop()

    st.session_state["batch_result"] = data
    st.session_state["input_citations"] = citations
    st.rerun()

if "batch_result" in st.session_state:
    results = st.session_state["batch_result"]["results"]
    input_citations = st.session_state["input_citations"]

    if len(results) == 1:
        r = results[0]
        label = r.get("label", "")
        confidence = r.get("confidence", 0.0)
        reason = r.get("reason", "")
        source = r.get("source", "unknown")

        source_label = {
            "db_heuristic": "Local Database Match",
            "llm_deepseek": "DeepSeek LLM Verified",
            "live_lookup": "Semantic Scholar API",
        }.get(source, source)

        if r.get("timed_out"):
            color = "#9e9e9e"
            bg = "#f5f5f5"
        elif label == "VALID":
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

        live = r.get("live_match")
        if live:
            st.markdown("### Live Match (Semantic Scholar)")
            st.markdown(
                f"**{live.get('title', '')}**  "
                f"({live.get('year', '')})<br>"
                f"{live.get('authors', '')}<br>"
                f"*{live.get('venue', '')}*",
                unsafe_allow_html=True,
            )

        matches = r.get("top_matches", [])
        if matches:
            st.subheader("Top DB Matches")
            for i, m in enumerate(matches[:3], 1):
                st.markdown(
                    f"**{i}.** {m.get('title', '')} ({m.get('year', '')})  "
                    f"— fuzzy:{m.get('fuzzy_score', 0):.0f} "
                    f"semantic:{m.get('semantic_score', 0):.2f} "
                    f"final:{m.get('final_score', 0):.2f}"
                )

        st.page_link("pages/results.py", label="View detailed results \u2192")

    else:
        st.subheader(f"Results ({len(results)} citations)")

        for i, r in enumerate(results):
            label = r.get("label", "")
            confidence = r.get("confidence", 0.0)
            reason = r.get("reason", "")
            source = r.get("source", "")

            source_label = {
                "db_heuristic": "DB",
                "llm_deepseek": "LLM",
                "live_lookup": "S2",
            }.get(source, source)

            if r.get("timed_out"):
                badge_color = "#9e9e9e"
                bg = "#f5f5f5"
            elif label == "VALID":
                badge_color = "#2e7d32"
                bg = "#e8f5e9"
            elif label == "PARTIALLY_VALID":
                badge_color = "#e65100"
                bg = "#fff3e0"
            else:
                badge_color = "#c62828"
                bg = "#ffebee"

            st.markdown(
                f'<div style="background:{bg};padding:0.75rem 1rem;'
                f'margin-bottom:0.5rem;border-radius:6px;'
                f'border-left:4px solid {badge_color};">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<span style="font-weight:600;">#{i+1}</span>'
                f'<span style="font-weight:700;color:{badge_color};">{label}</span>'
                f'<span>conf: {confidence:.2f}</span>'
                f'<span style="color:#888;font-size:0.85rem;">{source_label}</span>'
                f"</div>"
                f'<div style="color:#555;font-size:0.9rem;margin-top:0.25rem;">'
                f"{input_citations[i][:100]}{'...' if len(input_citations[i]) > 100 else ''}"
                f"</div>"
                f'<div style="color:#777;font-size:0.85rem;margin-top:0.15rem;">{reason}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
