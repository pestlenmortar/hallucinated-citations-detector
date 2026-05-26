import pandas as pd
import streamlit as st

st.set_page_config(page_title="Detailed Results", layout="centered")

with open("frontend/styles/minimal.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Detailed Results")

has_single = "result" in st.session_state
has_batch = "batch_result" in st.session_state

if not has_single and not has_batch:
    st.info("No results yet. Go back and validate a citation first.")
    st.page_link("app.py", label="\u2190 Back to validator")
    st.stop()

if has_single:
    result = st.session_state["result"]
    if has_batch:
        st.session_state["batch_selected"] = None
elif has_batch:
    batch = st.session_state["batch_result"]
    results = batch["results"]
    input_citations = st.session_state.get("input_citations", [])
    selected = st.session_state.get("batch_selected")

    if selected is None:
        options = {
            i: f"#{i+1}: {input_citations[i][:60]}..." if i < len(input_citations) else f"#{i+1}"
            for i, r in enumerate(results)
        }
        sel = st.selectbox(
            "Select a citation result to inspect:",
            options=list(options.keys()),
            format_func=lambda k: options[k],
        )
        if st.button("Show details"):
            st.session_state["batch_selected"] = sel
            st.rerun()
        st.page_link("app.py", label="\u2190 Back to validator")
        st.stop()

    result = results[selected]

st.page_link("app.py", label="\u2190 Back to validator")

matches = result.get("top_matches", [])

if result.get("timed_out"):
    st.warning("This citation timed out during batch processing. No detailed results available.")
    st.stop()

if not matches:
    st.warning("No matching candidates were found.")
    st.stop()

parsed_label = result.get("label", "")
parsed_confidence = result.get("confidence", 0.0)
parsed_reason = result.get("reason", "")

st.markdown(f"**Label:** {parsed_label}  |  **Confidence:** {parsed_confidence:.2f}")
st.markdown(f"**Reason:** {parsed_reason}")
st.divider()

rows = []
for m in matches:
    rows.append(
        {
            "Title": m.get("title", ""),
            "Authors": m.get("authors", ""),
            "Year": m.get("year", ""),
            "Venue": m.get("venue", ""),
            "Score": m.get("final_score", 0),
        }
    )
df = pd.DataFrame(rows)
st.subheader("Top Matches")
st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Metadata Comparison")

top = matches[0]
input_text = st.session_state.get("input_citation", st.session_state.get("input_citations", [None])[selected] if has_batch and selected is not None else "")

meta_fields = [
    ("Title", "title"),
    ("Authors", "authors"),
    ("Year", "year"),
    ("Venue", "venue"),
]

st.markdown(f"**Input citation:** `{input_text}`")
st.markdown("**Closest match in database:**")


def _v(val):
    return str(val) if val is not None else "\u2014"


st.markdown(
    f"""
| Field | Database Value |
|-------|----------------|
| Title | {_v(top.get('title'))} |
| Authors | {_v(top.get('authors'))} |
| Year | {_v(top.get('year'))} |
| Venue | {_v(top.get('venue'))} |
""",
    unsafe_allow_html=True,
)

st.divider()
st.subheader("Score Breakdown")

scores = matches[0]
score_fields = [
    ("Fuzzy Score", "fuzzy_score"),
    ("Semantic Score", "semantic_score"),
    ("Metadata Score", "metadata_score"),
    ("Author Similarity", "author_similarity"),
    ("Year Similarity", "year_similarity"),
    ("Venue Similarity", "venue_similarity"),
    ("Final Score", "final_score"),
]

score_df = pd.DataFrame(
    [
        {"Component": name, "Value": round(scores.get(key, 0), 4)}
        for name, key in score_fields
    ]
)
st.dataframe(score_df, use_container_width=True, hide_index=True)
