import pandas as pd
import streamlit as st

st.set_page_config(page_title="Detailed Results", layout="centered")

with open("frontend/styles/minimal.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Detailed Results")

if "result" not in st.session_state:
    st.info("No results yet. Go back and validate a citation first.")
    st.page_link("app.py", label="← Back to validator")
    st.stop()

result = st.session_state["result"]
matches = result.get("top_matches", [])

if not matches:
    st.warning("No matching candidates were found.")
    st.page_link("app.py", label="← Back to validator")
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
input_text = st.session_state.get("input_citation", "")

meta_fields = [
    ("Title", "title"),
    ("Authors", "authors"),
    ("Year", "year"),
    ("Venue", "venue"),
]

# We don't have parsed fields directly, so show input vs top DB record
st.markdown(f"**Input citation:** `{input_text}`")
st.markdown("**Closest match in database:**")

def _v(val):
    return str(val) if val is not None else "—"

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

st.page_link("app.py", label="← Back to validator")
