# The UI: Text box for reviews, button to classify, and result output
from app.infer import predict
from app.config import APP_TITLE, APP_TAGLINE
import streamlit as st

st.set_page_config(page_title=APP_TITLE, page_icon="🎬")
st.title(APP_TITLE)
st.caption(APP_TAGLINE)

with st.expander("Examples"):
    col1, col2 = st.columns(2)
    pos_ex = "An unexpectedly heartfelt film with fantastic performances and tight pacing. Highly recommended."
    neg_ex = "Boring, overlong, and full of clichés. I struggled to finish it."
    if col1.button("Load Positive Example"):
        st.session_state["user_text"] = pos_ex
    if col2.button("Load Negative Example"):
        st.session_state["user_text"] = neg_ex

user_text = st.text_area("Paste a movie review:", value=st.session_state.get("user_text", ""), height=200)

run = st.button("Predict")
if run:
    with st.spinner("Running model..."):
        result = predict(user_text)
    
    if "error" in result:
        st.error(result["error"])
    else:
        label = "👍 Positive" if result["label"] == "pos" else "👎 Negative"
        st.subheader(label)
        st.write(f"Probability (positive): **{result['prob']:.3f}**")
        st.write(f"Latency: **{result['latency_ms']} ms** | Device: **{result['device']}**")

st.markdown("---")
st.markdown(
    "Model: Fine-tuned DistilBERT • Training data: IMDB • "
    "Tip: Longer inputs may be truncated for speed."
)