"""Streamlit front end for Cache. Reuses all the logic in main.py (memory, skip, safety, limits)."""
import uuid

import streamlit as st

st.set_page_config(page_title="Cache", page_icon="💬")

import main  # noqa: E402  (needs GEMINI_API_KEY in the environment / Streamlit secrets)

# One id per visitor, kept in the URL so a refresh or bookmark keeps the same memory.
uid = st.query_params.get("u")
if not uid:
    uid = "u" + uuid.uuid4().hex
    st.query_params["u"] = uid

st.title("Cache")
st.caption("Your “Real” Online Friend")

with st.sidebar:
    if st.button("Forget me"):
        main.forget(uid)
        st.toast("Memory cleared.")

text = st.chat_input("Say something", max_chars=500)
if text:
    try:
        with st.spinner("..."):
            main.chat(main.ChatIn(user_id=uid, message=text))
    except Exception as e:
        st.toast(getattr(e, "detail", None) or "Message failed. Try again.")

for m in main.load(uid)["msgs"][-30:]:
    with st.chat_message("user" if m["role"] == "user" else "assistant"):
        st.write(m["text"])
