import streamlit as st
import requests

st.title("URL Shortener")
st.header("url shortner")
url = st.text_input("Enter the URL to shorten:")

if st.button("Shorten"):
    res = requests.post("http://localhost:8000/shorten", json={"url": url})
    if res.ok:
        st.success(f"Short URL: {res.json()['short_url']}")
    else:
        st.error("Error shortening URL")