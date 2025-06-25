import streamlit as st

st.title("Test Dashboard")
st.write("If you can see this, Streamlit is working!")

st.metric("Test Metric", "100")

if st.button("Click me"):
    st.write("Button clicked!") 