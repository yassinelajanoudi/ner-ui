import streamlit as st
from main import process_prompt

st.title("Entity Extraction with NLP Models")
st.write("Enter a text prompt to extract entities:")

prompt = st.text_input("Prompt:", "What projects have worked on youth livelihoods in West Africa?")
if st.button("Extract Entities"):
    with st.spinner("Processing..."):
        res = process_prompt(prompt)
        st.success("Extraction Complete!")
        # st.write(res)

        st.subheader("Extraction Results")
        st.write("Projects:", res.get("Projects"))
        st.write("Country:", res.get("Country"))
        st.write("Time:", res.get("Time"))
        st.write("Implementer:", res.get("Implementer"))

        st.subheader("Models Used")
        models_used = ["Projects", "Country", "Time", "Implementer"]
        used_status = ["Yes" if used else "No" for used in res["models"]]
        models_data = dict(zip(models_used, used_status))
        st.table(models_data)