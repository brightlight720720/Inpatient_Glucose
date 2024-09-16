import streamlit as st
from langchain.chat_models import ChatOpenAI
from .crew import run_crew

def main():
    st.title("Inpatient Glucose Control Assistant")

    # Set up the LLM
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.8, openai_api_key=st.secrets["OPENAI_API_KEY"])

    # Create a text area for the progress note input
    progress_note = st.text_area("Enter the patient's progress note:", height=200)

    if st.button("Generate Recommendations"):
        if progress_note:
            result = run_crew(progress_note, llm)
            st.subheader("Recommendations:")
            st.write(result)
        else:
            st.warning("Please enter a progress note.")

if __name__ == "__main__":
    main()
