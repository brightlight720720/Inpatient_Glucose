import streamlit as st
from langchain.chat_models import ChatOpenAI
from inpt_glucose.crew import run_crew



def main():
    st.title("Inpatient Glucose Control Assistant")

    # Create a text area for the progress note input
    progress_note = st.text_area(
        "Enter the patient's progress note:",
        """
        Patient: John Doe
        Age: 65
        Weight: 80 kg
        Recent HbA1c: 8.5%
        Current medications: Metformin 1000mg twice daily
        Recent glucose readings:
        - Fasting: 160 mg/dL
        - 2 hours after breakfast: 220 mg/dL
        - Before lunch: 180 mg/dL
        - 2 hours after lunch: 240 mg/dL
        - Before dinner: 190 mg/dL
        - 2 hours after dinner: 250 mg/dL
        - Bedtime: 200 mg/dL
        """
    )

    # Initialize the LLM
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.8,openai_api_key=st.secrets["OPENAI_API_KEY"])

    if st.button("Generate Recommendations"):
        with st.spinner("Generating recommendations..."):
            # Run the crew with the progress note
            result = run_crew(progress_note, llm)
           # raw_output = result.get("raw", "No recommendation found.")

        # Display the result
        st.subheader("Recommendations:")
        st.write(result)
        #st.write(raw_output)

if __name__ == "__main__":
    main()
