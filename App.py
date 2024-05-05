import streamlit as st
from openai import OpenAI
from pinecone import Pinecone

# Initialize Pinecone
pc = Pinecone(api_key='50006779-0bb6-4f4d-b1f4-2b081ee4b5c9')
index = pc.Index("hackdavis")

st.title("MediB, your healthcare companion!")

# Set OpenAI API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Display chat messages from history on app rerun
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def pad_vector_to_size(vector, target_size, padding_value=0.0):
    current_size = len(vector)
    if current_size >= target_size:
        return vector
    else:
        padding = [padding_value] * (target_size - current_size)
        return vector + padding

# Initialize or update the conversation state
if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = []

questions = [
    "Write your ailment here:",
    "How long have you been experiencing these symptoms?",
    "What is your Name, gender, height, weight?",
    "Do you have any allergies?",
    "Are you currently taking any medication?"
]

current_question = questions[len(st.session_state.conversation_history)]
a_input = st.text_area(current_question, height=100)

instructions = """
Assume the role of a doctor who is conducting an online consultation. You have received a series of responses from a patient detailing their symptoms and medical history. Your task is to analyze the information provided, understand the context of the patient's health concerns, and respond with a professional medical opinion. 
- Answer the patient's questions directly based on the symptoms and history provided.
- Provide clear, medically sound advice and potential next steps for diagnosis or treatment.
- Ignore context information if it is irrelevant to the patient, eg: relating menstrual cyle to men. Judge the patient info and choose whether the info shud be considered or not.
- Do not talk about the context info if it shud be ignored
- Maintain a professional tone throughout, focusing solely on the information pertinent to the patient's queries and health.
- Avoid making assumptions about details not explicitly mentioned by the patient.
- Do not mention about the context info if it is not relevant. ignore it.
- Try being concise and also clear
This approach will help you address the patient's concerns accurately and professionally, without introducing unnecessary or irrelevant details."
"""

if st.button("Submit"):
    st.session_state.conversation_history.append(a_input)

    if len(st.session_state.conversation_history) < len(questions):
        st.experimental_rerun()
    else:
        #All questions answered, proceed to generate response
        # Convert the input to embeddings
        concatenated_context = " ".join(st.session_state.conversation_history)
        try:
            embedding_response = client.embeddings.create(
                input=concatenated_context,
                model="text-embedding-3-large"
            )
            embedding_vector = embedding_response.data[0].embedding
        except Exception as e:
            st.error(f"Failed to convert the description to embeddings: {e}")

        embedding_vector = pad_vector_to_size(embedding_vector, 3072)
        # Query Pinecone for the nearest match
        query_result = index.query(vector=embedding_vector, top_k=3, include_metadata=True)
        top_matches = [query['metadata']['description'] for query in query_result['matches']]
        context=""
        for matches in top_matches:
            context+=matches
        

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": instructions },
                {"role": "user", "content": f'patient:{concatenated_context}, use this intuition only if needed:{context}, take the gender of the patient in copnsideration'}
            ]
        )
        answer = response.choices[0].message.content
        st.markdown("## Diagnosis and Advice:")
        st.write(answer)
        st.session_state.messages.append({"role": "user", "content": concatenated_context})
        st.session_state.messages.append({"role": "assistant", "content": answer})
        # Reset the conversation for a new interaction
        st.session_state.conversation_history = []

# Optionally, display previous chat messages

prompt_report = """
Please generate a professional report summarizing the patient's information.
Keep the report focused on the patients answers only.
The information was taken in the following order: ailment, since how many days the patient has the symtoms, name, gender
, height, weight, allergy if any, medication if any taken. 
Ensure the report is concise and contains only information directly sourced from the patient and provided by the doctor.

"""

for message in st.session_state.messages:
    if message["role"] == "assistant":
        with st.chat_message("HealthCareBuddy"):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompt_report},
                    {"role": "user", "content": f'patient:{concatenated_context}, use this intuition only if needed:{context}, take the gender of the patient in consideration'}
                ]
            )
            answer = response.choices[0].message.content
            st.write(answer)
