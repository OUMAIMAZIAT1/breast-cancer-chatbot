
import openai
import streamlit as st
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Fixer la détection pour des résultats reproductibles
DetectorFactory.seed = 0

# --- Configuration de Streamlit ---
st.set_page_config(page_title="Breast Cancer Support Chatbot", page_icon="🌸", layout="wide")

# --- Configuration de l'API OpenAI ---
openai.api_key = "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"

# --- CSS personnalisé ---
st.markdown("""
    <style>
    .suggestions {
        display: flex;
        justify-content: center;
        gap: 10px;
        margin-bottom: 5px;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        height: auto;
        overflow-y: auto;
        margin: 0;
        padding-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Détection de la langue ---
def detect_language(text):
    try:
        common_english_words = ["hi", "hello", "yes", "no", "ok", "thank you", "please"]
        if text.lower().strip() in common_english_words:
            return "en"
        return detect(text)
    except LangDetectException:
        return "en"

# --- Fonction pour obtenir les embeddings ---
def get_embedding(text, model="text-embedding-ada-002"):
    try:
        response = openai.Embedding.create(input=text, model=model)
        return np.array(response['data'][0]['embedding'])
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return None

# --- Fonction pour trouver la meilleure réponse ---
def find_best_response(user_query, responses):
    try:
        user_embedding = get_embedding(user_query)
        response_embeddings = [get_embedding(response) for response in responses]
        
        if any(embedding is None for embedding in response_embeddings):
            raise ValueError("Failed to generate embeddings for some responses.")
        
        similarities = [cosine_similarity([user_embedding], [embedding])[0][0] for embedding in response_embeddings]
        
        # Log the similarities
        st.markdown("### Debugging Similarity Scores:")
        for i, response in enumerate(responses):
            st.write(f"Response {i + 1}: {response}")
            st.write(f"Cosine Similarity: {similarities[i]:.4f}")
        
        best_response_index = np.argmax(similarities)
        best_similarity_score = similarities[best_response_index]
        return responses[best_response_index], best_similarity_score
    except Exception as e:
        st.error(f"Error finding best response: {e}")
        return "I'm sorry, I couldn't find a relevant answer.", 0

def get_response(prompt, user_language="en", user_name="", include_name=True, threshold=0.8):
    try:
        if include_name:
            system_instruction = f"You are a supportive assistant specialized in breast cancer. Address the user as {user_name} in the first message. Provide accurate and empathetic answers."
        else:
            system_instruction = "You are a supportive assistant specialized in breast cancer. Provide accurate and empathetic answers."
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300,
            n=3
        )
        possible_responses = [choice['message']['content'].strip() for choice in response['choices']]
        best_response, similarity_score = find_best_response(prompt, possible_responses)

        # Add a fallback for generic queries
        if similarity_score >= threshold:
            return best_response
        elif prompt.lower() in ["hello", "hi", "hey"]:
            return "Hello! How can I assist you today?"
        else:
            return "I'm sorry, I couldn't provide a highly relevant answer. Could you clarify your query?"
    except Exception as e:
        return f"Error: {e}"

import logging

# --- Configuration du fichier log ---
logging.basicConfig(
    filename="similarity_log.txt",  # Nom du fichier log
    level=logging.INFO,            # Niveau d'enregistrement
    format="%(asctime)s - %(levelname)s - %(message)s",  # Format des logs
    datefmt="%Y-%m-%d %H:%M:%S"    # Format de la date
)

# --- Fonction pour trouver la meilleure réponse ---
def find_best_response(user_query, responses):
    try:
        user_embedding = get_embedding(user_query)
        response_embeddings = [get_embedding(response) for response in responses]

        if any(embedding is None for embedding in response_embeddings):
            raise ValueError("Failed to generate embeddings for some responses.")

        # Calcul des similarités cosinus
        similarities = [cosine_similarity([user_embedding], [embedding])[0][0] for embedding in response_embeddings]
        best_response_index = np.argmax(similarities)
        best_similarity_score = similarities[best_response_index]

        # --- Écriture dans le fichier log ---
        logging.info("User Query: %s", user_query)
        for i, response in enumerate(responses):
            logging.info("Response %d: %s", i + 1, response)
            logging.info("Cosine Similarity: %.4f", similarities[i])
        logging.info("Best Response: %s", responses[best_response_index])
        logging.info("Best Similarity Score: %.4f", best_similarity_score)

        return responses[best_response_index], best_similarity_score
    except Exception as e:
        logging.error("Error finding best response: %s", e)
        return "I'm sorry, I couldn't find a relevant answer.", 0


# --- Affichage des messages ---
def display_message(content, is_user=True):
    if is_user:
        st.markdown(f"""
        <div style='background-color:#ffe6f0; padding:15px; border-radius:15px; margin-bottom:10px; max-width:80%;'>
            <strong>👤</strong> {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background-color:#e8f5e9; padding:15px; border-radius:15px; margin-bottom:10px; max-width:80%;'>
            <strong>🤖</strong> {content}
        </div>
        """, unsafe_allow_html=True)

# --- Initialisation de l'état ---
if "user_info" not in st.session_state:
    st.session_state.user_info = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "name_used" not in st.session_state:
    st.session_state.name_used = False

# --- Formulaire d'informations ---
if not st.session_state.user_info:
    with st.form("user_form"):
        st.markdown("### 📝 Please provide some basic information:")
        name = st.text_input("Your Name:", placeholder="Enter your name")
        age = st.number_input("Your Age:", min_value=1, max_value=120, step=1)
        gender = st.selectbox("Your Gender:", ["Select", "Female", "Male", "Other"])
        symptoms = st.text_area("Describe your symptoms (optional):", placeholder="E.g., lump in breast, pain, etc.")
        submitted = st.form_submit_button("Submit")

        if submitted:
            if name and gender != "Select":
                st.session_state.user_info = {"name": name, "age": age, "gender": gender, "symptoms": symptoms}
                st.session_state.messages.append({"role": "assistant", "content": f"Hello {name}, how can I assist you today?"})
                st.success(f"Thank you, {name}! Your information has been saved.")
            else:
                st.error("Please fill in all the required fields (Name and Gender).")

# --- Chatbot Section ---
if st.session_state.user_info:
    st.markdown(f"<h1 style='text-align: center; color: #e91e63;'>🌸 Breast Cancer Support Chatbot 🌸</h1>", unsafe_allow_html=True)

    st.markdown("<h3>Suggestions:</h3>", unsafe_allow_html=True)
    st.markdown("<div class='suggestions'>", unsafe_allow_html=True)
    suggestions = [
        "What are the symptoms of breast cancer?",
        "How can I reduce my risk of breast cancer?",
        "Tell me about breast cancer treatments.",
        "What is breast cancer?"
    ]
    cols = st.columns(len(suggestions))
    for i, suggestion in enumerate(suggestions):
        if cols[i].button(suggestion):
            st.session_state.messages.append({"role": "user", "content": suggestion})
            with st.spinner("🤖 Typing..."):
                chatbot_response = get_response(
                    suggestion,
                    user_name=st.session_state.user_info["name"],
                    include_name=not st.session_state.name_used
                )
            st.session_state.messages.append({"role": "assistant", "content": chatbot_response})
            st.session_state.name_used = True
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for message in st.session_state.messages:
        display_message(message["content"], is_user=message["role"] == "user")
    st.markdown("</div>", unsafe_allow_html=True)

    def send_message():
        if st.session_state.user_input:
            st.session_state.messages.append({"role": "user", "content": st.session_state.user_input})
            with st.spinner("🤖 Typing..."):
                chatbot_response = get_response(
                    st.session_state.user_input,
                    user_language=detect_language(st.session_state.user_input),
                    user_name=st.session_state.user_info["name"],
                    include_name=not st.session_state.name_used
                )
            st.session_state.messages.append({"role": "assistant", "content": chatbot_response})
            st.session_state.name_used = True
            st.session_state.user_input = ""

    st.text_input(
        "Type your message here...",
        key="user_input",
        on_change=send_message,
        placeholder="Type your message and press Enter...",
        label_visibility="collapsed",
    )
