import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import nltk

# Download punkt tokenizer (once)
nltk.download('punkt', quiet=True)

# ---------------------------
# Utility Functions
# ---------------------------

def extract_text_from_url(url):
    """Scrape text content from a given URL."""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        return text
    except Exception as e:
        return f"Error extracting content: {e}"

def find_best_answer(question, text):
    """Find the most relevant sentence from webpage text."""
    sentences = sent_tokenize(text)
    if not sentences:
        return "No content available to extract an answer from."

    all_text = [question] + sentences
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_text)
    question_vector = tfidf_matrix[0]
    sentence_vectors = tfidf_matrix[1:]
    similarities = cosine_similarity(question_vector, sentence_vectors)
    best_sentence_index = similarities.argmax()
    return sentences[best_sentence_index]

# ---------------------------
# Streamlit App
# ---------------------------

st.set_page_config(page_title="Webpage Q&A Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 Webpage Q&A Chatbot")
st.markdown("Ask questions about any webpage content. Just provide a URL and start chatting!")

# Sidebar Info
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        """
        This chatbot scrapes the text from a webpage (paragraphs only),
        then uses **TF-IDF + Cosine Similarity** to find the most relevant answer
        to your question.
        
        **Commands:**  
        - 🔄 Change URL → Enter new URL in the box  
        - 🧹 Clear Chat → Resets conversation  
        - ❌ Exit → Ends session  
        """
    )

# Input URL
url = st.text_input("🌐 Enter a URL to extract content:")

if url:
    if "text_content" not in st.session_state or st.session_state.url != url:
        st.session_state.text_content = extract_text_from_url(url)
        st.session_state.url = url
        st.session_state.conversation = []

    st.success("✅ Content extracted successfully!")

    # User question
    question = st.text_input("❓ Ask a question:")

    if st.button("Get Answer") and question:
        answer = find_best_answer(question, st.session_state.text_content)
        st.session_state.conversation.append((question, answer))

    # Clear chat button
    if st.button("🧹 Clear Chat"):
        st.session_state.conversation = []

    # Show conversation
    if st.session_state.conversation:
        st.markdown("### 💬 Conversation History")
        for q, a in st.session_state.conversation:
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {a}")
            st.markdown("---")