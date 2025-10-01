import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import nltk

# ---------------------------
# NLTK Setup
# ---------------------------
for resource in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource)

# ---------------------------
# Utility Functions
# ---------------------------
def extract_text_from_url(url):
    """Scrape text content from a given URL, more robust."""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")

        # Grab <p>, <div>, <span> text
        elements = soup.find_all(["p", "div", "span"])
        text = " ".join([el.get_text(separator=" ", strip=True) for el in elements])
        text = " ".join(text.split())  # remove excessive whitespace

        return text if text else "No extractable content found."
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
st.markdown("Ask questions about any webpage content. Enter a URL and start chatting!")

# Sidebar Info
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        """
        This chatbot scrapes the text from a webpage (paragraphs, divs, spans),
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
    if "text_content" not in st.session_state or st.session_state.get("url") != url:
        st.session_state.text_content = extract_text_from_url(url)
        st.session_state.url = url
        st.session_state.conversation = []

    # Show preview
    st.subheader("📄 Extracted text preview (first 500 chars):")
    st.write(st.session_state.text_content[:500] + ("..." if len(st.session_state.text_content) > 500 else ""))

    # Question input
    question = st.text_input("❓ Ask a question:")

    if st.button("Get Answer") and question:
        if st.session_state.text_content.strip() == "" or st.session_state.text_content.startswith("Error"):
            st.warning("No content available to answer questions.")
        else:
            answer = find_best_answer(question, st.session_state.text_content)
            st.session_state.conversation.append((question, answer))

    # Clear chat button
    if st.button("🧹 Clear Chat"):
        st.session_state.conversation = []

    # Show conversation
    if st.session_state.conversation:
        st.subheader("💬 Conversation History")
        for q, a in st.session_state.conversation:
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {a}")
            st.markdown("---")
