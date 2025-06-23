# Webpage-Chatbot-for-Content-Extraction-and-Q&A

This project demonstrates a web-based chatbot built using Python and Jupyter widgets. The chatbot extracts text from a user-provided URL, processes user questions, and provides relevant answers by leveraging TF-IDF (Term Frequency-Inverse Document Frequency) and Cosine Similarity for text matching. The chatbot also supports additional features such as URL changes, conversation history clearing, and exiting the session.

## Key Features
- **Text Extraction:** Extracts content from the < p > tags of the provided web page.
- **Question Answering:** Finds the most relevant answer to user questions based on the extracted content.
- **URL Change:** Allows users to change the URL to fetch new content without restarting the chatbot.
- **Conversation History:** Maintains a log of all questions and answers in a scrollable widget box.
- **Clear History:** Provides an option to clear the conversation history.
- **Exit Chatbot:** Terminates the chatbot session.

## Screenshots
- **Chatbot Interaction with First Website:** Demonstrates how the chatbot responds to user queries for content from the first website.
  ![image](https://github.com/user-attachments/assets/d4f74be6-5f5b-41c9-bb41-0475c31b3f80)

- **Chatbot After Clearing Conversation History:** Shows the interface after the conversation history is cleared.
  ![image](https://github.com/user-attachments/assets/2c54703c-1dae-4173-8411-a1d814e89f7a)

- **Chatbot Interaction with Second Website:** Illustrates a new interaction after changing the URL to a second website.
  ![image](https://github.com/user-attachments/assets/689658e4-e287-4ce6-a48c-5c0f0a70791e)

  
## Technologies Used
- Python
- BeautifulSoup for web scraping.
- NLTK for sentence tokenization.
- Scikit-learn for TF-IDF vectorization and cosine similarity computation.
- IPyWidgets for creating an interactive chatbot interface.
