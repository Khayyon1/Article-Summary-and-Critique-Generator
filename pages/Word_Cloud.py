import streamlit as st
import matplotlib.pyplot as plt
import base64
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import fitz
import pdfplumber

from io import BytesIO
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image  # Make sure to import Image from PIL (Pillow)

def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        return text

# Function to convert PDF to images using PyMuPDF (fitz)
def convert_pdf_to_images(pdf_file):
    # Open the PDF from the file-like object (BytesIO)
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")  # Open the file using a stream

    images = []
    
    # Iterate through each page and convert to image
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)  # Loads the page
        pix = page.get_pixmap()  # Converts the page to a pixmap (image)
        
        # Convert pixmap to PIL image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    
    return images

def displayPDF(pdf):
    # Convert PDF to images using PyMuPDF
    images = convert_pdf_to_images(pdf)

    # Display each page as an image
    for page_num, img in enumerate(images):
        st.image(img, caption=f"Page {page_num + 1}", use_column_width=True)


def show_word_frequencies(text):
    # Tokenize the text and count word frequencies
    words = text.split()
    word_counts = Counter(words)
    
    # Convert the word counts to a DataFrame for better display
    word_freq_df = pd.DataFrame(word_counts.items(), columns=["Word", "Frequency"])
    word_freq_df = word_freq_df.sort_values(by="Frequency", ascending=False).head(10)

    st.write("Top 10 Most Frequent Words:")
    st.dataframe(word_freq_df)

def generate_wordcloud(text, max_words=200, bg='white', colormap='viridis'):
    wordcloud = WordCloud(width=800, height=400, max_words=max_words, background_color=bg, colormap=colormap).generate(text)
    return wordcloud

def display_wordcloud(max_words=200, bg='white', colormap="viridis"):
    # Generate the word cloud using the cached text data
    wordcloud = generate_wordcloud(st.session_state.text, max_words=max_words, bg=bg)
    
    # Create a matplotlib figure to display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)  # Display the word cloud in Streamlit
    
    # Show word frequency information
    show_word_frequencies(text)
    
    # Save the wordcloud image to a BytesIO object and allow users to download
    img_buf = BytesIO()
    wordcloud.to_image().save(img_buf, format="PNG")
    img_buf.seek(0)
    img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
    img_download_link = f'<a href="data:image/png;base64,{img_base64}" download="wordcloud.png">Download Word Cloud Image</a>'
    st.markdown(img_download_link, unsafe_allow_html=True)

def interactive_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400).generate(text)

    # Convert word cloud data to a DataFrame
    words = list(wordcloud.words_.keys())
    frequencies = list(wordcloud.words_.values())
    
    word_df = pd.DataFrame({
        "word": words,
        "frequency": frequencies
    })

    # Create an interactive plotly word cloud
    fig = px.scatter(word_df, x="word", y="frequency", text="word", size="frequency",
                     title="Interactive Word Cloud", labels={"frequency": "Word Frequency"})
    fig.update_traces(marker=dict(symbol="circle"), textposition="top center")
    st.plotly_chart(fig)

def animated_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400).generate(text)
    word_freq = wordcloud.words_

    # Create an animated plot with Plotly
    fig = go.Figure()

    for word, freq in word_freq.items():
        fig.add_trace(go.Scatter(x=[freq], y=[word], mode="markers", marker=dict(size=freq*100, color="blue", opacity=0.5)))

    fig.update_layout(title="Animated Word Cloud", showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False))
    st.plotly_chart(fig)

def difference_wordclouds(text1, text2):
    wordcloud1 = WordCloud(width=800, height=400).generate(text1)
    wordcloud2 = WordCloud(width=800, height=400).generate(text2)

    # Find the words that are in text1 but not in text2
    unique_to_text1 = {word: freq for word, freq in wordcloud1.words_.items() if word not in wordcloud2.words_}
    unique_to_text2 = {word: freq for word, freq in wordcloud2.words_.items() if word not in wordcloud1.words_}

    # Create two separate word clouds for differences
    diff_wc1 = WordCloud(width=800, height=400).generate_from_frequencies(unique_to_text1)
    diff_wc2 = WordCloud(width=800, height=400).generate_from_frequencies(unique_to_text2)

    st.subheader("Unique Words to Text 1")
    plt.figure(figsize=(10, 5))
    plt.imshow(diff_wc1, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

    st.subheader("Unique Words to Text 2")
    plt.figure(figsize=(10, 5))
    plt.imshow(diff_wc2, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

def calculate_similarity(text1, text2):
    vectorizer = CountVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(vectors[0], vectors[1])
    return similarity[0][0]

def display_similarity_score(text1, text2):
    similarity = calculate_similarity(text1, text2)
    st.write(f"Cosine Similarity Score: {similarity:.2f}")

def display_wordclouds(text1, text2):
    display_similarity_score(text1, text2)
    difference_wordclouds(text1, text2)
# Word cloud generation section
# Streamlit UI Layout
st.title("Word Cloud Generator")

# Provide instructions or context
st.write("Use the word cloud below to get a quick visual representation of the most frequent terms in your uploaded text.")

# Add a text input or use session state to store the text (could be from a PDF, or user input)
if "text" in st.session_state:
    text = st.session_state.text
else:
    st.write("Please upload a document and generate a summary and critique first.")
    text = ""

# If the text is available, allow the user to generate and view the word cloud
if text:
    st.subheader("Generate Word Cloud")
    st.write("Click the button below to generate the word cloud based on the uploaded article's content.")
    
    # Customize the word cloud (optional controls for users)
    max_words = st.slider("Max number of words", 50, 500, 100)
    bg_color = st.selectbox("Background color", ["white", "black", "blue", "yellow", "green", "red"])
    colormap = st.selectbox("Color Scheme", ["viridis", "plasma", "inferno", "magma", "cividis", "coolwarm"])


    if st.button("Generate Word Cloud"):
        with st.spinner("Generating word cloud..."):
            display_wordcloud(max_words, bg_color, colormap)
            
            interactive_wordcloud(text)
            
            animated_wordcloud(text)
    
    st.write("**Do you want to compare this text with another text, just upload here")
    st.write("A difference wordcloud and a similarity score will be given to show the diference in the 2 text provided")    
    st.write("Upload another PDF file to get started.")
    pdf2 = st.file_uploader("Upload file", type='pdf')

    if pdf2:
        st.success("PDF uploaded successfully!")
        with st.expander("View Second PDF"):
            displayPDF(pdf2)
    
        with st.spinner('Processing your article...'):
            text2 = extract_text_from_pdf(pdf2)
            display_wordclouds(text, text2)

# Add link to go back to the main page for summary/critique
st.markdown("---")
st.markdown("Connect with me on Medium or Linkedin and let me know your thoughts")
# st.write("[Go back to the main page for Summary & Critique](#)")  # Replace with the actual link if needed