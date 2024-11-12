from openai import OpenAI
from io import BytesIO
from streamlit import session_state as ss
from wordcloud import WordCloud
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from textwrap import wrap
from PIL import Image

import streamlit as st
import matplotlib.pyplot as plt
import pdfplumber
import base64
import fitz

# It still has issues with the theme change (light/dark)
# and the file downloaded is not the best haha!!
# But it's a decent first try with GPT
# ##

# Define the model to use
model = "gpt-3.5-turbo"

# Set OpenAI API key from Streamlit secrets
api_key = st.secrets['openai']['api_key']

# Define the client
client = OpenAI(api_key=api_key)

# Error handling for OpenAI API requests
def safe_api_call(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        st.error(f"Error occurred: {e}")
        return None

# Function to save text to a file (e.g., PDF)
def save_text_to_file(text, filename="output.pdf"):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Initialize a text object for writing content at the top margin
    text_object = c.beginText(40, height - 40)
    text_object.setFont("Helvetica", 12)
    text_object.setFillColor(colors.black)

    # Wrap text to fit within page width
    max_line_width = 500  # Width limit for the text in pixels
    lines = text.splitlines()
    wrapped_text = []
    for line in lines:
        wrapped_text.extend(wrap(line, width=max_line_width // 6))  # Adjust wrap width based on font size

    # Add wrapped text line-by-line to the PDF
    for line in wrapped_text:
        text_object.textLine(line)

    c.drawText(text_object)
    c.showPage()
    c.save()
    buffer.seek(0)

    # Streamlit download button for the PDF
    st.download_button(
        label="Download",
        data=buffer,
        file_name=filename,
        mime="application/pdf"
    )
    
# Translate text function
def translate_text(text, target_language='English'):
    return safe_api_call(client.chat.completions.create,
                         model=model,
                         messages=[{"role": "system", "content": f"Translate the following text into {target_language}:\n\n{text}"}])

@st.cache_data
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wordcloud

def display_wordcloud():
    # Generate the word cloud using the cached text data
    wordcloud = generate_wordcloud(st.session_state.text)
    
    # Create a matplotlib figure to display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)  # Display the word cloud in Streamlit

def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        return text

def extract_text_from_pdf_bytes(file_bytes):
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        num_pages = len(pdf.pages)
        st.write(f"Number of pages in the PDF: {num_pages}")
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
        return text

def evaluate_article_quality(text):
    response = safe_api_call(client.chat.completions.create,
                             model=model,
                             messages=[{"role": "system", "content": "Evaluate the quality of the following article based on clarity, methodology, and depth of analysis. Provide a score between (1 - 10) with an explanation."},
                                       {"role": "user", "content": text}])
    return response.choices[0].message.content if response else "Unable to evaluate quality."

def analyze_tone(text):
    response = safe_api_call(client.chat.completions.create,
                             model=model,
                             messages=[{"role": "system", "content": "Analyze the tone of the following text (e.g., neutral, critical, optimistic, etc.)"},
                                       {"role": "user", "content": text}])
    return response.choices[0].message.content if response else "Unable to analyze tone."

def summarize_text(text):
    response = safe_api_call(client.chat.completions.create,
                             model=model,
                             messages=[{"role": "system", "content": "Summarize the following technical article in plain language."},
                                       {"role": "user", "content": text}])
    return response.choices[0].message.content if response else "Unable to summarize."

def critique_article(text):
    response = safe_api_call(client.chat.completions.create,
                             model=model,
                             messages=[{"role": "system", "content": "Provide a critique of the following technical article, focusing on methodology, data robustness, and any strengths and weaknesses."},
                                       {"role": "user", "content": text}])
    return response.choices[0].message.content if response else "Unable to provide critique."

def generate_real_world_insights(text):
    response = safe_api_call(client.chat.completions.create,
                             model=model,
                             messages=[{"role": "system", "content": "Identify real-world applications or implications of the findings in the following article."},
                                       {"role": "user", "content": text}])
    return response.choices[0].message.content if response else "Unable to generate real-world insights."

def process_article_by_path(file_path):
    text = extract_text_from_pdf(file_path)
    summary = summarize_text(text)
    critique = critique_article(text)
    insights = generate_real_world_insights(text)
    tone = analyze_tone(text)
    quality = evaluate_article_quality(text)

    st.session_state.text = text
    st.session_state.tone = tone
    st.session_state.quality = quality
    st.session_state.summary = summary
    st.session_state.critique = critique

    return {"Text": text, "Tone": tone, "Quality": quality, "Summary": summary, "Critique": critique, "Real-World Insights": insights}

def process_article_by_bytes(file_bytes):
    text = extract_text_from_pdf_bytes(file_bytes)
    summary = summarize_text(text)
    critique = critique_article(text)
    insights = generate_real_world_insights(text)
    return {"Summary": summary, "Critique": critique, "Real-World Insights": insights}

def displayPDF_with_viewer(pdf):
    base64_pdf = base64.b64encode(pdf.read()).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="650" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

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
        
# Validate file upload type
def validate_file_type(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.type != 'application/pdf':
            st.error("Please upload a valid PDF file.")
            return False
    return True

# Translation Section
def translation_section(results):
    # Language selection for translation
    target_lang = st.selectbox("Select language for translation", ["English", "Spanish", "French", "German", "Italian", "Portuguese", "Dutch", "Russian", "Chinese", "Japanese"])

    # Translate and display text
    with st.spinner(f'Translating your article to ({target_lang}), please wait...'):
        if target_lang != "English":
            with st.expander(f"Translated Summary ({target_lang})"):
                st.subheader(f"Translated Summary ({target_lang}):")
                translated_summary = translate_text(results['Summary'], target_lang)
                st.success('Processing complete!')
                st.write(translated_summary)

            with st.expander(f"Translated Critique ({target_lang})"):
                st.subheader(f"Translated Critique ({target_lang}):")
                translated_critique = translate_text(results['Critique'], target_lang)
                st.success('Processing complete!')
                st.write(translated_critique)

            with st.expander(f"Translated Real-World Insights ({target_lang})"):
                st.subheader(f"Translated Real-World Insights ({target_lang}):")
                translated_insights = translate_text(results['Real-World Insights'], target_lang)
                st.success('Processing complete!')
                st.write(translated_insights)

# Streamlit UI Layout
st.markdown('<p class="title">Article Summary and Critique Generator</p>', unsafe_allow_html=True)
st.write("Upload your article and get a summary, critique, and insights.")

# Document upload section with title
st.subheader("1. Upload Your Article")
st.write("Upload a PDF file to get started.")
pdf = st.file_uploader("Upload PDF file", type='pdf', label_visibility="collapsed")

if pdf and validate_file_type(pdf):
    st.success("PDF uploaded successfully!")
    with st.expander("View PDF"):
      displayPDF(pdf)
    
    with st.spinner('Processing your article...'):
        results = process_article_by_path(pdf)
        st.success('Processing complete!')

        # Interactive Article Summary, Tone, and Quality Analysis
        st.subheader("2. Article Summary, Tone, and Quality Analysis")
        
        # 2-column layout for the checkboxes
        col1, col2 = st.columns(2)
        with col1:
            include_summary = st.checkbox("Include Summary")
            include_critique = st.checkbox("Include Critique")

        with col2:
            include_tone = st.checkbox("Include Tone Analysis")
            include_quality = st.checkbox("Include Quality Evaluation")

        # Display the sections based on checkbox selections
        if include_summary:
            with st.expander("📝 **Summary**"):
                st.write(results['Summary'])
        
        if include_critique:
            with st.expander("🧐 **Critique**"):
                st.write(results['Critique'])
        
        if include_tone:
            with st.expander("🔊 **Tone**"):
                st.write(results['Tone'])
        
        if include_quality:
            with st.expander("⚖️ **Quality**"):
                st.write(results['Quality'])

        with st.expander("💡 **Real-World Insights**"):
            st.write(results['Real-World Insights'])

        # Save and download the summary
        save_text_to_file(results['Summary'], filename="summary.pdf")

    # Call translation section
    translation_section(results)

# Footer
st.markdown("---")
st.markdown("""**Contact**: [My Email](mailto:k_parker@coloradocollege.edu)  
**Source Code**: [GitHub Repo](https://github.com/khayyon1)  
**Disclaimer**: This tool uses OpenAI's GPT model for generating summaries and critiques of articles.""")
