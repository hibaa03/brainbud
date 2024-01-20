import streamlit as st
from clarifai.client.model import Model
import base64
import os
from dotenv import load_dotenv
from PIL import Image

# Loading environment variables
load_dotenv()
clarifai_pat = os.getenv("CLARIFAI_PAT")
openai_api_key = os.getenv("OPEN_AI")


if 'chapters' not in st.session_state:
    st.session_state['chapters'] = []
if 'correct_answer' not in st.session_state:
    st.session_state['correct_answer'] = ''

# Function to generate chapter options
def generate_chapter_options(subject):
    prompt = f"List chapters for the subject: {subject} for kids."
    model_prediction = Model(
        "https://clarifai.com/openai/chat-completion/models/gpt-4-turbo"
    ).predict_by_bytes(
        prompt.encode(), input_type="text"
    )
    chapters = model_prediction.outputs[0].data.text.raw.split('\n')
    return [chapter.strip() for chapter in chapters if chapter.strip()]



# Function to generate study content and quiz questions
def generate_study_content(subject, chapter, age_group):
    prompt = f"Create educational content suitable for children aged {age_group} on the subject of '{subject}', focusing on '{chapter}'."

    model_prediction = Model(
        "https://clarifai.com/openai/chat-completion/models/gpt-4-turbo"
    ).predict_by_bytes(
        prompt.encode(), input_type="text"
    )

    content = model_prediction.outputs[0].data.text.raw.strip()
    return content if content else "Study content not available."




def generate_practice_questions(subject, chapter, age_group, num_questions=3):
    prompt = (f"Generate {num_questions} practice quiz questions for children aged {age_group} "
              f"on the topic of '{subject}: {chapter}'.")

    model_prediction = Model(
        "https://clarifai.com/openai/chat-completion/models/gpt-4-turbo"
    ).predict_by_bytes(
        prompt.encode(), input_type="text"
    )

    questions = model_prediction.outputs[0].data.text.raw.strip().split('\n')
    return questions if questions else ["No questions available."]



# Function to generate images related to the chapter
def generate_related_image(keyword):
    prompt = f"Generate an image related to: {keyword}"
    model_prediction = Model(
        "https://clarifai.com/openai/dall-e/models/dall-e-3"
    ).predict_by_bytes(
        prompt.encode(), input_type="text"
    )
    image_base64 = model_prediction.outputs[0].data.image.base64
    return image_base64

# practice questions
def process_practice_questions(questions):
    grouped_questions = []
    current_question = []

    for line in questions:
        if line.strip():
            current_question.append(line)
        else:
            if current_question:
                grouped_questions.append("\n".join(current_question))
                current_question = []
    if current_question:  # Add the last question if there's no blank line after it
        grouped_questions.append("\n".join(current_question))

    return grouped_questions



def main():
    # Set page configuration
    st.set_page_config(page_title='BrainBud', layout='wide')

    # Page title and introduction using HTML for styling
    st.markdown("<h1 style='text-align: center; color: green;'>Welcome to BrainBud!</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: blue;'>Let's start learning!</h2>", unsafe_allow_html=True)

    # Initialize variables
    if 'study_content' not in st.session_state:
        st.session_state['study_content'] = "Click 'Get Study Material and Quiz' to generate content."
    if 'chapters' not in st.session_state:
        st.session_state['chapters'] = []
    chapter = None
    processed_questions = []  # Initialize with an empty list

    # Sidebar for subject and age group selection
    with st.sidebar:
        st.header("📘 Choose Your Subject and Age Group")
        subject = st.selectbox("Select a subject:", ["Science", "Mathematics", "Geography", "History", "English"])
        age_group = st.slider("Select age group:", 5, 15, 10)

    # Chapter selection
    st.subheader("📚 Choose a Chapter")
    if st.button("Get Chapters"):
        st.session_state['chapters'] = generate_chapter_options(subject)
    
    if st.session_state['chapters']:
        chapter = st.selectbox("Select a chapter:", st.session_state['chapters'])

    # Display study content and practice questions
    if chapter:
        st.subheader(f"📖 Learning Material for {chapter}")
        if st.button("Get Study Material and Practice Questions"):
            st.session_state['study_content'] = generate_study_content(subject, chapter, age_group)
            practice_questions = generate_practice_questions(subject, chapter, age_group)
            processed_questions = process_practice_questions(practice_questions)

        st.text_area("Study Content", st.session_state['study_content'], height=300)
        
        st.subheader("❓ Practice Questions")
        for i, question in enumerate(processed_questions, 1):
            st.markdown(f"**Q{i}:** {question}")

    # Image generation section
    if chapter and st.button("🖼️ Show Related Image"):
        related_image = generate_related_image(chapter)
        st.image(related_image, caption="Image Related to the Chapter", use_column_width=True)

if __name__ == "__main__":
    main()