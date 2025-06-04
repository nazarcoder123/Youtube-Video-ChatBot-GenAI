import streamlit as st
import requests
import json
from typing import Dict, List
import time

# Constants
API_BASE_URL = "http://localhost:8000/api"

# Page Configuration
st.set_page_config(
    page_title="YouTube Video Chat",
    page_icon="🎥",
    layout="wide"
)

# Initialize session state variables
if 'conversation_id' not in st.session_state:
    st.session_state.conversation_id = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'video_metadata' not in st.session_state:
    st.session_state.video_metadata = None

def initialize_video(youtube_url: str) -> Dict:
    """Initialize a new video chat session"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/initialize",
            json={"yt_link": youtube_url}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error initializing video: {str(e)}")
        return None

def send_chat_message(question: str) -> Dict:
    """Send a chat message and get response"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={"question": question}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error sending message: {str(e)}")
        return None

def get_conversation_history(conversation_id: str) -> Dict:
    """Fetch conversation history"""
    try:
        response = requests.get(f"{API_BASE_URL}/conversation/{conversation_id}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching conversation: {str(e)}")
        return None

def display_video_metadata():
    """Display video metadata in the sidebar"""
    if st.session_state.video_metadata:
        st.sidebar.image(st.session_state.video_metadata["thumbnail_url"], use_column_width=True)
        st.sidebar.title(st.session_state.video_metadata["title"])
        st.sidebar.write(f"👤 Author: {st.session_state.video_metadata['author']}")
        st.sidebar.write(f"👁️ Views: {st.session_state.video_metadata['view_count']:,}")
        st.sidebar.write(f"📅 Published: {st.session_state.video_metadata['publish_date']}")
        
        with st.sidebar.expander("📝 Description"):
            st.write(st.session_state.video_metadata["description"])

def display_chat_messages():
    """Display chat messages with a nice UI"""
    for question, answer in st.session_state.chat_history:
        # User message
        message_container = st.container()
        with message_container:
            col1, col2 = st.columns([1, 11])
            with col1:
                st.markdown("👤")
            with col2:
                st.markdown(f"**You:** {question}")

        # AI response
        message_container = st.container()
        with message_container:
            col1, col2 = st.columns([1, 11])
            with col1:
                st.markdown("🤖")
            with col2:
                st.markdown(f"**Assistant:** {answer}")
        
        st.markdown("---")

# Main UI Layout
st.title("🎥 YouTube Video Chat")
st.markdown("Chat with your favorite YouTube videos using AI!")

# URL Input Section
with st.container():
    youtube_url = st.text_input(
        "Enter YouTube URL",
        placeholder="https://www.youtube.com/watch?v=..."
    )
    
    if st.button("Start Chat", type="primary"):
        with st.spinner("Initializing video..."):
            result = initialize_video(youtube_url)
            if result and result["status"] == "success":
                st.session_state.conversation_id = result["conversation_id"]
                st.session_state.video_metadata = result["metadata"]
                st.session_state.chat_history = []
                st.success("Video initialized successfully!")
                st.rerun()

# Chat Interface
if st.session_state.conversation_id:
    display_video_metadata()
    
    # Chat input and messages
    st.markdown("### Chat")
    
    # Display chat history
    display_chat_messages()
    
    # Message input
    message = st.text_input(
        "Type your message",
        key="message_input",
        placeholder="Ask something about the video..."
    )
    
    if st.button("Send", type="primary"):
        if message:
            with st.spinner("Getting response..."):
                response = send_chat_message(message)
                if response and response["status"] == "success":
                    # Update chat history
                    st.session_state.chat_history.append(
                        (message, response["answer"])
                    )
                    # Clear input
                    st.rerun()
        else:
            st.warning("Please enter a message.")

# Health check indicator in sidebar footer
try:
    health_response = requests.get(f"{API_BASE_URL}/health")
    if health_response.status_code == 200:
        st.sidebar.markdown("---")
        st.sidebar.success("📡 API Connected")
except:
    st.sidebar.error("📡 API Disconnected")
