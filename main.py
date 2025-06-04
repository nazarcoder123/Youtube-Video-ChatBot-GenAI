from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field 
from typing import List, Tuple, Union 
import os
import openai 
import re 
from pytube import YouTube 
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings 
from langchain.chains import ConversationalRetrievalChain 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import TextLoader 
from langchain_community.vectorstores import Chroma 
from youtube_transcript_api import YouTubeTranscriptApi 
from dotenv import load_dotenv 
import yt_dlp 
import uuid
from datetime import datetime 

# Load environment variables
load_dotenv()

app = FastAPI()

# CORS Middleware Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
qna_chain = None
conversations_storage = {}  # In-memory storage for conversations
current_conversation_id = None
openai.api_key = os.environ.get("GOOGLE_API_KEY") 

# --- Pydantic Models ---
class InitRequest(BaseModel):
    yt_link: str

class ChatRequest(BaseModel):
    question: str

class ConversationResponse(BaseModel):
    conversation_id: str

# --- Helper functions ---
def load_db(file, chain_type, k):
    transcript = TextLoader(file).load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=70)
    docs = text_splitter.split_documents(transcript)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7, convert_system_message_to_human=True),
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa

def buffer(history: List[Tuple[str, str]], buff_length: int) -> List[Tuple[str, str]]: 
    if len(history) > buff_length:
        return history[-buff_length:]
    return history

def is_valid_yt(link):
    pattern = r'^(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/watch\?v=|youtu\.be\/)([\w\-_]{11})(?:\S+)?$'
    match = re.match(pattern, link)
    if match:
        return True, match.group(1)
    else:
        return False, None

def get_metadata_pytube(video_id) -> dict:
    try:
        yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
        video_info = {
            "title": yt.title or "Unknown",
            "description": yt.description or "Unknown", 
            "view_count": yt.views or 0,
            "thumbnail_url": yt.thumbnail_url or "Unknown",
            "publish_date": yt.publish_date.strftime("%Y-%m-%d %H:%M:%S") if yt.publish_date else "Unknown",
            "length": yt.length or 0,
            "author": yt.author or "Unknown",
        }
        return video_info
    except Exception as e:
        print(f"Pytube failed: {e}")
        return None

def get_metadata_ytdlp(video_id) -> dict:
    try:
        ydl_opts = {'quiet': True, 'no_warnings': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
            thumbnail_url = info.get('thumbnail') 
            if not thumbnail_url and info.get('thumbnails'): 
                thumbnail_url = info.get('thumbnails', [{}])[0].get('url', 'Unknown')

            publish_date_str = info.get('upload_date', 'Unknown') 
            if publish_date_str != "Unknown" and len(publish_date_str) == 8:
                publish_date_str = f"{publish_date_str[0:4]}-{publish_date_str[4:6]}-{publish_date_str[6:8]} 00:00:00"

            video_info = {
                "title": info.get('title', 'Unknown'),
                "description": info.get('description', 'Unknown'),
                "view_count": info.get('view_count', 0),
                "thumbnail_url": thumbnail_url,
                "publish_date": publish_date_str,
                "length": info.get('duration', 0),
                "author": info.get('uploader', 'Unknown'),
            }
            return video_info
    except Exception as e:
        print(f"yt-dlp failed: {e}")
        return None
        
def get_metadata(video_id) -> dict:
    metadata = get_metadata_pytube(video_id)
    if metadata:
        return metadata
    print("Pytube failed, trying yt-dlp...")
    metadata = get_metadata_ytdlp(video_id)
    if metadata:
        return metadata
    print("Both methods failed, returning basic metadata...")
    return {
        "title": "Unknown",
        "description": "Unable to fetch video description.",
        "view_count": 0,
        "thumbnail_url": f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg", 
        "publish_date": "Unknown",
        "length": 0,
        "author": "Unknown",
    }

def save_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript_data = None
        
        # Try to get manually created English transcript first
        try:
            transcript_data = transcript_list.find_manually_created_transcript(['en']).fetch()
        except Exception: 
            pass
        
        # If no manual transcript, try auto-generated English transcript
        if not transcript_data:
            try:
                transcript_data = transcript_list.find_generated_transcript(['en']).fetch()
            except Exception: 
                pass
        
        # If still no transcript, try the basic get_transcript method
        if not transcript_data:
            try:
                transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
            except Exception as e_orig:
                print(f"Original get_transcript failed: {e_orig}. Trying to find any available.")
                try:
                    available_transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
                    for tr_item in available_transcripts: 
                        try:
                            transcript_data = tr_item.fetch()
                            if transcript_data: 
                                break
                        except Exception: 
                            pass
                except Exception:
                    pass
                
                if not transcript_data: 
                    return None
                    
    except Exception as e:
        print(f"Error fetching transcript for video {video_id}: {e}")
        return None

    if transcript_data:
        try:
            with open('transcript.txt', 'w', encoding='utf-8') as file:
                for entry in transcript_data:
                    # Handle both dictionary and object formats
                    if hasattr(entry, 'start') and hasattr(entry, 'text'):
                        # Object format (FetchedTranscriptSnippet)
                        start_time = int(entry.start)
                        text = entry.text
                    elif isinstance(entry, dict):
                        # Dictionary format
                        start_time = int(entry['start'])
                        text = entry['text']
                    else:
                        # Fallback - try to convert to string and extract info
                        entry_str = str(entry)
                        print(f"Unexpected entry format: {entry_str}")
                        continue
                    
                    file.write(f"~{start_time}~{text} ")
                    
            print(f"Transcript saved to: transcript.txt")
            return True
        except Exception as e:
            print(f"Error writing transcript file: {e}")
            return False
    
    return False

# --- API Routes ---

# API 1: Initialize with YouTube URL
@app.post("/api/initialize")
async def initialize_video(request_data: InitRequest):
    """
    API 1: Initialize the system with a YouTube URL
    This processes the video and prepares it for chat
    """
    global qna_chain, current_conversation_id, conversations_storage
    qna_chain = None 
    current_conversation_id = None
    
    try:
        yt_link = request_data.yt_link.strip()
        
        # Validate YouTube URL
        valid, video_id = is_valid_yt(yt_link)
        if not valid or not video_id:
            raise HTTPException(status_code=400, detail="Invalid YouTube link provided.")
        
        # Get video metadata
        metadata = get_metadata(video_id)
        
        # Clean up existing transcript file
        if os.path.exists('./transcript.txt'):
            try:
                os.remove('./transcript.txt')
            except OSError as e:
                print(f"Error removing existing transcript file: {e}")
        
        # Download and save transcript
        transcript_saved = save_transcript(video_id)
        if not transcript_saved:
            raise HTTPException(
                status_code=400, 
                detail="Could not fetch or save video transcript. The video might not have captions available."
            )
        
        # Check for API key
        if not os.environ.get("GOOGLE_API_KEY"):
            raise HTTPException(
                status_code=500, 
                detail="Google API key (GOOGLE_API_KEY) not found in environment variables."
            )
        
        # Initialize the QnA chain
        qna_chain = load_db("./transcript.txt", 'stuff', 5)
        
        # Create new conversation
        conversation_id = str(uuid.uuid4())
        current_conversation_id = conversation_id
        conversations_storage[conversation_id] = {
            "video_id": video_id,
            "video_metadata": metadata,
            "chat_history": [],
            "created_at": str(datetime.now()),
        }
        
        return JSONResponse(content={
            "status": "success",
            "message": "Video initialized successfully. You can now start chatting!",
            "conversation_id": conversation_id,
            "video_id": video_id,
            "metadata": metadata,
        }, status_code=200)
        
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Unexpected error in initialize_video: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"An unexpected error occurred during video initialization: {str(e)}"
        )

# API 2: Chat with the video (Simplified - only question needed)
@app.post("/api/chat")
async def chat_with_video(request_data: ChatRequest):
    """
    API 2: Chat with the initialized video
    Send only a question and get an answer - conversation is automatically stored
    """
    global qna_chain, current_conversation_id, conversations_storage
    
    try:
        question = request_data.question.strip()
        
        # Validate inputs
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
        if qna_chain is None:
            raise HTTPException(
                status_code=400, 
                detail="No video initialized. Please call /api/initialize first with a YouTube URL."
            )
        
        if not current_conversation_id or current_conversation_id not in conversations_storage:
            raise HTTPException(
                status_code=400, 
                detail="No active conversation found. Please initialize a video first."
            )
        
        # Get current conversation history
        conversation = conversations_storage[current_conversation_id]
        chat_history = conversation["chat_history"]
        
        # Buffer the chat history to prevent it from becoming too long
        buffered_history = buffer(chat_history, 7)
        
        # Generate response using the QnA chain
        response_data = qna_chain({
            'question': question, 
            'chat_history': buffered_history
        })
        
        answer = response_data.get('answer', 'No answer found.')
          # Store the conversation (question and answer)
        conversations_storage[current_conversation_id]["chat_history"].append((question, answer))
        
        return JSONResponse(content={
            "status": "success",
            "question": question,
            "answer": answer,
            "conversation_id": current_conversation_id
        }, status_code=200)
        
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error in chat_with_video: {type(e).__name__} - {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate response: {str(e)}"
        )

# Additional utility endpoint to get transcript
@app.get("/api/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """
    API 3: Retrieve conversation history by conversation ID
    """
    global conversations_storage
    
    try:
        if conversation_id not in conversations_storage:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        
        conversation = conversations_storage[conversation_id]
        
        return JSONResponse(content={
            "status": "success",
            "conversation_id": conversation_id,
            "video_metadata": conversation["video_metadata"],
            "chat_history": conversation["chat_history"],
            "created_at": conversation["created_at"],
            "total_messages": len(conversation["chat_history"])
        })
        
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error in get_conversation: {type(e).__name__} - {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to retrieve conversation: {str(e)}"
        )

# API 4: Get all conversations
@app.get("/api/conversations")
async def get_all_conversations():
    """
    API 4: Get list of all conversations
    """
    global conversations_storage
    
    try:
        conversations_list = []
        for conv_id, conv_data in conversations_storage.items():
            conversations_list.append({
                "conversation_id": conv_id,
                "video_title": conv_data["video_metadata"].get("title", "Unknown"),
                "video_id": conv_data["video_id"],
                "created_at": conv_data["created_at"],
                "total_messages": len(conv_data["chat_history"]),
                "last_message": conv_data["chat_history"][-1][0] if conv_data["chat_history"] else None
            })
        
        return JSONResponse(content={
            "status": "success",
            "conversations": conversations_list,
            "total_conversations": len(conversations_list)
        })
        
    except Exception as e:
        print(f"Error in get_all_conversations: {type(e).__name__} - {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to retrieve conversations: {str(e)}"
        )

# API 5: Clear conversation history
@app.delete("/api/conversation/{conversation_id}")
async def clear_conversation(conversation_id: str):
    """
    API 5: Clear/delete a specific conversation
    """
    global conversations_storage, current_conversation_id
    
    try:
        if conversation_id not in conversations_storage:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        
        # Remove conversation from storage
        del conversations_storage[conversation_id]
        
        # If this was the current conversation, clear it
        if current_conversation_id == conversation_id:
            current_conversation_id = None
        
        return JSONResponse(content={
            "status": "success",
            "message": f"Conversation {conversation_id} deleted successfully"
        })
        
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error in clear_conversation: {type(e).__name__} - {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to delete conversation: {str(e)}"
        )
@app.get("/api/transcript")
async def get_transcript():
    """
    Utility endpoint to retrieve the current transcript
    """
    transcript_filename = "transcript.txt"
    
    try:
        if not os.path.exists(transcript_filename):
            raise HTTPException(status_code=404, detail="No transcript found. Please initialize a video first.")

        with open(transcript_filename, 'r', encoding='utf-8') as file:
            transcript_content = file.read()
        
        return JSONResponse(content={
            "status": "success", 
            "transcript": transcript_content
        })

    except HTTPException as e: 
        raise e
    except Exception as e:
        print(f"Error in get_transcript: {type(e).__name__} - {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to read transcript: {str(e)}"
        )

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """
    Health check endpoint
    """
    return JSONResponse(content={
        "status": "healthy",
        "message": "YouTube Video Chat API is running",
        "video_initialized": qna_chain is not None
    })

# --- Static files and SPA catch-all (should be last) ---
app.mount("/", StaticFiles(directory="dist", html=True), name="static_dist")

@app.get("/{full_path:path}")
async def serve_spa_catch_all_explicit(full_path: str):
    return FileResponse(os.path.join("dist", "index.html"))