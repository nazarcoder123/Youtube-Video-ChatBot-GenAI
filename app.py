from flask import Flask, request, jsonify, send_from_directory
# from flask_session import Session
from flask_cors import CORS  # <-- New import here
from flask_cors import cross_origin
import openai
import os
from pytube import YouTube
import re
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv
import yt_dlp  # Alternative to pytube

load_dotenv()

app = Flask(__name__, static_folder="./dist") 
CORS(app, resources={r"/*": {"origins": "*"}}) 
openai.api_key = os.environ["GOOGLE_API_KEY"]
llm_name = "gemini-2.0-flash"
qna_chain = None


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

def load_db(file, chain_type, k):
    """
    Central Function that:
        - Loads the database
        - Creates the retriever
        - Creates the chatbot chain
        - Returns the chatbot chain
        - A Dictionary containing 
                -- question
                -- llm answer
                -- chat history
                -- source_documents
                -- generated_question
                s
    Usage: question_answer_chain = load_db(file, chain_type, k) 
           response = question_answer_chain({"question": query, "chat_history": chat_history}})
    """

    transcript = TextLoader(file).load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=70)
    docs = text_splitter.split_documents(transcript)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    db = Chroma.from_documents(docs, embeddings)
    
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})

    # create a chatbot chain. Memory is managed externally.
    qa = ConversationalRetrievalChain.from_llm(
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7, convert_system_message_to_human=True),
        chain_type=chain_type,                               
        retriever=retriever, 
        return_source_documents=True,
        return_generated_question=True,
        # memory=memory
    )
    
    return qa 


def buffer(history, buff):
    """
    Buffer the history.
    Keeps only buff recent chats in the history

    Usage: history = buffer(history, buff)
    """

    if len(history) > buff :
        print(len(history)>buff)
        return history[-buff:]
    return history
    

def is_valid_yt(link):
    """
    Check if a link is a valid YouTube link.
    
    Usage: boolean, video_id = is_valid_yt(youtube_string)
    """

    pattern = r'^(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/watch\?v=|youtu\.be\/)([\w\-_]{11})(?:\S+)?$'
    match = re.match(pattern, link)
    if match:
        return True, match.group(1) 
    else:
        return False, None


def get_metadata_pytube(video_id) -> dict:
    """Get video metadata using pytube with error handling."""
    try:
        yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
        video_info = {
            "title": yt.title or "Unknown",
            "description": yt.description or "Unknown",
            "view_count": yt.views or 0,
            "thumbnail_url": yt.thumbnail_url or "Unknown",
            "publish_date": yt.publish_date.strftime("%Y-%m-%d %H:%M:%S")
            if yt.publish_date
            else "Unknown",
            "length": yt.length or 0,
            "author": yt.author or "Unknown",
        }
        return video_info
    except Exception as e:
        print(f"Pytube failed: {e}")
        return None


def get_metadata_ytdlp(video_id) -> dict:
    """Get video metadata using yt-dlp as fallback."""
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
            
            video_info = {
                "title": info.get('title', 'Unknown'),
                "description": info.get('description', 'Unknown'),
                "view_count": info.get('view_count', 0),
                "thumbnail_url": info.get('thumbnail', 'Unknown'),
                "publish_date": info.get('upload_date', 'Unknown'),
                "length": info.get('duration', 0),
                "author": info.get('uploader', 'Unknown'),
            }
            return video_info
    except Exception as e:
        print(f"yt-dlp failed: {e}")
        return None


def get_metadata(video_id) -> dict:
    """Get video metadata with fallback methods."""
    
    # Try pytube first
    metadata = get_metadata_pytube(video_id)
    if metadata:
        return metadata
    
    # Fallback to yt-dlp
    print("Pytube failed, trying yt-dlp...")
    metadata = get_metadata_ytdlp(video_id)
    if metadata:
        return metadata
    
    # If both fail, return basic metadata
    print("Both methods failed, returning basic metadata...")
    return {
        "title": "Unknown",
        "description": "Unable to fetch video description",
        "view_count": 0,
        "thumbnail_url": f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
        "publish_date": "Unknown",
        "length": 0,
        "author": "Unknown",
    }


def save_transcript(video_id):
    """
    Saves the transcript of a valid yt video to a text file.
    """

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
    except Exception as e:
        print(f"Error fetching transcript for video {video_id}: {e}")
        return None
    if transcript:
        with open('transcript.txt', 'w', encoding='utf-8') as file:
            for entry in transcript:
                file.write(f"~{int(entry['start'])}~{entry['text']} ")
        print(f"Transcript saved to: transcript.txt")
        return True
    return False

@app.route('/init', methods=['POST'])
@cross_origin()
def initialize():
    """
    Initialize the qna_chain for a user.
    """
    global qna_chain
    
    qna_chain = 0

    try:
        # NEED to authenticate the user here
        yt_link = request.json.get('yt_link', '')
        valid, id = is_valid_yt(yt_link)
        
        if not valid:
            return jsonify({"status": "error", "message": "Invalid YouTube link."}), 400
        
        # Get metadata with error handling
        metadata = get_metadata(id)
        
        # Clean up previous transcript
        try:
            os.remove('./transcript.txt')
        except:
            print("No transcript file to remove.")
        
        # Save transcript
        transcript_saved = save_transcript(id)
        if not transcript_saved:
            return jsonify({"status": "error", "message": "Could not fetch video transcript."}), 400

        # Initialize qna_chain for the user
        qna_chain = load_db("./transcript.txt", 'stuff', 5)

        return jsonify({"status": "success", 
                        "message": "qna_chain initialized.",
                        "metadata": metadata,
                        })
                        
    except Exception as e:
        print(f"Error in initialize: {e}")
        return jsonify({"status": "error", "message": f"Initialization failed: {str(e)}"}), 500


@app.route('/response', methods=['POST'])
def response():
    """
    - Expects youtube Video Link and chat-history in payload
    - Returns response on the query.
    """
    global qna_chain
    
    try:
        req = request.get_json()
        raw = req.get('chat_history', [])

        # raw is a list of list containing two strings convert that into a list of tuples
        if len(raw) > 0:
            chat_history = [tuple(x) for x in raw]
        else:
            chat_history = []
        
        memory = chat_history
        query = req.get('query', '')

        if memory is None:
            memory = []
        
        if qna_chain is None:
            return jsonify({"status": "error", "message": "qna_chain not initialized."}), 400

        response = qna_chain({'question': query, 'chat_history': buffer(memory, 7)})

        if response['source_documents']:
            pattern = r'~(\d+)~'
            backlinked_docs = [response['source_documents'][i].page_content for i in range(len(response['source_documents']))]
            timestamps = list(map(lambda s: int(re.search(pattern, s).group(1)) if re.search(pattern, s) else None, backlinked_docs))
            
            return jsonify(dict(timestamps=timestamps, answer=response['answer']))

        return jsonify({"answer": response['answer']})
        
    except Exception as e:
        print(f"Error in response: {e}")
        return jsonify({"status": "error", "message": f"Response generation failed: {str(e)}"}), 500

@app.route('/transcript', methods=['POST'])
@cross_origin()
def send_transcript():
    """
    Send the transcript of the video.
    """
    try:
        with open('transcript.txt', 'r', encoding='utf-8') as file:
            transcript = file.read()
        return jsonify({"status": "success", "transcript": transcript})
    except Exception as e:
        print(f"Error reading transcript: {e}")
        return jsonify({"status": "error", "message": "Transcript not found."}), 404
    

if __name__ == '__main__':
    app.run(debug=True)