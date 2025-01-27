import os
import uuid
import time
import logging
import threading
import speech_recognition as sr
import os
import uuid
from fastapi import UploadFile
from fastapi.responses import JSONResponse
import subprocess
import logging
from datetime import datetime
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import base64
import requests
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import speech_recognition as sr
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configuration
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 1000))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 200))
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-3.5-turbo')

# Default system message
DEFAULT_SYSTEM_MESSAGE = """You are a helpful AI assistant with access to knowledge about UBIK solutions and subsidaries. 
Answer questions based on the provided context. you will answer the question that user will ask. If you don't know something or if it's not in the context, 
say that you aren't trained for it but in a proper manner. you can respond to greetings or general greet things, Also, don't go out of context. Your name is UBIK AI. Answer mostly under 50 words unless very much required."""

# Logging configuration
logging.basicConfig(
    filename='app_logs.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Thread pool for concurrent tasks
executor = ThreadPoolExecutor(max_workers=3)

# Global vectorstore for document embeddings
global_vectorstore = None
vectorstore_lock = threading.Lock()

# User sessions to store conversation chains and memory
user_sessions = {}

# Initialize FastAPI app
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Utility functions
def log_event(event_type: str, details: str = "", user_id: str = "Unknown"):
    """Log events to a file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"{timestamp} | {event_type} | UserID: {user_id} | {details}\n"
    with open("app_logs.txt", "a", encoding="utf-8") as f:
        f.write(log_line)

def get_user_state(user_id: str):
    """Get or create a user session."""
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            'memory': ConversationBufferMemory(
                memory_key='chat_history',
                return_messages=True
            ),
            'chain': None,
            'history': [],
            'system_message': DEFAULT_SYSTEM_MESSAGE
        }
        log_event("UserJoined", "New user state created.", user_id=user_id)
    return user_sessions[user_id]

@lru_cache(maxsize=128)
def get_pdf_text(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        logger.error(f"Error reading {pdf_path}: {e}")
        return ""

def get_text_chunks(text: str):
    """Split text into chunks for processing."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    return text_splitter.split_text(text)

def initialize_global_vectorstore():
    """Initialize the global vectorstore with PDF data."""
    global global_vectorstore
    with vectorstore_lock:
        if global_vectorstore is not None:
            return True, "[SYSTEM MESSAGE] Vectorstore is already initialized."

        pdf_paths = [
            os.path.join('data', "Ilesh Sir (IK) - Words.pdf"),
            os.path.join('data', "UBIK SOLUTION.pdf"),
            os.path.join('data', "illesh3.pdf"),
            os.path.join('data', "website-data-ik.pdf")
        ]

        combined_text = ""
        for path in pdf_paths:
            combined_text += get_pdf_text(path) + " "

        if not combined_text.strip():
            return False, "No text could be extracted from the PDFs."

        text_chunks = get_text_chunks(combined_text)
        
        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        global_vectorstore = FAISS.from_texts(
            texts=text_chunks,
            embedding=embeddings
        )
        logger.info("Vectorstore has been created with OpenAI embeddings.")
    return True, "[SYSTEM MESSAGE] Vectorstore was created successfully."

def handle_userinput(user_question: str, user_id: str):
    """Handle user input and generate a response."""
    user_state = get_user_state(user_id)
    conversation_chain = user_state['chain']
    if not conversation_chain:
        return None

    # Step 1: Define the refinement template
    def refine_input(input_text):
        refinement_prompt = f"""
        Based on the following user input: "{input_text}"
        
        1. If it's a basic greeting:
            - Respond: "Hello! How can I help you with information about UBIK Solutions?"
        
        2. For questions:
            - Process them directly using available context
            - Keep responses focused and concise
        
        Maintain a professional tone and avoid unnecessary elaboration.
        
        Refined query:
        """
        return conversation_chain({'question': refinement_prompt})['answer'].strip()

    # Step 2: Generate the answer using the refined input
    def generate_answer(refined_input):
        answer_prompt = f"""
        User's Refined Input: "{refined_input}"

        Context: Respond to the user's query based on the refined input in a clear, concise, and contextually relevant manner.

        Answer:
        """
        return conversation_chain({'question': answer_prompt})['answer'].strip()

    # Step 3: Refine the user input
    refined_question = refine_input(user_question)
    log_event("RefinedInput", f"Refined Question: {refined_question}", user_id=user_id)

    # Step 4: Generate the final answer
    answer = generate_answer(refined_question)

    # Step 5: Log events and update history
    user_state['history'].append((user_question, answer))
    log_event("PromptSentToGPT", f"Original Prompt: {user_question}, Refined: {refined_question}", user_id=user_id)
    log_event("UserQuestion", f"Q: {user_question}", user_id=user_id)
    log_event("AIAnswer", f"A: {answer}", user_id=user_id)

    return {'text': answer}

def create_or_refresh_user_chain(user_id: str):
    """Create or refresh a conversation chain for a user."""
    user_state = get_user_state(user_id)
    if user_state['chain'] is None:
        if global_vectorstore is None:
            return False, "Global vectorstore is not initialized."

        # Create chat model with system message
        chat_llm = ChatOpenAI(model=MODEL_NAME, temperature=0.9)

        # Create custom QA prompt template that includes system message
        condense_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:"""
        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_template)

        qa_template = f"""
        {user_state['system_message']}

        Context: {{context}}
        
        Question: {{question}}
        
        Answer: """
        
        QA_PROMPT = PromptTemplate(
            template=qa_template,
            input_variables=["context", "question"]
        )

        # Create the chain
        user_state['chain'] = ConversationalRetrievalChain.from_llm(
            llm=chat_llm,
            retriever=global_vectorstore.as_retriever(),
            memory=user_state['memory'],
            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
            combine_docs_chain_kwargs={'prompt': QA_PROMPT}
        )
        
        logger.info(f"New conversation chain created for user {user_id} with system message: {user_state['system_message']}")
        return True, "Conversation chain created."
    else:
        return True, "Conversation chain already exists."

# FastAPI endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize the global vectorstore on startup."""
    success, message = initialize_global_vectorstore()
    if not success:
        logger.error(f"Failed to initialize vectorstore: {message}")
    else:
        logger.info("Vectorstore initialized successfully on startup")

@app.get("/")
async def hello_root():
    """Root endpoint."""
    return {
        "message": "Hello from FastAPI server. Use the endpoints to process data or ask questions.",
        "status": "operational"
    }

@app.post("/refresh_chain")
async def refresh_chain(request: Request):
    """Refresh the conversation chain for a user."""
    data = await request.json()
    if "user_id" not in data:
        return JSONResponse(
            status_code=400,
            content={"error": "Missing user_id."}
        )

    user_id = data["user_id"]
    success, message = create_or_refresh_user_chain(user_id)
    status = 'success' if success else 'error'
    return {"status": status, "message": message}

@app.post("/ask")
async def ask_question(request: Request):
    """Handle user questions."""
    try:
        data = await request.json()
    except:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON payload."})

    if "user_id" not in data or "question" not in data:
        return JSONResponse(status_code=400, content={"error": "Missing user_id or question."})

    user_id = data["user_id"]
    user_question = data["question"]
    FORWARD_ENDPOINT = os.getenv("FORWARD_ENDPOINT", "https://5vf2s8w3-10000.aue.devtunnels.ms/receive")

    if user_id not in user_sessions or user_sessions[user_id]['chain'] is None:
        create_or_refresh_user_chain(user_id)

    answer = handle_userinput(user_question, user_id)
    if not answer:
        return {"status": "error", "message": "No conversation chain or unable to handle question."}

    # Forward the response to the endpoint
    try:
        # Format payload to match ReceiveText model
        payload = {
            "text": answer['text']  # Just send the text field as expected by ReceiveText
        }
        
        forward_response = requests.post(FORWARD_ENDPOINT, json=payload)
        forwarding_status = "success" if forward_response.status_code == 200 else "failed"
        
        if forward_response.status_code != 200:
            logger.error(f"Forward request failed with status {forward_response.status_code}: {forward_response.text}")
        else:
            logger.info(f"Response forwarding {forwarding_status}")
            
    except Exception as e:
        logger.error(f"Error forwarding response: {e}")
        forwarding_status = "failed"

    prompt_sent = {'question': user_question}
    return {
        "status": "success", 
        "data": answer, 
        "prompt": prompt_sent,
        "forwarding_status": forwarding_status
    }

@app.post("/set_system_message")
async def set_system_message(request: Request):
    """Set a custom system message for a user."""
    try:
        data = await request.json()
    except:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON."})

    user_id = data.get("user_id")
    system_message = data.get("system_message")

    if not user_id or not system_message:
        return JSONResponse(
            status_code=400, 
            content={"error": "Missing user_id or system_message."}
        )

    user_state = get_user_state(user_id)
    user_state['system_message'] = system_message
    # Force recreation of chain with new system message
    user_state['chain'] = None
    
    success, message = create_or_refresh_user_chain(user_id)
    
    log_event(
        "SystemMessageUpdated",
        f"System message updated to: {system_message}",
        user_id=user_id
    )
    
    return {
        "status": "success" if success else "error",
        "message": f"System message updated. {message}"
    }

@app.get("/get_system_message")
async def get_system_message(user_id: str):
    """Get the system message for a user."""
    if not user_id:
        return JSONResponse(
            status_code=400,
            content={"error": "Missing user_id."}
        )

    user_state = get_user_state(user_id)
    return {
        "status": "success",
        "system_message": user_state['system_message']
    }

@app.post("/clear_history")
async def clear_history(request: Request):
    """Clear the chat history for a user."""
    try:
        data = await request.json()
    except:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON."})

    user_id = data.get("user_id", None)
    if not user_id:
        return JSONResponse(status_code=400, content={"error": "Missing user_id."})

    if user_id in user_sessions:
        user_state = user_sessions[user_id]
        user_state['memory'].clear()
        user_state['history'].clear()
        user_state['chain'] = None
        log_event("HistoryCleared", "User cleared their chat history.", user_id=user_id)
        return {"status": "success", "message": "Chat history cleared."}
    else:
        return {"status": "error", "message": "No user session found to clear."}

@app.post("/speech_to_text")
async def speech_to_text_endpoint(file: UploadFile = File(...)):
    """Convert speech to text."""
    return await handle_speech_to_text(file)
async def handle_speech_to_text(file: UploadFile):
    logger.info(f"Starting speech to text conversion for file: {file.filename}")
    
    if file is None:
        logger.error("No file provided")
        return JSONResponse(
            status_code=400,
            content={"error": "No file provided."}
        )

    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Generate temporary file paths
    webm_path = os.path.join("data", f"temp_{uuid.uuid4().hex}.webm")
    wav_path = os.path.join("data", f"temp_{uuid.uuid4().hex}.wav")
    
    try:
        # Save the uploaded file
        logger.debug("Reading uploaded file")
        content = await file.read()
        with open(webm_path, "wb") as f:
            f.write(content)
        logger.debug(f"Saved WebM file to {webm_path}")
        
        # Convert to WAV format
        logger.debug("Converting WebM to WAV")
        try:
            subprocess.run([
                'ffmpeg', '-i', webm_path, 
                '-acodec', 'pcm_s16le', 
                '-ar', '16000', 
                '-ac', '1', 
                wav_path
            ], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg conversion failed: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Audio conversion failed: {e.stderr}"}
            )

        # Initialize speech recognizer
        r = sr.Recognizer()
        
        # Process the audio file
        with sr.AudioFile(wav_path) as source:
            audio = r.record(source)  # Read the entire audio file
            logger.debug("Processing audio with Google Speech Recognition...")
            
            try:
                text = r.recognize_google(audio)
                logger.info("Successfully transcribed audio")
                return {"status": "success", "text": text}
                
            except sr.UnknownValueError:
                logger.error("Google Speech Recognition could not understand audio")
                return JSONResponse(
                    status_code=400,
                    content={"error": "Could not understand audio"}
                )
            except sr.RequestError as e:
                logger.error(f"Could not request results from Google SR service; {e}")
                return JSONResponse(
                    status_code=503,
                    content={"error": f"Speech service error: {str(e)}"}
                )

    except Exception as e:
        logger.exception("Unexpected error occurred")
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred during transcription: {str(e)}"}
        )
    
    finally:
        # Clean up temporary files
        logger.debug("Cleaning up temporary files")
        for path in [webm_path, wav_path]:
            if os.path.exists(path):
                os.remove(path)
                logger.debug(f"Removed {path}")
@app.post("/logout")
async def logout(request: Request):
    """Log out a user by clearing their session."""
    try:
        data = await request.json()
    except:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON."})

    user_id = data.get("user_id", None)
    if not user_id:
        return JSONResponse(status_code=400, content={"error": "Missing user_id."})

    if user_id in user_sessions:
        del user_sessions[user_id]
    return {"status": "success", "message": "User session cleared."}

# Text-to-speech endpoint
GOOGLE_TTS_API_KEY = os.getenv("GOOGLE_TTS_API_KEY")  # Load your API key from .env or environment variable

@app.post("/text_to_speech")
async def text_to_speech_api(request: Request):
    """Convert text to speech using Google Cloud TTS with API key authorization."""
    try:
        data = await request.json()
        text = data.get("text", "")
        voice_name = data.get("voice", "en-US-Wavenet-D")  # Default to a US English voice
        language_code = data.get("language_code", "en-US")
        speaking_rate = data.get("speaking_rate", 1.0)  # Normal speed

        if not text:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing 'text' in request."}
            )

        # Google TTS API endpoint with API key
        url = "https://texttospeech.googleapis.com/v1/text:synthesize?key=" + GOOGLE_TTS_API_KEY

        # Construct the payload
        payload = {
            "input": {"text": text},
            "voice": {
                "languageCode": language_code,
                "name": voice_name,
            },
            "audioConfig": {
                "audioEncoding": "MP3",
                "speakingRate": speaking_rate,
            },
        }

        # Make request to the Google TTS API
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            audio_content = response.json().get("audioContent", None)
            if not audio_content:
                return JSONResponse(
                    status_code=500,
                    content={"error": "No audio content received from TTS API."}
                )

            # Decode the base64 audio content and save it
            audio_path = os.path.join("data", f"tts_output_{uuid.uuid4().hex}.mp3")
            with open(audio_path, "wb") as audio_file:
                audio_file.write(base64.b64decode(audio_content))

            # Return path or URL to the generated audio
            return {"status": "success", "audio_url": audio_path}
        else:
            # If there's an error from Google
            return JSONResponse(
                status_code=response.status_code,
                content={"error": response.text}
            )

    except Exception as e:
        logger.error(f"Error in text-to-speech: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "An internal error occurred during text-to-speech synthesis."}
        )