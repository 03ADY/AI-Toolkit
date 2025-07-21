import uvicorn
import logging
from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import pipeline
from PIL import Image
import io
import soundfile as sf # For handling audio files (WAV)
import numpy as np # For numerical operations, especially with audio data
from scipy.io import wavfile # For writing WAV files to BytesIO
import wave # Fallback for WAV writing

import torch # For SpeechT5 speaker embeddings and general PyTorch operations
from datasets import load_dataset # For SpeechT5 speaker embeddings
import librosa # For audio resampling in STT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(
    title="Professional AI Toolkit (Python 3.10+)", # Adjusted Python version for broader compatibility
    description="A suite of high-performance, self-hosted AI models including TTS/STT.",
    version="1.4.5", # Updated version to reflect real QA model integration
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for broader compatibility with Streamlit
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models = {}
MODELS_LOADED = False

@app.on_event("startup")
def load_models():
    """
    Loads all AI models at application startup.
    This ensures models are ready for inference and avoids reloading.
    """
    global MODELS_LOADED
    logging.info("Starting to load AI models...")
    try:
        # Existing models
        models["sentiment"] = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        logging.info("Sentiment Analysis model loaded.")

        models["summarizer"] = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        logging.info("Text Summarization model loaded.")

        models["translator"] = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
        logging.info("Language Translation model loaded (EN-FR).")

        models["captioner"] = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
        logging.info("Image Captioning model loaded.")

        # Updated: Text Generation model for Chatbot functionality
        models["generator"] = pipeline("text-generation", model="microsoft/DialoGPT-medium")
        logging.info("Text Generation (DialoGPT-medium) model loaded for Chatbot.")

        # Updated: Text-to-Speech model to microsoft/speecht5_tts
        models["tts"] = pipeline("text-to-speech", model="microsoft/speecht5_tts")
        logging.info("Text-to-Speech model (microsoft/speecht5_tts) loaded.")

        # Load a default speaker embedding for SpeechT5.
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        models["tts_speaker_embedding"] = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
        logging.info("Default SpeechT5 speaker embedding loaded.")

        # New: Speech-to-Text (Automatic Speech Recognition) model
        models["stt"] = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
        logging.info("Speech-to-Text model loaded.")

        # New: Question Answering model
        # Using a distilled BERT model fine-tuned on SQuAD for extractive QA.
        models["qa_model"] = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
        logging.info("Question Answering model loaded.")

        MODELS_LOADED = True
        logging.info("All models loaded successfully.")

    except Exception as e:
        logging.error(f"Fatal error during model loading: {e}", exc_info=True)
        # In a production environment, you might want to exit or disable services
        # if critical models fail to load.

class TextIn(BaseModel):
    """
    Pydantic model for text input, now with an optional chat_history field
    to support conversational AI models.
    """
    text: str
    chat_history: list[str] = [] # Optional: For sending conversational context to backend

class TranslationOut(BaseModel):
    """Pydantic model for translation output."""
    original_text: str
    translated_text: str

class TranscriptionOut(BaseModel):
    """Pydantic model for transcription output."""
    transcribed_text: str

class QuestionAnsweringIn(BaseModel):
    """Pydantic model for Question Answering input."""
    question: str
    context: str = "" # Context is optional, but required for extractive QA models

class QuestionAnsweringOut(BaseModel):
    """Pydantic model for Question Answering output."""
    answer: str
    confidence: float = 0.0
    sources: list[str] = []

api_router = APIRouter()

@api_router.get("/status")
def get_status():
    """
    Returns the current loading status of the AI models.
    """
    return {"models_loaded": MODELS_LOADED}

@api_router.post("/sentiment/analyze")
def analyze_sentiment(payload: TextIn):
    """
    Analyzes the sentiment of a given text.
    """
    try:
        if not MODELS_LOADED:
            raise HTTPException(status_code=503, detail="Models are not loaded yet. Please wait.")
        result = models["sentiment"](payload.text)
        return result[0]
    except Exception as e:
        logging.error(f"Error in sentiment analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing sentiment analysis.")

@api_router.post("/summarization/summarize")
def summarize_text(payload: TextIn):
    """
    Generates a concise summary of a long piece of text.
    """
    try:
        if not MODELS_LOADED:
            raise HTTPException(status_code=503, detail="Models are not loaded yet. Please wait.")
        summary = models["summarizer"](payload.text, max_length=150, min_length=30, do_sample=False)
        return summary[0]
    except Exception as e:
        logging.error(f"Error in text summarization: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing summarization.")

@api_router.post("/translation/translate", response_model=TranslationOut)
def translate_text(payload: TextIn):
    """
    Translates text from English to French.
    Note: Current model only supports EN-FR.
    """
    try:
        if not MODELS_LOADED:
            raise HTTPException(status_code=503, detail="Models are not loaded yet. Please wait.")
        translation = models["translator"](payload.text)
        return TranslationOut(original_text=payload.text, translated_text=translation[0]['translation_text'])
    except Exception as e:
        logging.error(f"Error in language translation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing translation.")

@api_router.post("/image/caption")
async def caption_image(file: UploadFile = File(...)):
    """
    Generates a descriptive caption for an uploaded image.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    try:
        if not MODELS_LOADED:
            raise HTTPException(status_code=503, detail="Models are not loaded yet. Please wait.")
        image = Image.open(io.BytesIO(await file.read()))
        caption = models["captioner"](image)
        return caption[0]
    except Exception as e:
        logging.error(f"Error in image captioning: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing image captioning.")

@api_router.post("/generation/generate")
def generate_text(payload: TextIn):
    """
    Generates text based on a given prompt or continues a conversation for the chatbot.
    For chatbot use, `payload.text` should contain the concatenated conversational context.
    """
    try:
        if not MODELS_LOADED:
            raise HTTPException(status_code=503, detail="Models are not loaded yet. Please wait.")
        if not payload.text.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

        # For conversational models like DialoGPT, the input usually consists of
        # concatenated turns. The 'text-generation' pipeline will try to complete this.
        # The frontend should construct `payload.text` to contain the full conversation context.
        
        # DialoGPT specific generation parameters
        generated = models["generator"](
            payload.text,
            max_length=150, # Increased max_length for longer responses
            num_return_sequences=1,
            pad_token_id=models["generator"].tokenizer.eos_token_id, # Essential for DialoGPT
            do_sample=True, # Allow sampling for more diverse responses
            top_k=50,       # Top-K sampling
            top_p=0.95,     # Nucleus sampling
            temperature=0.7 # Controls randomness (0.0 for deterministic, higher for more creative)
        )
        
        ai_response = generated[0]["generated_text"]

        # Post-processing for DialoGPT: The model might regenerate the input prompt.
        # We need to extract only the new response.
        if ai_response.startswith(payload.text):
            ai_response = ai_response[len(payload.text):].strip()
            # DialoGPT often generates its own End-Of-Text token, split by it.
            ai_response = ai_response.split(models["generator"].tokenizer.eos_token)[0].strip()

        # Ensure response is not empty after trimming
        if not ai_response:
            ai_response = "I'm sorry, I couldn't generate a meaningful response based on that input."
        
        return {"generated_text": ai_response}
    except Exception as e:
        logging.error(f"Error in text generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing text generation.")

@api_router.post("/tts")
async def text_to_speech(payload: TextIn):
    """
    Converts text to speech and returns an audio file.
    Uses microsoft/speecht5_tts with a default speaker embedding.
    """
    try:
        if not MODELS_LOADED:
            raise HTTPException(status_code=503, detail="Models are not loaded yet. Please wait.")
        if not payload.text.strip():
            raise HTTPException(status_code=400, detail="Text for speech cannot be empty.")

        # Generate audio using TTS model with speaker embedding
        tts_output = models["tts"](
            payload.text,
            forward_params={"speaker_embeddings": models["tts_speaker_embedding"]}
        )
        audio_array = tts_output["audio"]
        sampling_rate = tts_output["sampling_rate"]

        # Create BytesIO buffer
        buffer = io.BytesIO()

        try:
            # Method 1: Using scipy.io.wavfile (most reliable for numpy arrays)
            # Ensure audio is in int16 format for broad WAV compatibility
            # scipy.io.wavfile.write handles float32 arrays normalized to [-1, 1] too,
            # but int16 is often preferred for PCM WAVs.
            if audio_array.dtype != np.int16:
                if np.max(np.abs(audio_array)) > 1.0: # Normalize if values exceed [-1, 1]
                    audio_array = audio_array / np.max(np.abs(audio_array))
                audio_int16 = (audio_array * 32767).astype(np.int16)
            else:
                audio_int16 = audio_array

            wavfile.write(buffer, sampling_rate, audio_int16)
            buffer.seek(0) # Rewind the buffer

        except Exception as scipy_error:
            # Method 2: Fallback using built-in wave module if scipy fails
            logging.warning(f"scipy.io.wavfile failed, falling back to wave module: {scipy_error}", exc_info=True)
            buffer = io.BytesIO()  # Reset buffer if error occurred

            # Ensure audio is in int16 format
            if audio_array.dtype != np.int16:
                if np.max(np.abs(audio_array)) > 1.0:
                    audio_array = audio_array / np.max(np.abs(audio_array))
                audio_int16 = (audio_array * 32767).astype(np.int16)
            else:
                audio_int16 = audio_array

            with wave.open(buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
                wav_file.setframerate(sampling_rate)
                wav_file.writeframes(audio_int16.tobytes())

            buffer.seek(0) # Rewind the buffer

        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"}
        )

    except Exception as e:
        logging.error(f"Error in Text-to-Speech: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing Text-to-Speech.")

@api_router.post("/stt", response_model=TranscriptionOut)
async def speech_to_text(file: UploadFile = File(...)):
    """
    Transcribes an audio file to text.
    Includes resampling to 16kHz for Whisper model compatibility.
    """
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file.")
    try:
        if not MODELS_LOADED:
            raise HTTPException(status_code=503, detail="Models are not loaded yet. Please wait.")
        audio_bytes = await file.read()
        buffer = io.BytesIO(audio_bytes)

        # Load audio data using soundfile
        audio_data, current_sampling_rate = sf.read(buffer)

        # If audio_data is stereo, convert to mono by averaging channels
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Resample audio to 16kHz if necessary, as Whisper models typically expect this rate
        target_sampling_rate = 16000
        if current_sampling_rate != target_sampling_rate:
            logging.info(f"Resampling audio from {current_sampling_rate}Hz to {target_sampling_rate}Hz for STT.")
            audio_data = librosa.resample(audio_data, orig_sr=current_sampling_rate, target_sr=target_sampling_rate)
            current_sampling_rate = target_sampling_rate # Update sampling rate after resampling

        # Prepare input for STT model
        stt_input = {"sampling_rate": current_sampling_rate, "raw": audio_data}
        transcription = models["stt"](stt_input)

        return TranscriptionOut(transcribed_text=transcription["text"])

    except Exception as e:
        logging.error(f"Error in Speech-to-Text: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing Speech-to-Text.")

@api_router.post("/qa/answer", response_model=QuestionAnsweringOut)
def answer_question(payload: QuestionAnsweringIn):
    """
    Provides an answer to a question based on a provided context.
    Now uses a real Question Answering model. Context is required for meaningful answers.
    """
    try:
        if not MODELS_LOADED:
            raise HTTPException(status_code=503, detail="Models are not loaded yet. Please wait.")

        # For extractive QA models, context is crucial.
        if not payload.context or not payload.context.strip():
            return QuestionAnsweringOut(
                answer="Please provide a context to answer the question. This model requires text to find an answer within.",
                confidence=0.0,
                sources=[]
            )

        # Use the loaded Question Answering model
        qa_output = models["qa_model"](question=payload.question, context=payload.context)
        
        answer = qa_output['answer']
        confidence = qa_output['score']
        
        # Extractive QA models typically don't provide "sources" beyond the given context,
        # but you can indicate the context itself as a source if desired.
        sources = ["Provided Context"] 

        return QuestionAnsweringOut(
            answer=answer,
            confidence=confidence,
            sources=sources
        )
    except Exception as e:
        logging.error(f"Error in Question Answering: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing Question Answering.")

app.include_router(api_router, prefix="/api")

@app.get("/")
def root():
    """
    Root endpoint for the API.
    """
    return {"message": "Welcome to the Professional AI Toolkit API. Visit /docs for details."}

if __name__ == "__main__":
    print("Starting Professional AI Toolkit Backend Server...")
    print("API documentation will be available at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
