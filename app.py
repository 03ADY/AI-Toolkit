import os
import streamlit as st
import requests
import io
import time
import json
import base64
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any
import hashlib
import zipfile
from PIL import Image
import numpy as np

# --- Configuration ---
from toolkit.cloud import is_streamlit_cloud, use_builtin_runtime
from toolkit.config import API_BASE, APP_NAME

if use_builtin_runtime():
    st.session_state.builtin_mode = True
    st.session_state.offline_demo = True
elif "offline_demo" not in st.session_state and is_streamlit_cloud():
    st.session_state.offline_demo = True
from toolkit.api_client import check_status as _check_status
from toolkit.ui import render_demo_sidebar, render_enterprise_home, render_enterprise_dashboard
from toolkit.extras import (
    render_api_playground,
    render_batch_lab,
    render_cost_center,
    render_demo_runner,
    render_document_analyzer,
    render_integrations_hub,
    render_service_compare,
    render_session_report,
)
from toolkit.helpers import call_api
from toolkit.theme_ui import hero_html, inject_theme

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_theme()

# --- Session State Initialization ---
# Initialize all session state variables to ensure persistence across reruns
if 'history' not in st.session_state:
    st.session_state.history = []
if 'favorites' not in st.session_state:
    st.session_state.favorites = []
if 'api_calls_count' not in st.session_state:
    st.session_state.api_calls_count = 0
if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {
        'theme': 'Light',
        'default_voice': 'Female',
        'default_language': 'English',
        'auto_save': True
    }
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'translation_history' not in st.session_state:
    st.session_state.translation_history = []

# --- Helper Functions ---
@st.cache_data(ttl=3)
def get_backend_status():
    """Checks the backend status by pinging the /status endpoint."""
    loaded, err, _ = _check_status()
    return loaded, err

def log_to_history(service: str, input_data: str, output_data: str, success: bool = True, latency_ms: float | None = None):
    """
    Logs API calls to the session history.
    Input and output are truncated for display in the history table.
    """
    entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'service': service,
        'input': input_data[:100] + "..." if len(input_data) > 100 else input_data,
        'output': output_data[:100] + "..." if len(output_data) > 100 else output_data,
        'success': success,
    }
    if latency_ms is not None:
        entry['latency_ms'] = round(latency_ms, 1)
    st.session_state.history.append(entry)
    st.session_state.api_calls_count += 1

def display_spinner_and_message(message):
    """Displays a spinner and a message for better UX during processing."""
    with st.spinner(message):
        time.sleep(0.5)  # Brief pause for UX to show spinner

def display_error(error_message):
    """Displays an error message in a styled box."""
    st.error(f"🚨 Error: {error_message}")

def display_success(message):
    """Displays a success message in a styled box."""
    st.success(f"✅ {message}")

def create_download_link(data, filename, text):
    """Creates an HTML download link for given data."""
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{text}</a>'

def add_to_favorites(item_type: str, content: dict):
    """Adds an item to the session favorites list."""
    st.session_state.favorites.append({
        'type': item_type,
        'content': content,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

# --- AI Service Components ---

def sentiment_analysis_component():
    """Streamlit component for Sentiment Analysis."""
    st.header("🎭 Sentiment Analysis")
    st.write("Determine the emotional tone of text with advanced analytics.")

    fill = st.session_state.pop("demo_fill_sentiment", "")
    text_input = st.text_area(
        "Enter text to analyze sentiment:",
        value=fill,
        height=150,
        placeholder="Enter your text here...",
        help="Type or paste any text to analyze its emotional sentiment",
    )
        
    if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
        if not text_input.strip():
            st.warning("Please enter text to analyze.")
            return

        with st.spinner("Analyzing sentiment..."):
            try:
                ok, result, err, latency_ms = call_api(
                    "/sentiment/analyze",
                    {"text": text_input},
                    offline=st.session_state.get("offline_demo", False),
                    timeout=30,
                )
                if not ok:
                    raise requests.exceptions.RequestException(err)
                
                st.subheader("📊 Analysis Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    sentiment_class = "sentiment-positive" if result["label"] == "POSITIVE" else "sentiment-negative"
                    st.markdown(f"""
                    <div class="{sentiment_class}">
                        <h3>{result['label']}</h3>
                        <p>{result['score']:.1%} Confidence</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.metric("Confidence Score", f"{result['score']:.1%}")
                with col3:
                    polarity = "High" if result['score'] > 0.7 else "Moderate" if result['score'] > 0.5 else "Low"
                    st.metric("Polarity Strength", polarity)
                
                # Visualization
                fig = go.Figure(data=[
                    go.Bar(name='Positive', x=['Sentiment'], y=[result['score']], marker_color='green'),
                    go.Bar(name='Negative', x=['Sentiment'], y=[1-result['score']], marker_color='red')
                ])
                fig.update_layout(
                    title="Sentiment Distribution",
                    barmode='stack',
                    height=400,
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("⭐ Save to Favorites", key="save_sentiment"):
                        add_to_favorites("Sentiment Analysis", {
                            'text': text_input,
                            'result': result
                        })
                        st.success("Saved to favorites!")
                
                log_to_history("Sentiment Analysis", text_input, str(result), latency_ms=latency_ms)
                
            except requests.exceptions.RequestException as e:
                display_error(f"Could not analyze sentiment: {e}")
                log_to_history("Sentiment Analysis", text_input, str(e), False)
            except Exception as e:
                display_error(f"An unexpected error occurred: {e}")
                log_to_history("Sentiment Analysis", text_input, str(e), False)

def text_summarization_component():
    """Streamlit component for Text Summarization."""
    st.header("📄 Text Summarization")
    st.write("Generate concise summaries of your text.")

    fill = st.session_state.pop("demo_fill_summarization", "")
    text_input = st.text_area(
        "Paste text to summarize:",
        value=fill,
        height=300,
        placeholder="Paste your long text here...",
        help="Enter a longer text document to get a concise summary"
    )
        
    if st.button("📝 Generate Summary", type="primary", use_container_width=True):
        if not text_input.strip():
            st.warning("Please provide text to summarize.")
            return

        with st.spinner("Generating summary..."):
            try:
                ok, result, err, latency_ms = call_api(
                    "/summarization/summarize",
                    {"text": text_input},
                    offline=st.session_state.get("offline_demo", False),
                    timeout=120,
                )
                if not ok:
                    raise requests.exceptions.RequestException(err)
                summary = result.get("summary_text", "No summary generated.")
                
                st.subheader("📝 Summary")
                st.markdown(f'<div class="content-display">{summary}</div>', unsafe_allow_html=True)
                
                st.subheader("📊 Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Original Length", f"{len(text_input)} chars")
                with col2:
                    st.metric("Summary Length", f"{len(summary)} chars")
                with col3:
                    compression_ratio = len(summary) / len(text_input) if text_input else 0
                    st.metric("Compression Ratio", f"{compression_ratio:.1%}")
                with col4:
                    reading_time = len(summary.split()) / 200  # Average reading speed
                    st.metric("Reading Time", f"{reading_time:.1f} min")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download Summary",
                        data=summary,
                        file_name="summary.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                with col2:
                    if st.button("⭐ Save to Favorites", key="save_summary"):
                        add_to_favorites("Text Summary", {
                            'original_text': text_input[:200] + "...",
                            'summary': summary,
                        })
                        st.success("Saved to favorites!")
                
                log_to_history("Text Summarization", text_input[:100], summary, latency_ms=latency_ms)
                
            except requests.exceptions.RequestException as e:
                display_error(f"Could not summarize text: {e}")
                log_to_history("Text Summarization", text_input[:100], str(e), False)
            except Exception as e:
                display_error(f"An unexpected error occurred: {e}")
                log_to_history("Text Summarization", text_input[:100], str(e), False)

def text_generation_component():
    """Streamlit component for Creative Text Generation."""
    st.header("✍️ Creative Text Generation")
    st.write("Generate creative content with AI assistance.")

    # Use a simple text area without complex session state management
    text_input = st.text_area(
        "Enter your prompt:", 
        height=150, 
        placeholder="Start your creative prompt here...",
        help="Provide a starting prompt for creative text generation"
    )
        
    if st.button("🚀 Generate Content", type="primary", use_container_width=True):
        if not text_input.strip():
            st.warning("Please enter a prompt.")
            return

        with st.spinner("Generating creative content..."):
            try:
                ok, result, err, latency_ms = call_api(
                    "/generation/generate",
                    {"text": text_input},
                    offline=st.session_state.get("offline_demo", False),
                    timeout=60,
                )
                if not ok:
                    raise requests.exceptions.RequestException(err)
                generated_text = result.get("generated_text", "No text generated.")
                
                st.subheader("📖 Generated Content")
                st.markdown(f'<div class="generated-content">{generated_text}</div>', unsafe_allow_html=True)
                
                st.subheader("📊 Content Analysis")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Words", len(generated_text.split()))
                with col2:
                    st.metric("Characters", len(generated_text))
                with col3:
                    sentences = generated_text.split('.')
                    st.metric("Sentences", len([s for s in sentences if s.strip()]))
                with col4:
                    avg_words = len(generated_text.split()) / len([s for s in sentences if s.strip()]) if sentences else 0
                    st.metric("Avg Words/Sentence", f"{avg_words:.1f}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download Text",
                        data=generated_text,
                        file_name="generated_content.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                with col2:
                    if st.button("⭐ Save to Favorites", key="save_generation"):
                        add_to_favorites("Generated Text", {
                            'prompt': text_input,
                            'content': generated_text
                        })
                        st.success("Saved to favorites!")
                
                log_to_history("Text Generation", text_input, generated_text, latency_ms=latency_ms)
                
            except requests.exceptions.RequestException as e:
                display_error(f"Could not generate text: {e}")
                log_to_history("Text Generation", text_input, str(e), False)
            except Exception as e:
                display_error(f"An unexpected error occurred: {e}")
                log_to_history("Text Generation", text_input, str(e), False)

def image_captioning_component():
    """Streamlit component for Image Captioning."""
    st.header("🖼️ Image Captioning")
    st.write("Generate descriptive captions for your images.")

    from toolkit.assets_util import demo_image_bytes

    col_u1, col_u2 = st.columns([3, 1])
    with col_u1:
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            help="Supported formats: JPG, JPEG, PNG, BMP, TIFF",
        )
    with col_u2:
        use_demo = st.button("Use demo image", use_container_width=True)
    if use_demo:
        st.session_state.demo_image_bytes = demo_image_bytes()
        st.session_state.demo_image_name = "demo_workshop.png"

    if st.session_state.get("demo_image_bytes") and uploaded_file is None:
        st.image(st.session_state.demo_image_bytes, caption="Demo workshop image", use_container_width=True)

    active_file = uploaded_file
    if active_file is None and st.session_state.get("demo_image_bytes"):
        class _DemoFile:
            def __init__(self, name, data):
                self.name = name
                self.type = "image/png"
                self._data = data
            def getvalue(self):
                return self._data
            def read(self):
                return self._data
            def seek(self, pos):
                pass
        active_file = _DemoFile(st.session_state.get("demo_image_name", "demo.png"), st.session_state.demo_image_bytes)

    if active_file is not None:
        uploaded_file = active_file
        
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        file_size = len(uploaded_file.read()) / 1024
        uploaded_file.seek(0)
        st.info(f"📄 File: {uploaded_file.name} | Size: {file_size:.1f} KB | Type: {uploaded_file.type}")
        
        if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
            with st.spinner("Analyzing image with AI..."):
                try:
                    from toolkit.builtin import should_use_builtin

                    if should_use_builtin():
                        t0 = time.perf_counter()
                        from toolkit.builtin import invoke

                        result = invoke("caption", {}, file_bytes=uploaded_file.getvalue())
                        latency_ms = (time.perf_counter() - t0) * 1000
                    else:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        t0 = time.perf_counter()
                        response = requests.post(f"{API_BASE}/image/caption", files=files, timeout=60)
                        latency_ms = (time.perf_counter() - t0) * 1000
                        response.raise_for_status()
                        result = response.json()
                    caption = result.get("generated_text", "No caption generated.")
                    
                    st.subheader("🔍 Analysis Results")
                    st.markdown(f'<div class="content-display"><h4>📝 Image Caption</h4><p>{caption}</p></div>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        analysis_report = f"""Image Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
File: {uploaded_file.name}

Caption: {caption}
"""
                        st.download_button(
                            label="📥 Download Report",
                            data=analysis_report,
                            file_name="image_analysis_report.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    with col2:
                        if st.button("⭐ Save to Favorites", key="save_image_analysis"):
                            add_to_favorites("Image Analysis", {
                                'filename': uploaded_file.name,
                                'caption': caption,
                            })
                            st.success("Saved to favorites!")
                    
                    log_to_history("Image Analysis", uploaded_file.name, caption, latency_ms=latency_ms)
                    
                except requests.exceptions.RequestException as e:
                    display_error(f"Could not analyze image: {e}")
                    log_to_history("Image Analysis", uploaded_file.name, str(e), False)
                except Exception as e:
                    display_error(f"An unexpected error occurred: {e}")
                    log_to_history("Image Analysis", uploaded_file.name, str(e), False)

def translation_component():
    """Streamlit component for Language Translation."""
    st.header("🌍 Language Translation")
    st.write("Translate text between English and French.")

    text_input = st.text_area(
        "Enter text to translate (English to French only):", 
        height=200, 
        placeholder="Enter English text here...",
        help="Enter English text to translate to French"
    )
        
    if st.button("🔄 Translate", type="primary", use_container_width=True):
        if not text_input.strip():
            st.warning("Please enter text to translate.")
            return

        with st.spinner("Translating text..."):
            try:
                ok, result, err, latency_ms = call_api(
                    "/translation/translate",
                    {"text": text_input},
                    offline=st.session_state.get("offline_demo", False),
                    timeout=30,
                )
                if not ok:
                    raise requests.exceptions.RequestException(err)
                translated_text = result.get("translated_text", "Translation failed.")
                
                st.session_state.translation_history.append({
                    'original': text_input,
                    'translated': translated_text,
                    'source_lang': 'English',
                    'target_lang': 'French',
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                st.subheader("🔄 Translation Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f'<div class="original-text"><h5>📝 Original (English)</h5><p>{text_input}</p></div>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<div class="translated-text"><h5>🔄 Translation (French)</h5><p>{translated_text}</p></div>', unsafe_allow_html=True)
                
                st.subheader("📊 Translation Metrics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Original Length", f"{len(text_input)} chars")
                with col2:
                    st.metric("Translated Length", f"{len(translated_text)} chars")

                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download Translation",
                        data=f"Original: {text_input}\n\nTranslation: {translated_text}",
                        file_name="translation.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                with col2:
                    if st.button("⭐ Save to Favorites", key="save_translation_result"):
                        add_to_favorites("Translation", {
                            'original': text_input,
                            'translated': translated_text,
                            'source_lang': 'English',
                            'target_lang': 'French'
                        })
                        st.success("Saved to favorites!")
                
                log_to_history("Language Translation", "English → French", translated_text, latency_ms=latency_ms)
                
            except requests.exceptions.RequestException as e:
                display_error(f"Could not translate text: {e}")
                log_to_history("Language Translation", text_input[:100], str(e), False)
            except Exception as e:
                display_error(f"An unexpected error occurred: {e}")
                log_to_history("Language Translation", text_input[:100], str(e), False)

def speech_to_text_component():
    """Streamlit component for Speech-to-Text."""
    st.header("🎤 Speech to Text")
    st.write("Convert spoken audio into written text.")
    st.warning("This feature uses a self-hosted model, which may require initial download time on the backend.")

    st.subheader("📁 Audio File Upload")
    uploaded_audio = st.file_uploader(
        "Upload an audio file (.wav, .mp3, .flac, .m4a)", 
        type=["wav", "mp3", "flac", "m4a"], 
        help="Supported formats: WAV, MP3, FLAC, M4A"
    )
    
    if uploaded_audio:
        st.audio(uploaded_audio, format=uploaded_audio.type)
        file_size = len(uploaded_audio.read()) / 1024
        st.info(f"📄 File: {uploaded_audio.name} | Size: {file_size:.1f} KB | Type: {uploaded_audio.type}")
        uploaded_audio.seek(0) # Reset file pointer after reading size
    
    if st.button("🔊 Convert to Text", type="primary", use_container_width=True):
        if uploaded_audio is None:
            st.warning("Please upload an audio file to transcribe.")
            return

        with st.spinner("Converting speech to text..."):
            try:
                from toolkit.builtin import invoke, should_use_builtin

                if should_use_builtin():
                    t0 = time.perf_counter()
                    result = invoke("stt", {}, file_bytes=uploaded_audio.getvalue())
                    latency_ms = (time.perf_counter() - t0) * 1000
                else:
                    files = {"file": (uploaded_audio.name, uploaded_audio.getvalue(), uploaded_audio.type)}
                    t0 = time.perf_counter()
                    response = requests.post(f"{API_BASE}/stt", files=files, timeout=120)
                    latency_ms = (time.perf_counter() - t0) * 1000
                    response.raise_for_status()
                    result = response.json()
                transcription = result.get("transcribed_text", "Could not transcribe audio.")
                
                word_count = len(transcription.split())
                
                st.subheader("📝 Transcription Results")
                st.markdown(f'<div class="transcription-result">{transcription}</div>', unsafe_allow_html=True)
                display_success("Audio transcribed successfully!")

                st.subheader("📊 Transcription Statistics")
                st.metric("Word Count", word_count)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download Transcript",
                        data=transcription,
                        file_name="transcript.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                with col2:
                    if st.button("⭐ Save to Favorites", key="save_stt_result"):
                        add_to_favorites("Speech to Text", {
                            'audio_source': uploaded_audio.name,
                            'transcription': transcription,
                        })
                        st.success("Saved to favorites!")
                
                log_to_history("Speech to Text", uploaded_audio.name, transcription, latency_ms=latency_ms)
                
            except requests.exceptions.RequestException as e:
                display_error(f"Could not transcribe audio: {e}")
                log_to_history("Speech to Text", uploaded_audio.name, str(e), False)
            except Exception as e:
                display_error(f"An unexpected error occurred: {e}")
                log_to_history("Speech to Text", uploaded_audio.name, str(e), False)

def text_to_speech_component():
    """Streamlit component for Text-to-Speech."""
    st.header("🔊 Text to Speech")
    st.write("Convert text to natural-sounding speech.")

    text_input = st.text_area(
        "Enter text to convert to speech:", 
        height=200, 
        value="Hello! This is a sample text for text-to-speech conversion.",
        placeholder="Enter your text here...",
        help="Enter text to convert to speech audio"
    )
        
    word_count = len(text_input.split()) if text_input else 0
    estimated_duration = word_count / 150 * 60  # Average speaking rate (seconds)

    if st.button("🎤 Generate Speech", type="primary", use_container_width=True):
        if not text_input.strip():
            st.warning("Please enter text to convert.")
            return

        with st.spinner("Generating speech..."):
            try:
                from toolkit.builtin import invoke, should_use_builtin

                if should_use_builtin():
                    t0 = time.perf_counter()
                    audio_bytes = invoke("tts", {"text": text_input})
                    latency_ms = (time.perf_counter() - t0) * 1000
                elif st.session_state.get("offline_demo"):
                    st.info("Offline demo: TTS returns mock metadata only (no audio file).")
                    audio_bytes = b""
                    latency_ms = 0
                else:
                    response = requests.post(
                        f"{API_BASE}/tts",
                        json={"text": text_input},
                        timeout=60,
                    )
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "")
                    if "application/json" in content_type:
                        error_data = response.json()
                        raise Exception(f"TTS API Error: {error_data.get('detail', 'Unknown error')}")
                    audio_bytes = response.content
                    latency_ms = 0
                if not should_use_builtin() and not st.session_state.get("offline_demo") and len(audio_bytes) == 0:
                    raise Exception("No audio content received from the server")
                
                st.subheader("🎵 Generated Audio")
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/wav")
                    display_success("Speech generated successfully!")
                else:
                    st.markdown(
                        f'<div class="content-display">Mock TTS for: <em>{text_input[:200]}</em></div>',
                        unsafe_allow_html=True,
                    )
                
                st.subheader("📊 Audio Information")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Estimated Duration", f"{estimated_duration:.1f}s")
                with col2:
                    # FIX: Corrected file size display
                    st.metric("File Size", f"{len(audio_bytes) / 1024:.1f} KB")
                
                col1, col2 = st.columns(2)
                with col1:
                    if audio_bytes:
                        st.download_button(
                            label="📥 Download Audio",
                            data=audio_bytes,
                            file_name="generated_speech.wav",
                            mime="audio/wav",
                            use_container_width=True,
                        )
                with col2:
                    if st.button("⭐ Save to Favorites", key="save_tts_result"):
                        add_to_favorites("Text to Speech", {
                            'text': text_input,
                            'audio_size_kb': f"{len(audio_bytes) / 1024:.1f} KB",
                            'estimated_duration_s': f"{estimated_duration:.1f}s"
                        })
                        st.success("Saved to favorites!")
                
                log_to_history("Text to Speech", text_input, f"Audio generated ({len(audio_bytes) / 1024:.1f} KB)")
                
            except requests.exceptions.RequestException as e:
                display_error(f"Could not generate speech: {e}")
                log_to_history("Text to Speech", text_input, str(e), False)
            except Exception as e:
                display_error(f"An unexpected error occurred: {e}")
                log_to_history("Text to Speech", text_input, str(e), False)

def chatbot_component():
    """Streamlit component for a simple Chatbot."""
    st.header("💬 AI Chatbot")
    st.write("Engage in a conversation with an AI assistant.")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to ask?"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("AI is thinking..."):
                try:
                    ok, result, err, _ = call_api(
                        "/generation/generate",
                        {"text": prompt},
                        offline=st.session_state.get("offline_demo", False),
                        timeout=90,
                    )
                    if not ok:
                        raise requests.exceptions.RequestException(err)
                    ai_response = result.get("generated_text", "I'm sorry, I couldn't generate a response.")
                    
                    # Basic trimming to avoid prompt repetition in simple models
                    if ai_response.startswith(prompt):
                        ai_response = ai_response[len(prompt):].strip()
                        if ai_response.startswith("\n"): # Remove leading newline if present
                            ai_response = ai_response[1:].strip()
                    
                    # Ensure the response is not empty after trimming
                    if not ai_response:
                        ai_response = "I'm sorry, I couldn't generate a meaningful response."

                    st.markdown(ai_response)
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                    log_to_history("AI Chatbot", prompt, ai_response)

                except requests.exceptions.RequestException as e:
                    error_msg = f"Error communicating with AI: {e}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                    log_to_history("AI Chatbot", prompt, error_msg, False)
                except Exception as e:
                    error_msg = f"An unexpected error occurred: {e}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                    log_to_history("AI Chatbot", prompt, error_msg, False)

def Youtubeing_component(): # Renamed from Youtubeing_component
    """Streamlit component for Question Answering."""
    st.header("❓ Question Answering")
    st.write("Get answers to your questions from provided text or general knowledge.")
    qa_demo = st.session_state.pop("demo_fill_qa", None)
    q_default = qa_demo.get("question", "") if isinstance(qa_demo, dict) else ""
    c_default = qa_demo.get("context", "") if isinstance(qa_demo, dict) else ""

    question = st.text_input(
        "Your Question:",
        value=q_default,
        placeholder="e.g., What is the main benefit of self-hosted AI?",
        help="Enter the question you want the AI to answer.",
    )
    context = st.text_area(
        "Provide Context (Optional):",
        value=c_default,
        height=150,
        placeholder="Paste relevant text here for contextual answers...",
        help="Extractive QA works best with context supplied.",
    )

    if st.button("🧠 Get Answer", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Finding answer..."):
            try:
                ok, result, err, latency_ms = call_api(
                    "/qa/answer",
                    {"question": question, "context": context},
                    offline=st.session_state.get("offline_demo", False),
                    timeout=45,
                )
                if not ok:
                    raise requests.exceptions.RequestException(err)

                st.subheader("✅ Answer")
                st.markdown(f'<div class="content-display"><h3>{result.get("answer", "No answer found.")}</h3></div>', unsafe_allow_html=True)
                
                st.subheader("📊 Answer Details")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Confidence", f"{result.get('confidence', 0.0):.1%}")
                with col2:
                    st.metric("Sources", ", ".join(result.get('sources', ['N/A'])))

                if st.button("⭐ Save QA Result", key="save_qa"):
                    add_to_favorites("Question Answering", {
                        'question': question,
                        'context': context,
                        'answer': result.get("answer"),
                        'confidence': result.get("confidence")
                    })
                    st.success("Saved to favorites!")

                log_to_history("Question Answering", question, result.get("answer", "N/A"), latency_ms=latency_ms)

            except requests.exceptions.RequestException as e:
                display_error(f"Error getting answer: {e}")
                log_to_history("Question Answering", question, str(e), False)
            except Exception as e:
                display_error(f"An unexpected error occurred: {e}")
                log_to_history("Question Answering", question, str(e), False)

# --- Utility Components ---

def history_component():
    """Displays the history of API calls."""
    st.header("📜 API Call History")
    st.write("Track all your interactions with the AI services.")

    if not st.session_state.history:
        st.info("No history yet. Start interacting with the AI services!")
        return

    history_df = pd.DataFrame(st.session_state.history)
    history_df['success'] = history_df['success'].apply(lambda x: '✅ Success' if x else '❌ Failed')
    history_df.index += 1 # Start index from 1 for better readability

    if not history_df.empty and 'latency_ms' in history_df.columns:
        st.metric("Avg latency (logged)", f"{history_df['latency_ms'].dropna().mean():.0f} ms")

    st.dataframe(history_df, use_container_width=True, height=400)
    st.download_button("Export history CSV", history_df.to_csv().encode(), "api_history.csv")

    # Optional: Clear history button
    if st.button("🗑️ Clear History", key="clear_history", type="secondary"):
        st.session_state.history = []
        st.session_state.api_calls_count = 0
        st.rerun()
        st.success("History cleared!")

def favorites_component():
    """Displays user's favorite AI results."""
    st.header("⭐ My Favorites")
    st.write("Your saved AI outputs for quick access.")

    if not st.session_state.favorites:
        st.info("No favorites saved yet. Click the '⭐ Save to Favorites' button on results to add them!")
        return

    for i, item in enumerate(st.session_state.favorites):
        with st.expander(f"{item['type']} - {item['timestamp']}"):
            st.json(item['content'])
            if st.button(f"🗑️ Remove from Favorites", key=f"remove_fav_{i}"):
                st.session_state.favorites.pop(i)
                st.rerun()
                st.success("Removed from favorites.")

def user_preferences_component():
    """Manages user preferences."""
    st.header("⚙️ User Preferences")
    st.write("Customize your AI Toolkit experience.")

    # Theme selection (mock functionality for now)
    st.session_state.user_preferences['theme'] = st.radio(
        "Select Theme:",
        options=['Light', 'Dark'],
        index=0 if st.session_state.user_preferences['theme'] == 'Light' else 1,
        help="This is a mock setting. Theme changes are not yet applied.",
        key="theme_selector"
    )

    # Auto-save results to history
    st.session_state.user_preferences['auto_save'] = st.checkbox(
        "Automatically save results to history:",
        value=st.session_state.user_preferences['auto_save'],
        help="If checked, all successful AI interactions will be logged in your history.",
        key="auto_save_checkbox"
    )

    # Display current preferences
    st.subheader("Current Preferences:")
    for key, value in st.session_state.user_preferences.items():
        st.write(f"- **{key.replace('_', ' ').title()}:** {value}")

    if st.button("💾 Save Preferences", type="primary"):
        # In a real app, you'd save these to a database or file
        st.success("Preferences saved (mock)!")

def system_dashboard_component():
    """Enterprise system dashboard with real session analytics."""
    render_enterprise_dashboard()


# --- Main Application Layout ---

# Header Section
st.markdown(
    hero_html(
        f"{APP_NAME} 🤖",
        "Built-in NLP on Streamlit Cloud · Optional FastAPI gateway for local full models",
    ),
    unsafe_allow_html=True,
)

# Sidebar Navigation
st.sidebar.title("🚀 Navigation")
render_demo_sidebar()
NAV_PAGES = [
    ("Home", "🏠"),
    ("Sentiment Analysis", "🎭"),
    ("Text Summarization", "📄"),
    ("Creative Text Generation", "✍️"),
    ("Image Captioning", "🖼️"),
    ("Language Translation", "🌍"),
    ("Text to Speech", "🔊"),
    ("Speech to Text", "🎤"),
    ("AI Chatbot", "💬"),
    ("Question Answering", "❓"),
    ("Demo Runner", "🎬"),
    ("Document Analyzer", "📑"),
    ("Service Compare", "⚖️"),
    ("Batch Lab", "📦"),
    ("API Playground", "🧪"),
    ("Session Report", "📑"),
    ("Integrations Hub", "🔗"),
    ("Cost Center", "💰"),
    ("API Call History", "📜"),
    ("My Favorites", "⭐"),
    ("User Preferences", "⚙️"),
    ("System Dashboard", "🚀"),
]
_nav_icons = dict(NAV_PAGES)

st.sidebar.caption("Choose a page")
selected_option = st.sidebar.radio(
    "Navigation",
    options=[name for name, _ in NAV_PAGES],
    format_func=lambda x: f"{_nav_icons[x]} {x}",
    key="main_menu_selector",
    label_visibility="collapsed",
)

# Content Display based on selection
st.markdown("---")

if selected_option == "Home":
    render_enterprise_home()

elif selected_option == "Sentiment Analysis":
    sentiment_analysis_component()
elif selected_option == "Text Summarization":
    text_summarization_component()
elif selected_option == "Creative Text Generation":
    text_generation_component()
elif selected_option == "Image Captioning":
    image_captioning_component()
elif selected_option == "Language Translation":
    translation_component()
elif selected_option == "Text to Speech":
    text_to_speech_component()
elif selected_option == "Speech to Text":
    speech_to_text_component()
elif selected_option == "AI Chatbot":
    chatbot_component()
elif selected_option == "Question Answering":
    # Renamed the component function for consistency
    Youtubeing_component() 
elif selected_option == "Demo Runner":
    render_demo_runner()
elif selected_option == "Document Analyzer":
    render_document_analyzer()
elif selected_option == "Service Compare":
    render_service_compare()
elif selected_option == "Batch Lab":
    render_batch_lab()
elif selected_option == "API Playground":
    render_api_playground()
elif selected_option == "Session Report":
    render_session_report()
elif selected_option == "Integrations Hub":
    render_integrations_hub()
elif selected_option == "Cost Center":
    render_cost_center()
elif selected_option == "API Call History":
    history_component()
elif selected_option == "My Favorites":
    favorites_component()
elif selected_option == "User Preferences":
    user_preferences_component()
elif selected_option == "System Dashboard":
    system_dashboard_component()

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style="text-align: center; color: gray;">
        <p>{APP_NAME} | Self-hosted FastAPI + Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True,
)
