# AI Services Toolkit Pro 🤖

## Empowering your tasks with advanced AI capabilities

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face%20Spaces-Deploy-blue)](https://huggingface.co/spaces/your-username/your-space-name)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36.0-FF4B4B.svg)](https://streamlit.io/)

---

## 🚀 Project Overview

The **AI Services Toolkit Pro** is a powerful and versatile application designed to provide a suite of advanced AI capabilities through a user-friendly web interface. It leverages a high-performance FastAPI backend to host various machine learning models and a responsive Streamlit frontend for seamless user interaction. The application is containerized using Docker, making it highly portable and easy to deploy on platforms like Hugging Face Spaces.

This toolkit allows users to perform tasks such as:

* **Sentiment Analysis:** Understand the emotional tone of text.
* **Text Summarization:** Condense long documents into concise summaries.
* **Creative Text Generation:** Generate new text based on a given prompt.
* **Image Captioning:** Create descriptive captions for uploaded images.
* **Language Translation:** Translate text from English to French.
* **Text-to-Speech (TTS):** Convert written text into natural-sounding speech.
* **Speech-to-Text (STT):** Transcribe spoken audio into written text.
* **AI Chatbot:** Engage in a conversational dialogue with an AI assistant.
* **Question Answering (Mock):** Get answers to questions (currently a mock feature, ready for real model integration).

---

## ✨ Key Features & Technologies

* **FastAPI Backend:**
    * High-performance Python web framework for building robust APIs.
    * Asynchronous operations for efficient handling of concurrent requests.
    * Automatic interactive API documentation (Swagger UI).
* **Streamlit Frontend:**
    * Rapidly build interactive web applications with pure Python.
    * User-friendly interface for all AI services.
    * Custom CSS for enhanced aesthetics.
* **Hugging Face Transformers Pipelines:**
    * Utilizes pre-trained state-of-the-art AI models for various tasks.
    * Models include:
        * `distilbert-base-uncased-finetuned-sst-2-english` (Sentiment Analysis)
        * `sshleifer/distilbart-cnn-12-6` (Text Summarization)
        * `Helsinki-NLP/opus-mt-en-fr` (Language Translation)
        * `nlpconnect/vit-gpt2-image-captioning` (Image Captioning)
        * **`microsoft/DialoGPT-medium` (for Chatbot / Text Generation)**
        * **`microsoft/speecht5_tts` (Text-to-Speech) with speaker embeddings**
        * **`openai/whisper-tiny` (Speech-to-Text) with audio resampling**
* **Docker Containerization:**
    * Ensures a consistent and isolated environment for deployment.
    * Streamlines the deployment process to any container-compatible platform.
* **Hugging Face Spaces:**
    * Seamless deployment and hosting of the full-stack application.
    * Free tier available for experimentation and demonstration.
* **Robust Audio Handling:** Includes `soundfile`, `scipy.io.wavfile`, and `wave` for reliable audio input/output.
* **Session Management:** Keeps track of API call history and user favorites.
* **System Dashboard:** Provides real-time insights into backend status and API usage.

---

## 🛠️ Getting Started

Follow these steps to set up and run the AI Services Toolkit Pro locally.

### Prerequisites

Before you begin, ensure you have the following installed:

* **Python 3.10+**: [Download Python](https://www.python.org/downloads/)
* **pip**: Python package installer (comes with Python)
* **Git**: [Download Git](https://git-scm.com/downloads)
* **Docker** (Optional, for containerized deployment): [Download Docker](https://www.docker.com/products/docker-desktop)
* **Hugging Face Account** (Optional, for cloud deployment): [Sign up on Hugging Face](https://huggingface.co/join)

### 1. Clone the Repository

First, clone the project repository to your local machine:

```bash
git clone [https://github.com/your-username/ai-services-toolkit-pro.git](https://github.com/your-username/ai-services-toolkit-pro.git)
cd ai-services-toolkit-pro
