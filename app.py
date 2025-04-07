import streamlit as st
import whisper
import google.generativeai as genai
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

st.title("Subtitle Search Engine")

# Load the embeddings model
embed_model = SentenceTransformer("models/all-MiniLM-L6-v2")

# Configure Google Gemini API
genai.configure(api_key="AIzaSyB92k02wczwkOK3VWuLQZ5JyJWj-uAV6Tk")

# Load Whisper model
whisper_model = whisper.load_model("base")

# Load FAISS index and metadata
faiss_index = faiss.read_index("data/faiss_index.index")
with open("data/faiss_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

def transcribe_audio(audio_file):
    """Transcribe audio file to text using Whisper."""
    with open(audio_file, "rb") as f:
        audio = whisper.load_audio(f)
    transcription = whisper_model.transcribe(audio)
    return transcription["text"]

def search_subtitles(query):
    """Search subtitles using FAISS."""
    query_embedding = embed_model.encode([query]).astype("float32")
    distances, indices = faiss_index.search(query_embedding, k=5)
    return [metadata[i]["text"] for i in indices[0]]

# UI
query = st.text_input("Enter your search query:")

# Audio input
audio_file = st.file_uploader("Upload an audio query (optional)", type=["wav", "mp3", "m4a"])

if st.button("Search"):
    if audio_file:
        with st.spinner("Transcribing audio..."):
            query = transcribe_audio(audio_file)
        st.write(f"Transcribed Text: {query}")

    if query.strip():
        with st.spinner("Searching..."):
            results = search_subtitles(query)
        st.write("### Search Results")
        for i, result in enumerate(results):
            st.write(f"**Match {i+1}:** {result}")
    else:
        st.warning("Please enter a search query!")
