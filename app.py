import streamlit as st
import whisper
import google.generativeai as genai
import chromadb
from sentence_transformers import SentenceTransformer

st.title("Subtitle Search Engine")

# Load the embeddings model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Configure Google Gemini API
genai.configure(api_key="AIzaSyB92k02wczwkOK3VWuLQZ5JyJWj-uAV6Tk")

# Load Whisper model
whisper_model = whisper.load_model("base")

def transcribe_audio(audio_file):
    """Transcribe audio file to text using Whisper."""
    with open(audio_file, "rb") as f:
        audio = whisper.load_audio(f)
    transcription = whisper_model.transcribe(audio)
    return transcription["text"]

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path="data/chroma_db")
collection = chroma_client.get_collection("subtitles")

def search_subtitles(query):
    """Search subtitles using ChromaDB."""
    query_embedding = embed_model.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=5)
    return [res["text"] for res in results["metadatas"][0]]

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

