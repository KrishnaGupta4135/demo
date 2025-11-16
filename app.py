# app_teacher_assistant_fixed.py
import os
import re
import shutil
import streamlit as st
from datetime import datetime

# ---- Keep your original imports (adjust if your environment differs) ----
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

# ---------------------- CONFIGURATION ----------------------
st.set_page_config(page_title="📘 NCERT Teacher Assistant", layout="wide")

# NOTE: store your real key securely (do NOT commit to source control)
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "AIzaSyDBxeWsg0PNU6oshQegElRrRjcT06L_9_w")

# ensure directories exist
UPLOAD_DIR = "uploaded_pdfs"
PERSIST_DIR_ROOT = "ncert_chroma_langchain_db"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR_ROOT, exist_ok=True)

# ---------------------- HELPERS ----------------------
def sanitize_collection_name(name: str) -> str:
    """
    Sanitize user-selected filename into a Chroma-friendly collection name:
      - Remove extension
      - Replace spaces with underscores
      - Keep only allowed chars [a-zA-Z0-9._-]
      - Ensure 3-512 length and starts/ends with alnum
    """
    if not name:
        return "NCERT_collection"

    base = os.path.splitext(name)[0]
    base = base.replace(" ", "_")
    base = re.sub(r"[^A-Za-z0-9._-]", "", base)

    if not base:
        base = "NCERT_collection"

    if not re.match(r"^[A-Za-z0-9].*[A-Za-z0-9]$", base):
        base = re.sub(r"^[^A-Za-z0-9]+", "", base)
        base = re.sub(r"[^A-Za-z0-9]+$", "", base)
        if not base:
            base = "NCERT_collection"

    if len(base) < 3:
        base = (base + "x" * 3)[:3]

    if len(base) > 512:
        base = base[:512]

    return base

def make_persist_directory_for_collection(collection_name: str) -> str:
    """Create per-collection persist directory to avoid sqlite locking between collections."""
    safe_dir = os.path.join(PERSIST_DIR_ROOT, collection_name)
    os.makedirs(safe_dir, exist_ok=True)
    return safe_dir

def clear_all_data():
    """Remove uploaded PDFs and Chroma persistence directories (use with caution)."""
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
        os.makedirs(UPLOAD_DIR, exist_ok=True)
    if os.path.exists(PERSIST_DIR_ROOT):
        shutil.rmtree(PERSIST_DIR_ROOT)
        os.makedirs(PERSIST_DIR_ROOT, exist_ok=True)
    st.success("🧹 Cleared uploaded PDFs and Chroma DB folders. You can upload fresh files now.")

# ---------------------- MODEL & EMBEDDINGS ----------------------
@st.cache_resource
def get_model_and_embeddings():
    """
    Initialize model and embeddings once. Cached by Streamlit.
    """
    model = init_chat_model("google_genai:gemini-2.5-flash-lite")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    return model, embeddings

model, embeddings = get_model_and_embeddings()

# ---------------------- LOAD & STORE CHAPTER ----------------------
# NOTE: leading underscore in `_embeddings_obj` stops Streamlit trying to hash it
@st.cache_resource(show_spinner=False)
def load_and_store_book(pdf_path: str, raw_collection_name: str, _embeddings_obj) -> Chroma:
    """
    Load, split, embed, and store PDF content in Chroma DB.
    - pdf_path: local path to pdf
    - raw_collection_name: original filename (will be sanitized)
    - _embeddings_obj: embeddings instance (prefixed with _ to avoid hashing)
    """
    sanitized = sanitize_collection_name(raw_collection_name)
    persist_dir = make_persist_directory_for_collection(sanitized)

    # load PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)

    # Create Chroma vector store. Try embedding object, fallback to embed_documents callable
    try:
        vector_store = Chroma(
            collection_name=sanitized,
            embedding_function=_embeddings_obj,
            persist_directory=persist_dir,
        )
    except Exception as e:
        try:
            vector_store = Chroma(
                collection_name=sanitized,
                embedding_function=_embeddings_obj.embed_documents,
                persist_directory=persist_dir,
            )
        except Exception as e2:
            raise RuntimeError(f"Failed to create Chroma collection: {e} ; fallback: {e2}")

    # add documents
    vector_store.add_documents(all_splits)
    return vector_store

# ---------------------- DYNAMIC PROMPT AGENT ----------------------
def create_dynamic_agent(vector_store):
    @dynamic_prompt
    def prompt_with_context(request: ModelRequest) -> str:
        """Dynamically inject retrieved context into the chat."""
        last_query = request.state["messages"][-1].text
        retrieved_docs = vector_store.similarity_search(last_query, k=3)
        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

        system_message = (
            "You are an intelligent NCERT teaching assistant. "
            "Use and cite only the uploaded book content for answers.\n\n"
            f"Context:\n{docs_content}"
        )
        return system_message

    return create_agent(model, tools=[], middleware=[prompt_with_context])

# ---------------------- MAIN APP UI ----------------------
st.title("📘 NCERT Teacher Assistant")
st.markdown("An AI assistant that reads NCERT chapters and helps create lesson plans, summaries, and quizzes.")

# ---- Sidebar for document management ----
st.sidebar.header("📂 Manage Documents")
if st.sidebar.button("🧹 Clear All Data"):
    clear_all_data()

# File upload
uploaded_file = st.sidebar.file_uploader("Upload NCERT Chapter (PDF)", type=["pdf"])

# Save uploaded file
if uploaded_file:
    safe_name = uploaded_file.name
    file_path = os.path.join(UPLOAD_DIR, safe_name)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"✅ Uploaded: {safe_name}")

# Show existing PDFs
pdf_files = [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith(".pdf")] if os.path.exists(UPLOAD_DIR) else []

if pdf_files:
    selected_pdf = st.sidebar.selectbox("Select a document", pdf_files)
else:
    selected_pdf = None
    st.sidebar.info("No PDFs in uploaded_pdfs. Upload a PDF to start.")

mode = st.sidebar.radio("Select Mode", ["qa", "lesson_plan", "summary", "quiz"], horizontal=True)

st.sidebar.markdown("---")
st.sidebar.info("💡 Modes:\n- **Q&A:** Ask direct questions\n- **Lesson Plan:** Generate teaching plans\n- **Summary:** Summarize topics\n- **Quiz:** Create NCERT-based quizzes")

# ---- Chat Interface ----
if selected_pdf:
    pdf_path = os.path.join(UPLOAD_DIR, selected_pdf)

    # Load vector store for selected PDF
    with st.spinner("📚 Loading and indexing document..."):
        try:
            vector_store = load_and_store_book(pdf_path, raw_collection_name=selected_pdf, _embeddings_obj=embeddings)
        except Exception as e:
            st.error(f"Failed to index document: {e}")
            st.stop()

    agent = create_dynamic_agent(vector_store)

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

    # Input box
    user_query = st.chat_input(f"Ask or request ({mode} mode)...")

    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Build prompt based on mode
        if mode == "qa":
            # Add a small intro line and spacing for clarity
            prompt = (
                f"Answer the following question clearly and accurately based only on the uploaded NCERT content.\n\n"

            )
        elif mode == "lesson_plan":
            prompt = (
                f"Create a comprehensive lesson plan for the topic '{user_query}' "
                "based on the uploaded NCERT content only. Include learning objectives, activities, materials, examples, and summary.\n\n"
            )
        elif mode == "summary":
            prompt = f"Provide a detailed summary for '{user_query}' based only on the uploaded NCERT content.\n\n"
        elif mode == "quiz":
            prompt = (
                f"Generate a 10-question quiz with answers for the topic '{user_query}' "
                "based only on the uploaded NCERT chapter.\n\n"
            )
        else:   
            prompt = user_query


        # Stream AI response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            for step in agent.stream(
                {"messages": [{"role": "user", "content": prompt}]},
                stream_mode="values",
            ):
                # different agent implementations have different step shapes; guard access
                try:
                    last_msg = step["messages"][-1]
                    text = getattr(last_msg, "content", "") if last_msg else ""
                except Exception:
                    # fallback if structure differs
                    text = step.get("text", "") if isinstance(step, dict) else str(step)
                full_response += text
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})

else:
    st.info("📥 Please upload or select an NCERT PDF from the sidebar to start.")
    