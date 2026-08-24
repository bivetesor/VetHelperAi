import os
import datetime
import streamlit as st
import gspread
import gc

st.set_page_config(
    page_title="VetHelper AI", 
    page_icon="🐾", 
    layout="centered"
)

st.title("🐾 VetHelper AI")
st.caption("Veteriner Hekimlik Asistanı")

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- GOOGLE SHEETS LOGLAMA ---
def log_to_gsheets(query: str, response: str, feedback: str = "Yok"):
    try:
        if "connections" not in st.secrets or "gsheets" not in st.secrets["connections"]:
            return
        sheet_url = st.secrets["connections"]["gsheets"]["spreadsheet"]
        # 'gc' yerine 'gs_client' kullanılarak modül çakışması önlendi
        gs_client = gspread.public_authorize()
        sh = gs_client.open_by_url(sheet_url)
        worksheet = sh.get_worksheet(0)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        worksheet.append_row([timestamp, query[:200], response[:200], feedback])
    except Exception:
        pass

# --- SIDEBAR & AYARLAR ---
with st.sidebar:
    st.markdown("### 📌 Kurumsal")
    st.markdown("**Sanveta Animal Healthcare**")
    st.markdown("Veteriner Hekimlik AI Asistanı")
    st.markdown("---")
    
    groq_api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        groq_api_key = st.text_input("Groq API Key:", type="password")
        if not groq_api_key:
            st.info("API anahtarınızı girin")
            st.stop()
    
    hf_token = st.secrets.get("HF_TOKEN") or os.environ.get("HF_TOKEN")
    st.caption("Model: Llama 3.1-8B | Vektör DB: Chroma")

# --- CACHE'LENMİŞ KAYNAKLAR (ttl kaldırıldı) ---
@st.cache_resource
def load_embedding_model(token):
    # PyTorch çalıştırmaz, API üzerinden embedding alır (~0 MB RAM)
    return HuggingFaceInferenceAPIEmbeddings(
        api_key=token,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def load_vectorstore(_embedding):
    try:
        return Chroma(
            persist_directory="./chroma_db",
            embedding_function=_embedding,
            collection_metadata={"hnsw:space": "cosine"}
        )
    except Exception as e:
        st.error(f"Vectorstore hatası: {e}")
        return None

@st.cache_resource
def load_llm(api_key):
    return ChatGroq(
        groq_api_key=api_key.strip(),
        model="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=256,
        timeout=20,
        max_retries=1
    )

# --- SİSTEMİ YÜKLE ---
embedding = load_embedding_model(hf_token)
vectorstore = load_vectorstore(embedding)

if vectorstore is None:
    st.error("Vektör veritabanı yüklenemedi.")
    st.stop()

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
llm = load_llm(groq_api_key)

system_prompt = (
    "Veteriner hekim asistanısın. Verilen bağlama göre kullanıcıya kısa ve öz yanıt ver.\n"
    "Bilmiyorsan 'Bu konuda bilgim yok' de.\n\n"
    "Bağlam: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# --- CHAT ARAYÜZÜ ---
if "messages" not in st.session_state:
    st.session_state.messages = []

start_idx = max(0, len(st.session_state.messages) - 10)
for i in range(start_idx, len(st.session_state.messages)):
    msg = st.session_state.messages[i]
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and i >= len(st.session_state.messages) - 5:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍", key=f"like_{i}"):
                    user_q = st.session_state.messages[i-1]["content"] if i > 0 else ""
                    log_to_gsheets(user_q, msg["content"], "Beğenildi")
                    st.toast("Teşekkürler!")
            with col2:
                if st.button("👎", key=f"dislike_{i}"):
                    user_q = st.session_state.messages[i-1]["content"] if i > 0 else ""
                    log_to_gsheets(user_q, msg["content"], "Beğenilmedi")
                    st.toast("Geri bildirim kaydedildi")

if user_input := st.chat_input("Sorunuzu yazın..."):
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    if len(st.session_state.messages) > 20:
        st.session_state.messages = st.session_state.messages[-20:]
    
    with st.chat_message("assistant"):
        with st.spinner("Düşünüyorum..."):
            try:
                response = rag_chain.invoke({"input": user_input})
                answer = response["answer"][:500]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                log_to_gsheets(user_input, answer, "Henüz Seçilmedi")
            except Exception as e:
                st.error(f"Hata: {str(e)[:100]}")
                fallback = "Teknik bir sorun oluştu. Lütfen tekrar deneyin."
                st.markdown(fallback)
                st.session_state.messages.append({"role": "assistant", "content": fallback})
            finally:
                gc.collect()
    
    st.rerun()
