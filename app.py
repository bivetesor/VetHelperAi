import os
import datetime
import streamlit as st
import gspread
import traceback

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Sayfa yapılandırması
st.set_page_config(
    page_title="VetHelper AI", 
    page_icon="🐾", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🐾 VetHelper AI Asistanı")
st.caption("Veteriner Hekimliği Bilgi Bankası ve Doküman Analiz Sistemi")

# BELBEK OPTİMİZASYONU - Cache temizleme
@st.cache_resource(ttl=3600, max_entries=1)  # 1 saat cache, max 1 giriş
def get_vectorstore():
    """Vektör veritabanını yükler - cache ile optimize edildi"""
    try:
        # Daha küçük embedding modeli
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        db = Chroma(
            persist_directory="./chroma_db", 
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"}  # Daha hızlı sorgu
        )
        return db
    except Exception as e:
        st.error(f"❌ Vektör veritabanı yüklenirken hata: {str(e)}")
        return None

@st.cache_resource(ttl=3600, max_entries=1)
def get_llm(api_key):
    """LLM modelini yükler - cache ile optimize edildi"""
    try:
        return ChatGroq(
            groq_api_key=api_key.strip(),
            model="llama-3.1-8b-instant",
            temperature=0.1,
            timeout=30,
            max_retries=2,
            max_tokens=512  # Daha az token = daha az bellek
        )
    except Exception as e:
        st.error(f"❌ LLM yüklenirken hata: {str(e)}")
        return None

# --- GOOGLE SHEETS KAYIT (Basitleştirilmiş) ---
def log_to_gsheets(query: str, response: str, feedback: str = "Yok"):
    """Google Sheets'e log kaydeder - basitleştirilmiş versiyon"""
    try:
        if "connections" not in st.secrets or "gsheets" not in st.secrets["connections"]:
            return  # Secrets yoksa sessizce geç
        
        sheet_url = st.secrets["connections"]["gsheets"]["spreadsheet"]
        gc = gspread.public_authorize()
        sh = gc.open_by_url(sheet_url)
        worksheet = sh.get_worksheet(0)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        worksheet.append_row([timestamp, query[:500], response[:500], feedback])  # Kısaltılmış
    except Exception:
        pass  # Sessizce geç - loglama kritik değil

# --- SIDEBAR: KURUMSAL BİLGİLER (Basitleştirilmiş) ---
st.sidebar.title("📌 Kurumsal Bilgiler")

with st.sidebar.expander("🏢 Sanveta Animal Healthcare", expanded=True):
    st.markdown("""
    **Sanveta Animal Healthcare** - 2025 yılında **Elis Büşra Kılıç Esen** tarafından kurulmuştur.
    
    Hayvan sağlığı sektöründe yenilikçi çözümler sunan, veteriner hekimlerin klinik süreçlerini dijitalleştiren akıllı platformlar geliştirir.
    """)

with st.sidebar.expander("👨‍⚕️ Bilim Danışmanı"):
    st.markdown("""
    **Uzman Vet. Hek. Mustafa Esen**  
    *Bilim Danışmanı* - İç hastalıkları uzmanı, doktora öğrencisi.
    """)

with st.sidebar.expander("💻 Teknoloji Altyapısı"):
    st.markdown("""
    - **Model:** Groq Llama 3.1 (8B)
    - **DB:** ChromaDB
    - **Embedding:** MiniLM-L12-v2
    - **Mimari:** RAG
    """)

st.sidebar.markdown("---")

# 1. API Key Yönetimi
groq_api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

if not groq_api_key:
    groq_api_key = st.sidebar.text_input("Groq API Key:", type="password")
    if not groq_api_key:
        st.info("Lütfen API anahtarınızı girin.")
        st.stop()

# 2. Vektör veritabanını yükle (optimize edilmiş)
with st.spinner("Bilgi tabanı yükleniyor..."):
    vectorstore = get_vectorstore()
    if vectorstore is None:
        st.error("Vektör veritabanı yüklenemedi. Lütfen daha sonra tekrar deneyin.")
        st.stop()
    
    # Daha az doküman getir
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}  # 4'ten 3'e düşürüldü - daha az bellek
    )

# 3. LLM yükle
llm = get_llm(groq_api_key)
if llm is None:
    st.error("LLM yüklenemedi. API anahtarınızı kontrol edin.")
    st.stop()

# 4. RAG Zinciri (basitleştirilmiş prompt)
system_prompt = (
    "Sen uzman bir veteriner hekim asistanısın. Verilen bağlamı kullanarak kullanıcının sorusuna "
    "doğrudan, tıbbi açıdan doğru ve Türkçe yanıt ver.\n\n"
    "KURALLAR:\n"
    "- Bilmiyorsan uydurma\n"
    "- Bulamazsan 5 olası terim öner\n\n"
    "Bağlam:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 5. Chat Geçmişi (sınırlı tut)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sadece son 10 mesajı göster (bellek tasarrufu)
display_messages = st.session_state.messages[-20:]  # En son 20 mesaj

for idx, msg in enumerate(display_messages):
    actual_idx = len(st.session_state.messages) - len(display_messages) + idx
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        if msg["role"] == "assistant":
            col1, col2, _ = st.columns([1, 1, 10])
            with col1:
                if st.button("👍", key=f"like_{actual_idx}"):
                    st.toast("Beğenildi!")
                    user_q = st.session_state.messages[actual_idx-1]["content"] if actual_idx > 0 else ""
                    log_to_gsheets(user_q, msg["content"], "Beğenildi")
            with col2:
                if st.button("👎", key=f"dislike_{actual_idx}"):
                    st.toast("Beğenilmedi!")
                    user_q = st.session_state.messages[actual_idx-1]["content"] if actual_idx > 0 else ""
                    log_to_gsheets(user_q, msg["content"], "Beğenilmedi")

# 6. Yeni girdi alma
if user_input := st.chat_input("Klinik vaka, semptom veya soru yazın..."):
    # Kullanıcı mesajını ekle
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Mesaj geçmişini sınırla (bellek tasarrufu)
    if len(st.session_state.messages) > 50:
        st.session_state.messages = st.session_state.messages[-50:]
    
    with st.chat_message("assistant"):
        with st.spinner("Yanıt hazırlanıyor..."):
            try:
                # RAG çağrısı
                response = rag_chain.invoke({"input": user_input})
                answer = response["answer"]
                st.markdown(answer)
                
                # Cevabı kaydet
                st.session_state.messages.append({"role": "assistant", "content": answer})
                log_to_gsheets(user_input, answer, "Henüz Seçilmedi")
                
            except Exception as e:
                error_msg = f"❌ Hata: {str(e)}"
                st.error(error_msg)
                
                # Basit hata mesajı
                fallback = "Üzgünüm, bir teknik sorun oluştu. Lütfen tekrar deneyin."
                st.markdown(fallback)
                st.session_state.messages.append({"role": "assistant", "content": fallback})
    
    st.rerun()

# --- BELBEK KULLANIMI GÖSTERGESİ ---
if st.sidebar.checkbox("📊 Sistem Durumu"):
    import psutil
    import gc
    
    process = psutil.Process()
    memory_info = process.memory_info()
    
    st.sidebar.metric("Bellek Kullanımı", f"{memory_info.rss / 1024 / 1024:.1f} MB")
    st.sidebar.metric("GC Nesne Sayısı", len(gc.get_objects()))
    
    if st.sidebar.button("🧹 Bellek Temizle"):
        gc.collect()
        st.sidebar.success("Bellek temizlendi!")
