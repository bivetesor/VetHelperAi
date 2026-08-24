import os
import datetime
import pandas as pd
import streamlit as st
import gspread

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from groq import Groq

st.set_page_config(page_title="VetHelper AI", page_icon="🐾", layout="wide")

st.title("🐾 VetHelper AI Asistanı")
st.caption("Veteriner Hekimliği Bilgi Bankası ve Doküman Analiz Sistemi")

# --- GOOGLE SHEETS BİLGİSİ VE KAYIT (GSPREAD İLE) ---
def log_to_gsheets(query: str, response: str, feedback: str = "Yok"):
    try:
        if "connections" not in st.secrets or "gsheets" not in st.secrets["connections"]:
            return
            
        sheet_url = st.secrets["connections"]["gsheets"]["spreadsheet"]
        gs_client = gspread.public_authorize()
        sh = gs_client.open_by_url(sheet_url)
        worksheet = sh.get_worksheet(0)
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        worksheet.append_row([timestamp, query[:200], response[:200], feedback])
    except Exception as e:
        print(f"Google Sheets Log Hatası: {e}")

# --- SIDEBAR ---
st.sidebar.title("📌 Kurumsal Bilgiler")

with st.sidebar.expander("🏢 Sanveta Animal Healthcare", expanded=True):
    st.markdown("""
    2025 yılında **Elis Büşra Kılıç Esen** tarafından kurulan **Sanveta Animal Healthcare**, veteriner hekimlerin saha tecrübesi ve uzmanlığını odağına alan, hayvan sağlığı sektöründe faaliyet gösteren yenilikçi bir kuruluştur.
    
    Hayvan sağlığına yönelik geliştirdiğimiz bütüncül çözümlerin yanı sıra, veteriner hekimlerin klinik süreçlerini ve operasyonlarını dijitalleştiren akıllı mobil uygulamalar ve web platformları tasarlıyoruz.
    """)

with st.sidebar.expander("👨‍⚕️ Bilim Danışmanı"):
    st.markdown("""
    **Uzman Vet. Hek. Mustafa Esen**  
    *Unvan:* Bilim Danışmanı  
    
    *Hakkında:* İç hastalıkları uzmanlığını 2022 yılında tamamlamış olup, iç hastalıkları anabilim dalında doktora çalışmalarına devam etmektedir.
    """)

with st.sidebar.expander("💻 Yazılım & Teknoloji Altyapısı"):
    st.markdown("""
    - **Geliştirici:** Sanveta Yazılım & AI Ekibi
    - **Vektör Veritabanı:** ChromaDB
    - **Embedding:** Multilingual MiniLM-L12-v2
    - **Mimari:** RAG (Retrieval-Augmented Generation)
    """)

st.sidebar.markdown("---")

# 1. API Key Yönetimi
groq_api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

if not groq_api_key:
    groq_api_key = st.sidebar.text_input("Groq API Key giriniz:", type="password")
    if not groq_api_key:
        st.info("Lütfen devam etmek için kenar çubuğundan Groq API anahtarınızı girin.")
        st.stop()

# 2. Embedding ve ChromaDB Yükleme
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    db = Chroma(
        persist_directory="./chroma_db", 
        embedding_function=embedding_model
    )
    return db

with st.spinner("Bilgi tabanı yükleniyor..."):
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. SADECE CHAT MODELLERİNİ SEÇEN YAPI
@st.cache_resource
def get_active_groq_llm(api_key: str):
    clean_key = api_key.strip()
    client = Groq(api_key=clean_key)
    
    # Tüm modelleri çek
    models_data = client.models.list()
    available_model_ids = [m.id for m in models_data.data]
    
    # Sadece sohbet (Chat/LLM) modelleri (Guard, Whisper, Vision ve classification modelleri hariç)
    valid_chat_priority = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "qwen-2.5-32b",
        "deepseek-r1-distill-llama-70b",
        "gemma2-9b-it"
    ]
    
    chosen_model = None
    for target in valid_chat_priority:
        if target in available_model_ids:
            chosen_model = target
            break
            
    # Eğer öncelik listesindekiler bulunamazsa, içinde "guard", "whisper", "vision" geçmeyen ilk modeli seç
    if not chosen_model:
        for mid in available_model_ids:
            if not any(excluded in mid.lower() for excluded in ["guard", "whisper", "vision", "embed"]):
                chosen_model = mid
                break
                
    if not chosen_model:
        chosen_model = "llama-3.1-8b-instant"

    return ChatGroq(
        groq_api_key=clean_key,
        model=chosen_model,
        temperature=0.1
    ), chosen_model

try:
    llm, active_model_name = get_active_groq_llm(groq_api_key)
    st.sidebar.caption(f"Aktif Sohbet Modeli: `{active_model_name}`")
except Exception as e:
    st.error(f"Groq bağlantı hatası: {e}")
    st.stop()

# 4. RAG Prompt Şablonu
system_prompt = (
    "Sen uzman bir veteriner hekim asistanısın. Aşağıda sağlanan doküman bağlamını (context) "
    "kullanarak kullanıcının sorusuna doğrudan, tıbbi açıdan doğru ve Türkçe yanıt ver.\n"
    "Eğer kullanıcı sadece selam veriyorsa veya genel bir soru soruyorsa nazikçe yanıt ver.\n\n"
    "Doküman Bağlamı:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 5. Mesaj Geçmişi Başlatma
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mesaj geçmişini ekrana yazdır
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        if msg["role"] == "assistant":
            col1, col2, _ = st.columns([1, 1, 10])
            with col1:
                if st.button("👍", key=f"like_{idx}"):
                    user_q = st.session_state.messages[idx-1]["content"] if idx > 0 else ""
                    log_to_gsheets(user_q, msg["content"], feedback="Beğenildi (👍)")
                    st.toast("Beğenildi!")
            with col2:
                if st.button("👎", key=f"dislike_{idx}"):
                    user_q = st.session_state.messages[idx-1]["content"] if idx > 0 else ""
                    log_to_gsheets(user_q, msg["content"], feedback="Beğenilmedi (👎)")
                    st.toast("Beğenilmedi!")

# Yeni Mesaj Girişi
if user_input := st.chat_input("Klinik vaka, semptom veya kaynak soru yazın..."):
    # 1. Kullanıcı mesajını ekrana bas ve hafızaya al
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 2. Asistan yanıtını üret ve ekrana bas
    with st.chat_message("assistant"):
        with st.spinner("Düşünüyorum..."):
            try:
                response = rag_chain.invoke({"input": user_input})
                answer = response.get("answer", "Yanıt oluşturulamadı.")
                st.markdown(answer)
                
                # Hafızaya ekle
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # Arka planda logla
                log_to_gsheets(query=user_input, response=answer, feedback="Henüz Seçilmedi")
                
            except Exception as e:
                error_details = str(e)
                st.error(f"Hata Detayı: {error_details}")
                st.session_state.messages.append({"role": "assistant", "content": f"Hata: {error_details}"})
