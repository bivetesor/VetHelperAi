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
    - **Model:** Groq Llama 3.3 (70B)
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

# 3. Groq LLM Yapılandırması
llm = ChatGroq(
    groq_api_key=groq_api_key.strip(),
    model="llama-3.3-70b-versatile",
    temperature=0.1
)

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
                
                # Arka planda logla (hata verse bile akışı bozmaz)
                log_to_gsheets(query=user_input, response=answer, feedback="Henüz Seçilmedi")
                
            except Exception as e:
                error_details = str(e)
                st.error(f"Hata Detayı: {error_details}")
                st.session_state.messages.append({"role": "assistant", "content": f"Hata: {error_details}"})
