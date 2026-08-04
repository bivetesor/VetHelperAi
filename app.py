import os
import difflib
import streamlit as st
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="VetHelper AI", page_icon="🐾", layout="wide")

st.title("🐾 VetHelper AI Asistanı")
st.caption("Veteriner Hekimliği Bilgi Bankası ve Doküman Analiz Sistemi")

# 1. API Key Yönetimi
groq_api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

if not groq_api_key:
    groq_api_key = st.sidebar.text_input("Groq API Key giriniz:", type="password")
    if not groq_api_key:
        st.info("Lütfen devam etmek için kenar çubuğundan Groq API anahtarınızı girin.")
        st.stop()

# 2. Embedding, ChromaDB ve Kelime Listesi Yükleme
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

@st.cache_data
def load_pdf_words():
    # PDF'leri işlerken kaydettiğin pdf_words.txt dosyasını okur
    words_file = "./pdf_words.txt"
    if os.path.exists(words_file):
        with open(words_file, "r", encoding="utf-8") as f:
            return f.read().split()
    return []

with st.spinner("Bilgi tabanı yükleniyor..."):
    vectorstore = get_vectorstore()
    pdf_words = load_pdf_words()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 3. Groq LLM Yapılandırması
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.1
)

# 4. RAG Prompt Şablonu
system_prompt = (
    "Sen uzman bir veteriner hekim asistanısın. Aşağıda sağlanan doküman bağlamını (context) "
    "kullanarak kullanıcının sorusuna doğrudan, tıbbi açıdan doğru ve Türkçe yanıt ver.\n\n"
    "Eğer aranan terim, semptom veya vaka bilgisi doküman bağlamında açıkça yer almıyorsa veya "
    "kullanıcı yanlış/hatalı bir veterinerlik terminolojisi kullanmışsa:\n"
    "Doğrudan 'Aradığınız terminoloji kaynaklarda bulunamadı. Bunlardan birini mi demek istediniz?' "
    "ifadesini kullan ve kullanıcının sorusuna en yakın 5 alternatif tıbbi terimi/konuyu nedenleriyle birlikte 5 madde halinde sırala.\n\n"
    "Doküman Bağlamı:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Kelime Öneri Fonksiyonu (Fuzzy Matching)
def get_similar_terms(query_text, words_list, n=5):
    if not words_list:
        return []
    # Girdideki son veya en uzun kelimeyi odak terim alarak arayalım
    words_in_query = query_text.lower().split()
    target_word = max(words_in_query, key=len) if words_in_query else query_text
    matches = difflib.get_close_matches(target_word, words_list, n=n, cutoff=0.5)
    return matches

# 5. Chat Geçmişi Arayüzü
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Klinik vaka, semptom veya kaynak soru yazın..."):
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Kaynaklar taranıyor..."):
            response = rag_chain.invoke({"input": user_input})
            answer = response["answer"]
            
            # Eğer yanıt kaynakta bulunamadığına işaret ediyorsa kelime listesinden destek alalım
            if "bulunamamıştır" in answer.lower() or "yer almamaktadır" in answer.lower() or "demek istediniz" in answer.lower():
                similar_words = get_similar_terms(user_input, pdf_words, n=5)
                if similar_words:
                    suggestions_text = "\n\n**Kitaplarınızda geçen en yakın kelimeler:**\n" + "\n".join([f"- **{word}**" for word in similar_words])
                    answer += suggestions_text

            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
