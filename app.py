import os
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

# --- SIDEBAR: KURUMSAL VE BİLİMSEL KADRO ---
st.sidebar.title("📌 Kurumsal Bilgiler")

with st.sidebar.expander("🏢 Sanveta Animal Healthcare", expanded=True):
    st.markdown("""
    2025 yılında **Elis Büşra Kılıç Esen** tarafından kurulan **Sanveta Animal Healthcare**, veteriner hekimlerin saha tecrübesi ve uzmanlığını odağına alan, hayvan sağlığı sektöründe faaliyet gösteren yenilikçi bir kuruluştur.
    
    Hayvan sağlığına yönelik geliştirdiğimiz bütüncül çözümlerin yanı sıra, veteriner hekimlerin klinik süreçlerini ve operasyonlarını dijitalleştiren akıllı mobil uygulamalar ve web platformları tasarlıyoruz. Teknolojiyi veteriner tıbbın gereksinimleriyle buluşturarak meslektaşlarımızın iş yükünü hafifletmeyi, klinik verimliliğini artırmayı ve sektördeki hizmet standartlarını daha ileriye taşımayı hedefliyoruz.
    """)

with st.sidebar.expander("👨‍⚕️ Bilim Danışmanı"):
    st.markdown("""
    **Uzman Vet. Hek. Mustafa Esen**  
    *Unvan:* Bilim Danışmanı  
    
    *Hakkında:* İç hastalıkları uzmanlığını 2022 yılında tamamlamış olup, iç hastalıkları anabilim dalında doktora çalışmalarına devam etmektedir. Sanveta bünyesinde geliştirilen tüm yazılım ve dijital platform süreçlerinin medikal/bilimsel denetimini ve koordinasyonunu yürütmektedir.
    """)

with st.sidebar.expander("💻 Yazılım & AI Altyapısı"):
    st.markdown("""
    - **Platform:** VetHelper AI
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
    "ÖNEMLİ KURALLAR:\n"
    "1. Kullanıcının sorduğu tıbbi terminoloji veya vaka bilgisi verilen doküman bağlamında "
    "doğrudan bulunamıyorsa ya da terimde bir yazım/terminoloji hatası varsa bildiklerini uydurma.\n"
    "2. Böyle bir durumda kullanıcıya şu formatta yanıt ver:\n"
    "   'Aradığınız terminoloji veya vaka bilgisi kaynaklarımızda doğrudan bulunamadı. Bunlardan birini mi demek istediniz?'\n"
    "3. Ardından veteriner hekimliği literatürüne uygun olarak kullanıcının neyi kastetmiş olabileceğine dair 5 olası seçeneği maddeler halinde sırala:\n"
    "   1. [Olası Terim / Konu 1] - (Kısa klinik açıklaması)\n"
    "   2. [Olası Terim / Konu 2] - (Kısa klinik açıklaması)\n"
    "   3. [Olası Terim / Konu 3] - (Kısa klinik açıklaması)\n"
    "   4. [Olası Terim / Konu 4] - (Kısa klinik açıklaması)\n"
    "   5. [Olası Terim / Konu 5] - (Kısa klinik açıklaması)\n\n"
    "Doküman Bağlamı:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

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
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
