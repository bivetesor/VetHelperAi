                """
                
                st.error(error_details)
                
                # Hata logla
                log_error(f"RAG çağrısı hatası:\n{error_details}")
                
                # Kullanıcıya basitleştirilmiş mesaj
                fallback_answer = """
Üzgünüm, şu anda yanıt oluştururken bir teknik sorun yaşandı. 

**Olası nedenler:**
- Groq API'sine erişim sorunu
- Zaman aşımı
- API anahtarı geçersizliği

**Öneriler:**
1. Lütfen birkaç dakika bekleyip tekrar deneyin
2. Sorunuzu daha basit ifadelerle tekrar sorun
3. Sayfayı yenileyip tekrar deneyin

Hata detayları yukarıda gösterilmiştir.
"""
                st.markdown(fallback_answer)
                st.session_state.messages.append({"role": "assistant", "content": fallback_answer})
                
                try:
                    log_to_gsheets(query=user_input, response=f"HATA: {str(e)}", feedback="Sistem Hatası")
                except:
                    pass
    
    st.rerun()

# Debug modu - error_log.txt dosyasını göster
if st.sidebar.checkbox("🔧 Debug Modu"):
    st.sidebar.subheader("Hata Logları")
    try:
        if os.path.exists(ERROR_LOG_FILE):
            with open(ERROR_LOG_FILE, "r", encoding="utf-8") as f:
                log_content = f.read()
            st.sidebar.text_area("Log İçeriği", log_content, height=300)
        else:
            st.sidebar.info("Henüz hata logu oluşturulmamış.")
    except Exception as e:
        st.sidebar.error(f"Log okuma hatası: {e}")
