
import streamlit as st
import joblib
import re
import os

def bersihkan_teks(teks):
    teks = teks.lower()
    teks = re.sub(r"http\\S+|www\\S+|https\\S+", '', teks)
    teks = re.sub(r'[^a-zA-Z\\s]', '', teks)
    teks = re.sub(r'\\s+', ' ', teks)
    return teks.strip()

model_path = os.path.join("models", "naive_bayes_model.pkl")
vectorizer_path = os.path.join("models", "vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

st.set_page_config(page_title="Sentimen IKN", page_icon="🏙️")
st.title("📊 Analisis Sentimen Relokasi IKN")
st.markdown("Masukkan komentar dan dapatkan klasifikasi sentimen.")

user_input = st.text_area("💬 Komentar TikTok")

if st.button("Analisis Sentimen"):
    if not user_input.strip():
        st.warning("Komentar tidak boleh kosong.")
    else:
        cleaned = bersihkan_teks(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == "positif":
            st.success("✅ Sentimen Positif terhadap IKN.")
        else:
            st.error("⚠️ Sentimen Negatif terhadap IKN.")

st.markdown("---")
st.caption("Mahasiswa Statistika | Analisis Sentimen IKN")
