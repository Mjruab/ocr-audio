import streamlit as st
import os
import time
import glob
import cv2
import numpy as np
import pytesseract
from PIL import Image
from gtts import gTTS
from googletrans import Translator

# ── Configuración ──────────────────────────────────────────────────
st.set_page_config(page_title="OCR + Traductor", page_icon="📷", layout="wide")

# ── Estilos ────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #fffde7; color: #333333; }

div.stButton > button {
    background-color: #f9a825;
    color: white;
    border-radius: 10px;
    padding: 10px 24px;
    border: none;
    font-size: 16px;
    transition: background-color 0.3s ease;
}
div.stButton > button:hover { background-color: #f57f17; color: white; }
section[data-testid="stSidebar"] { background-color: #fff9c4; }
h1, h2, h3 { color: #f57f17; }

[data-testid="metric-container"] {
    background: #fff8e1;
    border: 1px solid #ffe082;
    border-top: 3px solid #f9a825;
    border-radius: 8px;
    padding: 18px 22px;
}
[data-testid="metric-container"] label {
    color: #f57f17 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #333333 !important;
    font-weight: 700 !important;
}
div[data-testid="stExpander"] {
    border: 1px solid #ffe082 !important;
    border-radius: 8px !important;
    background: #fff8e1 !important;
}
hr { border-color: #ffe082 !important; }

.texto-resultado {
    background: #fff8e1;
    border: 1px solid #ffe082;
    border-left: 5px solid #f9a825;
    border-radius: 8px;
    padding: 20px;
    font-size: 16px;
    line-height: 1.8;
    color: #333333;
    white-space: pre-wrap;
}
</style>
""", unsafe_allow_html=True)

# ── Limpieza de archivos viejos ────────────────────────────────────
def remove_files(n):
    mp3_files = glob.glob("temp/*mp3")
    if len(mp3_files) != 0:
        now = time.time()
        for f in mp3_files:
            if os.stat(f).st_mtime < now - n * 86400:
                os.remove(f)
remove_files(7)

try:
    os.mkdir("temp")
except:
    pass

# ── Título ─────────────────────────────────────────────────────────
st.title("📷 OCR + Traductor de Texto")
st.markdown("Extrae texto de imágenes usando OCR y tradúcelo al idioma que necesites.")

# ── Sidebar ────────────────────────────────────────────────────────
translator = Translator()
idiomas = {
    "Inglés":   "en",
    "Español":  "es",
    "Bengali":  "bn",
    "Coreano":  "ko",
    "Mandarín": "zh-cn",
    "Japonés":  "ja",
}
acentos = {
    "Defecto":        "com",
    "India":          "co.in",
    "Reino Unido":    "co.uk",
    "Estados Unidos": "com",
    "Canada":         "ca",
    "Australia":      "com.au",
    "Irlanda":        "ie",
    "Sudáfrica":      "co.za",
}

with st.sidebar:
    st.title("⚙️ Configuración")

    st.markdown("### 🖼️ Imagen")
    filtro = st.radio("Filtro para cámara:", ["Sin filtro", "Invertir colores",
                                               "Escala de grises", "Alto contraste"])

    st.markdown("---")
    st.markdown("### 🌐 Traducción")
    in_lang  = st.selectbox("Idioma de entrada:",  list(idiomas.keys()))
    out_lang = st.selectbox("Idioma de salida:",   list(idiomas.keys()), index=1)
    acento   = st.selectbox("Acento:", list(acentos.keys()))
    input_language  = idiomas[in_lang]
    output_language = idiomas[out_lang]
    tld             = acentos[acento]

    display_output_text = st.checkbox("Mostrar texto traducido")

    st.markdown("---")
    st.markdown("### 📖 ¿Cómo funciona?")
    st.markdown("**1.** Captura o sube una imagen con texto.")
    st.markdown("**2.** El OCR extrae el texto automáticamente.")
    st.markdown("**3.** Selecciona idiomas y convierte a audio.")

# ── Fuente de imagen ───────────────────────────────────────────────
st.markdown("### 📸 Fuente de imagen")
tab_cam, tab_file = st.tabs(["📷 Cámara", "🖼️ Subir archivo"])

img_file_buffer = None
bg_image        = None

with tab_cam:
    cam_ = st.checkbox("Activar cámara")
    if cam_:
        img_file_buffer = st.camera_input("Toma una foto")

with tab_file:
    bg_image = st.file_uploader("Cargar imagen:", type=["png", "jpg", "jpeg"])

# ── Procesamiento de imagen ────────────────────────────────────────
text = ""

def aplicar_filtro(cv2_img, filtro):
    if filtro == "Invertir colores":
        return cv2.bitwise_not(cv2_img)
    elif filtro == "Escala de grises":
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif filtro == "Alto contraste":
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    return cv2_img

# Desde archivo
if bg_image is not None:
    st.markdown("---")
    st.markdown("### 🖼️ Imagen cargada")
    file_bytes   = np.asarray(bytearray(bg_image.read()), dtype=np.uint8)
    cv2_img      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    cv2_img_proc = aplicar_filtro(cv2_img, filtro)
    img_rgb      = cv2.cvtColor(cv2_img_proc, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original**")
        st.image(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col2:
        st.markdown(f"**Con filtro: {filtro}**")
        st.image(img_rgb, use_container_width=True)

    with st.spinner("🔍 Extrayendo texto..."):
        text = pytesseract.image_to_string(img_rgb)

# Desde cámara
elif img_file_buffer is not None:
    st.markdown("---")
    st.markdown("### 📷 Foto tomada")
    bytes_data   = img_file_buffer.getvalue()
    cv2_img      = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    cv2_img_proc = aplicar_filtro(cv2_img, filtro)
    img_rgb      = cv2.cvtColor(cv2_img_proc, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original**")
        st.image(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col2:
        st.markdown(f"**Con filtro: {filtro}**")
        st.image(img_rgb, use_container_width=True)

    with st.spinner("🔍 Extrayendo texto..."):
        text = pytesseract.image_to_string(img_rgb)

# ── Resultados ─────────────────────────────────────────────────────
if text.strip():
    st.markdown("---")

    # Métricas
    palabras   = len(text.split())
    caracteres = len(text.strip())
    lineas     = len([l for l in text.split("\n") if l.strip()])

    st.markdown("### 📊 Resumen")
    m1, m2, m3 = st.columns(3)
    m1.metric("📝 Palabras",   palabras)
    m2.metric("🔤 Caracteres", caracteres)
    m3.metric("📄 Líneas",     lineas)

    # Texto extraído
    st.markdown("### 📋 Texto extraído")
    st.markdown(
        f'<div class="texto-resultado">{text}</div>',
        unsafe_allow_html=True
    )
    st.download_button("⬇️ Descargar texto (.txt)",
                       data=text, file_name="texto_ocr.txt", mime="text/plain")

    # Traducción y audio
    st.markdown("---")
    st.markdown("### 🔊 Traducción y Audio")

    def text_to_speech(input_language, output_language, text, tld):
        translation = translator.translate(text, src=input_language, dest=output_language)
        trans_text  = translation.text
        tts         = gTTS(trans_text, lang=output_language, tld=tld, slow=False)
        try:
            my_file_name = text[0:20]
        except:
            my_file_name = "audio"
        tts.save(f"temp/{my_file_name}.mp3")
        return my_file_name, trans_text

    if st.button("🔄 Traducir y convertir a audio"):
        with st.spinner("Traduciendo..."):
            result, output_text = text_to_speech(
                input_language, output_language, text, tld)
            audio_file  = open(f"temp/{result}.mp3", "rb")
            audio_bytes = audio_file.read()

        st.success("✅ ¡Audio generado!")
        st.audio(audio_bytes, format="audio/mp3", start_time=0)

        if display_output_text:
            st.markdown("### 📝 Texto traducido:")
            st.info(output_text)

elif img_file_buffer is not None or bg_image is not None:
    st.warning("⚠️ No se detectó texto. Intenta con mejor iluminación o contraste.")
