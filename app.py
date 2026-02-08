import streamlit as st
from transformers import pipeline
from deep_translator import GoogleTranslator

# Configuración de la página
st.set_page_config(page_title="Asistente Médico IA + Traductor", page_icon="⚕️")

st.title("⚕️ Asistente Médico Inteligente")
st.write("Consulta en inglés y traduce la respuesta al español con un clic.")

# Carga del modelo de IA
@st.cache_resource
def cargar_asistente():
    modelo_ia = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    return pipeline("text-generation", model=modelo_ia)

asistente = cargar_asistente()

# Entrada de usuario
pregunta = st.text_input("Describe tus síntomas (en inglés):", placeholder="Ej: Why does my back hurt?")

if pregunta:
    with st.spinner('La IA está analizando...'):
        prompt = f"<|system|>\nEres un asistente médico breve y profesional.\n<|user|>\n{pregunta}\n<|assistant|>\n"
        output = asistente(prompt, max_new_tokens=150, temperature=0.7)
        respuesta_en = output[0]["generated_text"].split("<|assistant|>\n")[-1]
        
        # Guardamos la respuesta en la "memoria" de la sesión para poder traducirla luego
        st.session_state['respuesta_original'] = respuesta_en
        
        st.success("Respuesta original (Inglés):")
        st.write(respuesta_en)

    # BOTÓN DE TRADUCCIÓN (Aparece si hay una respuesta)
    if 'respuesta_original' in st.session_state:
        if st.button("🔄 Traducir respuesta al Español"):
            with st.spinner('Traduciendo...'):
                traduccion = GoogleTranslator(source='en', target='es').translate(st.session_state['respuesta_original'])
                st.info("Traducción al Español:")
                st.write(traduccion)

st.divider()
st.caption("Aviso: Esta IA es informativa. Consulta siempre a un médico real.")