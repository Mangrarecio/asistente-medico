import streamlit as st
from transformers import pipeline
from deep_translator import GoogleTranslator

# Configuración de la página
st.set_page_config(page_title="Simulador Médico IA", page_icon="⚕️")

st.title("⚕️ Analizador de Síntomas (Educativo)")
st.write("Escribe tus síntomas en inglés. La IA analizará el caso de forma teórica.")

# Carga del modelo
@st.cache_resource
def cargar_asistente():
    return pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

asistente = cargar_asistente()

pregunta = st.text_input("Describe los síntomas (en inglés):", placeholder="Ej: I have a high fever and a sore throat")

if pregunta:
    with st.spinner('Analizando caso clínico...'):
        # Nuevo PROMPT más descriptivo
        prompt = f"<|system|>\nEres un experto en medicina educativa. Analiza los síntomas presentados por el usuario, explica qué condiciones médicas suelen asociarse a ellos y qué pruebas se suelen realizar. Sé detallado.\n<|user|>\n{pregunta}\n<|assistant|>\n"
        
        output = asistente(prompt, max_new_tokens=250, temperature=0.7)
        respuesta_en = output[0]["generated_text"].split("<|assistant|>\n")[-1]
        
        st.session_state['respuesta_original'] = respuesta_en
        st.success("Análisis Educativo (Inglés):")
        st.write(respuesta_en)

    if 'respuesta_original' in st.session_state:
        if st.button("🔄 Traducir análisis al Español"):
            with st.spinner('Traduciendo...'):
                traduccion = GoogleTranslator(source='en', target='es').translate(st.session_state['respuesta_original'])
                st.info("Traducción al Español:")
                st.write(traduccion)

st.divider()
st.warning("⚠️ IMPORTANTE: Esta IA no es un médico real. Si tienes fiebre alta, acude a un centro de salud.")