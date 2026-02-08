import streamlit as st
from transformers import pipeline

# Configuración de la aplicación
st.set_page_config(page_title="Asistente Médico IA", page_icon="⚕️")

st.title("⚕️ Asistente Médico Virtual")
st.markdown("---")

# Carga del modelo (IA gratuita TinyLlama)
@st.cache_resource
def cargar_cerebro():
    # Modelo ligero para que funcione en cualquier PC
    return pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device_map="auto")

asistente = cargar_cerebro()

# Interfaz de usuario
pregunta = st.text_input("¿Cuál es tu consulta?", placeholder="Ej: What are the main signs of dehydration?")

if pregunta:
    with st.spinner('Analizando consulta médica...'):
        prompt = f"<|system|>\nEres un asistente médico atento. Responde de forma clara.\n<|user|>\n{pregunta}\n<|assistant|>\n"
        outputs = asistente(prompt, max_new_tokens=200, temperature=0.7)
        respuesta = outputs[0]["generated_text"].split("<|assistant|>\n")[-1]
        
        st.success("Respuesta:")
        st.write(respuesta)

st.divider()
st.caption("Nota: Esta herramienta es informativa. Consulta siempre a un médico profesional.")