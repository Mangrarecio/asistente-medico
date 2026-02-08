import streamlit as st
from transformers import pipeline

# 1. Configuración de la Identidad de la App
st.set_page_config(page_title="Asistente Médico IA", page_icon="⚕️")

st.title("⚕️ Asistente Médico Virtual")
st.markdown("---")
st.write("Bienvenido. Soy una IA entrenada para responder dudas médicas generales.")

# 2. Carga Inteligente del Modelo (TinyLlama)
@st.cache_resource
def cargar_cerebro():
    # Modelo ligero, gratuito y moderno de Hugging Face
    modelo_ia = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    return pipeline("text-generation", model=modelo_ia, device_map="auto")

asistente = cargar_cerebro()

# 3. Interfaz de Consulta
pregunta = st.text_input("¿En qué puedo ayudarte hoy?", placeholder="Ej: What are the main causes of a headache?")

if pregunta:
    with st.spinner('Consultando base de conocimientos...'):
        # Formato de conversación profesional (Prompt Engineering)
        prompt = f"<|system|>\nEres un asistente médico atento y profesional. Responde de forma clara.\n<|user|>\n{pregunta}\n<|assistant|>\n"
        
        # Generar la respuesta
        outputs = asistente(prompt, max_new_tokens=200, temperature=0.7)
        respuesta_final = outputs[0]["generated_text"].split("<|assistant|>\n")[-1]
        
        st.success("Análisis del Asistente:")
        st.write(respuesta_final)

# 4. Pie de página y Advertencias
st.divider()
st.info("💡 **Dato:** Este asistente funciona mejor con preguntas en inglés.")
st.caption("Aviso legal: Esta herramienta no sustituye el consejo de un médico colegiado.")