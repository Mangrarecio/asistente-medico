import streamlit as st
from transformers import pipeline

# 1. Configuración de la Identidad de la App
st.set_page_config(page_title="Asistente Médico IA", page_icon="⚕️")

st.title("⚕️ Asistente Médico Virtual")
st.write("El sistema está listo. Por favor, describe tus dudas en inglés para mayor precisión.")

# 2. Carga Inteligente del Modelo (TinyLlama)
@st.cache_resource
def cargar_asistente():
    # Modelo TinyLlama: ligero y eficiente para despliegue en la nube
    # Se carga sin 'device_map' para asegurar compatibilidad con la CPU de Streamlit
    modelo_ia = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    return pipeline("text-generation", model=modelo_ia)

asistente = cargar_asistente()

# 3. Interfaz de Consulta
pregunta = st.text_input("Haz tu consulta médica (en inglés):", placeholder="Ej: What are the symptoms of flu?")

if pregunta:
    with st.spinner('La IA está analizando tu consulta...'):
        # Formato de "Prompt" para guiar la personalidad de la IA
        prompt = f"<|system|>\nEres un asistente médico breve y profesional.\n<|user|>\n{pregunta}\n<|assistant|>\n"
        
        # Generación de la respuesta
        output = asistente(prompt, max_new_tokens=150, temperature=0.7)
        respuesta_final = output[0]["generated_text"].split("<|assistant|>\n")[-1]
        
        st.success("Respuesta del Asistente:")
        st.write(respuesta_final)

# 4. Pie de página y Advertencias
st.divider()
st.caption("Aviso: Esta herramienta es puramente educativa. Consulte siempre a un médico real.")