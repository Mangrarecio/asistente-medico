import streamlit as st
from transformers import pipeline

# Configuración de la página
st.set_page_config(page_title="Asistente Médico IA", page_icon="⚕️")

st.title("⚕️ Asistente Médico Virtual")
st.write("Cargando modelo inteligente optimizado para la nube...")

# Carga del modelo ajustada para servidores gratuitos
@st.cache_resource
def cargar_asistente():
    # Usamos el mismo modelo pero con una configuración más sencilla
    modelo_ia = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    # Hemos quitado device_map="auto" para evitar el error de memoria
    return pipeline("text-generation", model=modelo_ia)

asistente = cargar_asistente()

# Interfaz de usuario
pregunta = st.text_input("Haz tu consulta médica (en inglés):", placeholder="Ej: Symptoms of flu")

if pregunta:
    with st.spinner('La IA está pensando...'):
        prompt = f"<|system|>\nEres un asistente médico breve.\n<|user|>\n{pregunta}\n<|assistant|>\n"
        
        # Generación de texto
        output = asistente(prompt, max_new_tokens=150, temperature=0.7)
        respuesta = output[0]["generated_text"].split("<|assistant|>\n")[-1]
        
        st.success("Respuesta:")
        st.write(respuesta)

st.divider()
st.caption("Aviso: Esta herramienta es informativa. Consulta a un médico real.")
