import streamlit as st
from transformers import pipeline

# Configuración de la página
st.set_page_config(page_title="Asistente Médico IA", page_icon="⚕️")

st.title("⚕️ Asistente Médico Virtual")
st.write("Bienvenido. Soy una IA experimental. Por favor, consulta siempre a un profesional.")

# Función para cargar el modelo (el "cerebro")
@st.cache_resource # Esto hace que solo se cargue una vez y no cada vez que escribas
def cargar_modelo():
    # Usamos un modelo ligero y moderno (TinyLlama)
    pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    return pipe

asistente = cargar_modelo()

# Interfaz de usuario
pregunta = st.text_input("¿En qué puedo ayudarte hoy?", placeholder="Ej: ¿Qué síntomas tiene la gripe?")

if pregunta:
    with st.spinner('Pensando...'):
        # Creamos el formato de chat para el modelo
        messages = [
            {"role": "system", "content": "Eres un asistente médico amable y profesional. Responde de forma clara."},
            {"role": "user", "content": pregunta},
        ]
        
        # Generar respuesta
        prompt = asistente.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = asistente(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)
        respuesta = outputs[0]["generated_text"].split("<|assistant|>")[-1]
        
        st.subheader("Respuesta del Asistente:")
        st.write(respuesta)

st.info("Nota: Este modelo está en inglés por defecto, pero podemos ajustarlo más adelante.")