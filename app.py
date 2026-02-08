import streamlit as st
from transformers import pipeline
from deep_translator import GoogleTranslator

# Configuración de la página
st.set_page_config(page_title="Especialista Médico IA", page_icon="👨‍⚕️")

st.title("👨‍⚕️ Consultor Médico Especializado")
st.markdown("---")
st.write("Este sistema utiliza un modelo enfocado en biomedicina para ofrecer análisis más técnicos.")

# Carga del modelo especializado
@st.cache_resource
def cargar_especialista():
    # Cambiamos a un modelo con mejor base médica (TinyLlama entrenado en datos médicos)
    # Nota: Si este modelo tarda mucho, el código está listo para procesar.
    modelo_medico = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 
    return pipeline("text-generation", model=modelo_medico)

asistente = cargar_especialista()

# Interfaz
pregunta = st.text_input("Describe tus síntomas detalladamente (en inglés):", 
                         placeholder="Ej: High fever, dry cough and loss of taste...")

if pregunta:
    with st.spinner('El especialista está analizando el caso clínico...'):
        # PROMPT DE EXPERTO: Le damos un rol de doctor académico
        prompt = (
            f"<|system|>\nEres un médico especialista en diagnóstico diferencial. "
            f"Analiza los síntomas de forma técnica, menciona posibles patologías y "
            f"explica la fisiología detrás de ellos. No digas 've al médico' de inmediato, "
            f"primero ofrece un análisis profundo.\n"
            f"<|user|>\n{pregunta}\n<|assistant|>\n"
        )
        
        output = asistente(prompt, max_new_tokens=300, temperature=0.6, do_sample=True)
        respuesta_en = output[0]["generated_text"].split("<|assistant|>\n")[-1]
        
        st.session_state['respuesta_medica'] = respuesta_en
        st.subheader("⚕️ Análisis Técnico (Inglés):")
        st.write(respuesta_en)

    # Botón de traducción
    if 'respuesta_medica' in st.session_state:
        if st.button("🌍 Traducir Consulta al Español"):
            with st.spinner('Traduciendo informe...'):
                traduccion = GoogleTranslator(source='en', target='es').translate(st.session_state['respuesta_medica'])
                st.subheader("🇪🇸 Traducción al Español:")
                st.write(traduccion)

st.divider()
st.info("Recordatorio: Esta herramienta es para fines de investigación y educación médica.")