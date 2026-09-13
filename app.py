"""
app.py
Interfaz Streamlit del Asistente de Diagnóstico por Síntomas (v2).

Fuente de datos: tablas de "causas de <síntoma>" del Manual MSD (versión
profesional), consultadas en vivo mediante scraping puntual (ver scraper.py).
No se descarga ni almacena el manual completo.

AVISO IMPORTANTE:
Esta aplicación es una herramienta de apoyo y aprendizaje, NO sustituye el
juicio clínico ni un diagnóstico médico profesional. El Manual MSD no publica
una licencia abierta para su contenido; revisa sus términos de uso antes de
un despliegue público o un uso intensivo.
"""

import streamlit as st

from diagnosis_engine import MotorDiagnostico, SintomaNoDisponible, SYMPTOM_PAGES

st.set_page_config(page_title="Asistente de Diagnóstico por Síntomas", page_icon="🩺")

st.title("🩺 Asistente de Diagnóstico por Síntomas")
st.caption("Fuente: tablas de causas del Manual MSD (versión profesional) — consulta en vivo.")

if "motor" not in st.session_state:
    st.session_state.motor = None
    st.session_state.pregunta_actual = None
    st.session_state.resultado = None
    st.session_state.error = None

# --- Paso 1: síntoma inicial ---
if st.session_state.motor is None:
    st.write(f"Síntomas disponibles ahora mismo: *{', '.join(sorted(SYMPTOM_PAGES.keys()))}*")
    sintoma_inicial = st.text_input("Introduce el síntoma inicial (ej. 'dolor toracico'):")
    if st.button("Empezar diagnóstico") and sintoma_inicial.strip():
        try:
            with st.spinner("Consultando la tabla de causas en el Manual MSD..."):
                st.session_state.motor = MotorDiagnostico(sintoma_inicial.strip())
            st.session_state.error = None
        except SintomaNoDisponible as e:
            st.session_state.error = str(e)
        st.rerun()

    if st.session_state.error:
        st.warning(st.session_state.error)

# --- Paso 2: preguntas interactivas ---
else:
    motor = st.session_state.motor

    if not motor.candidatas:
        st.error(
            "No se ha podido extraer ninguna tabla de causas de esta página. "
            "Puede que el Manual MSD haya cambiado el formato: revisa scraper.py."
        )
        if st.button("Volver a empezar"):
            st.session_state.motor = None
            st.rerun()
    else:
        if st.session_state.resultado is None:
            resultado = motor.resultado_final()
            if resultado:
                st.session_state.resultado = resultado
            else:
                pregunta = motor.siguiente_pregunta()
                if pregunta is None:
                    puntuaciones = motor.puntuar_candidatas()
                    st.session_state.resultado = puntuaciones[0][0] if puntuaciones else None
                else:
                    st.session_state.pregunta_actual = pregunta

        if st.session_state.resultado:
            r = st.session_state.resultado
            st.success(f"Causa más probable según la fuente: **{r['titulo']}**")
            if r["url"]:
                st.markdown(f"[Ver ficha completa en el Manual MSD]({r['url']})")
            with st.expander("Hallazgos sugestivos descritos en la fuente"):
                for h in r["hallazgos"]:
                    st.write(f"- {h}")
            with st.expander("Abordaje diagnóstico sugerido"):
                for a in r["abordaje"]:
                    st.write(f"- {a}")
            st.divider()
            st.subheader("Otras causas consideradas")
            for c, score in motor.puntuar_candidatas()[1:4]:
                st.write(f"- {c['titulo']} (puntuación: {score})")
            if st.button("Empezar un nuevo diagnóstico"):
                st.session_state.motor = None
                st.session_state.resultado = None
                st.session_state.pregunta_actual = None
                st.rerun()

        elif st.session_state.pregunta_actual:
            st.write(f"**Hallazgos confirmados:** {len(motor.hallazgos_confirmados)}")
            st.write(f"¿Presenta también: **{st.session_state.pregunta_actual}**?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Sí"):
                    motor.responder(st.session_state.pregunta_actual, True)
                    st.session_state.pregunta_actual = None
                    st.rerun()
            with col2:
                if st.button("No"):
                    motor.responder(st.session_state.pregunta_actual, False)
                    st.session_state.pregunta_actual = None
                    st.rerun()

st.divider()
st.caption(
    "⚠️ Esta app es una herramienta de apoyo educativo y no sustituye el diagnóstico "
    "de un profesional sanitario. Los datos se consultan en vivo desde msdmanuals.com."
)
