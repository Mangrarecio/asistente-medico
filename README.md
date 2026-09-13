# Asistente de Diagnóstico por Síntomas

App interactiva en Streamlit que, partiendo de un síntoma inicial, va haciendo
preguntas de sí/no basadas en los **hallazgos sugestivos curados por el
propio Manual MSD** (versión profesional) y sugiere la causa más probable.

⚠️ **Herramienta de apoyo educativo. No sustituye el diagnóstico ni el juicio
de un profesional sanitario.**

## ⚖️ Aviso legal importante

El Manual MSD es de consulta **gratuita online**, pero **no publica una
licencia abierta** para reutilizar su contenido: su página de
[Permisos](https://www.msdmanuals.com/es/professional/content/permissions)
pide solicitar autorización por correo (`msdmanualpermissions@msd.com`)
para reutilizaciones. Esta app consulta el contenido en vivo, página a
página, sin descargarlo ni redistribuirlo — pero si vas a darle un uso más
amplio que el personal/educativo, escribe primero pidiendo permiso.

## 🧠 Cómo funciona

1. El script `build_index.py` recorre el **sitemap oficial** del Manual MSD
   (`https://www.msdmanuals.com/es/sitemap.ashx`, un XML estático que no
   depende de JavaScript) y localiza todas las páginas de síntoma de la
   versión profesional que tienen una tabla de "Algunas causas de..."
   reconocible. Guarda el resultado en `symptom_index.json`.
2. La app (`app.py`) usa ese índice: cuando escribes un síntoma, busca la
   página correspondiente y extrae su tabla de causas (`scraper.py`):
   causa, hallazgos sugestivos y abordaje diagnóstico.
3. El motor (`diagnosis_engine.py`) usa esos hallazgos —tal cual los
   redacta la fuente— como preguntas de sí/no, y va puntuando las causas
   candidatas según tus respuestas.
4. Cuando una causa destaca claramente, la muestra con enlace a la ficha
   original.

## 📁 Estructura del proyecto

```
asistente_diagnostico_msd/
├── app.py                  # Interfaz Streamlit
├── scraper.py               # Extracción de tablas de causas (scraping puntual)
├── diagnosis_engine.py       # Lógica de preguntas y puntuación
├── build_index.py            # Construye symptom_index.json a partir del sitemap
├── symptom_index.json        # (se genera al ejecutar build_index.py)
├── requirements.txt
└── README.md
```

## 🚀 Puesta en marcha en local

```bash
pip install -r requirements.txt

# Paso único (o periódico, para mantener el índice actualizado):
# recorre el sitemap y construye symptom_index.json.
# Tarda varios minutos porque respeta una pausa entre peticiones.
python build_index.py

streamlit run app.py
```

Si no ejecutas `build_index.py`, la app sigue funcionando con una lista de
respaldo de 5 síntomas (dolor torácico, disnea, palpitaciones, síncope,
dolor abdominal) — suficiente para probarla, pero mucho más limitada.

## 🛠️ Si algo deja de funcionar

Ni `scraper.py` ni `build_index.py` han podido probarse contra el sitio
real desde el entorno donde se generó este código (sin acceso a
msdmanuals.com). Antes de darlos por buenos:

1. **Si `build_index.py` no encuentra ninguna URL candidata:** imprime un
   fragmento del sitemap descargado y comprueba a mano si las URLs de
   síntoma tienen el formato esperado
   (`/es/professional/<categoría>/síntomas-.../<síntoma>`). Ajusta
   `PATRON_SINTOMA` en `build_index.py` si el formato real difiere.
2. **Si una página de síntoma concreta no devuelve ninguna causa:** puede
   que su tabla no tenga exactamente 3 columnas, o que use una estructura
   distinta. Ajusta `obtener_tabla_causas()` en `scraper.py` tras
   inspeccionar el HTML real de esa página.
3. **Si el sitemap ha cambiado de sitio o de formato:** revisa que
   `https://www.msdmanuals.com/es/sitemap.ashx` siga respondiendo; si el
   sitio decide dividirlo en varios sitemaps (un índice con `<sitemap>` en
   vez de `<url>` directamente), habría que descargar cada uno.

## ☁️ Subir a GitHub

```bash
cd asistente_diagnostico_msd
git init
git add .
git commit -m "Asistente de diagnóstico: índice completo vía sitemap del Manual MSD"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/asistente-diagnostico-msd.git
git push -u origin main
```

**Nota:** no subas `symptom_index.json` si prefieres que cada persona lo
genere con sus propios permisos/uso, o súbelo si quieres compartir el
índice ya construido — es tu decisión. Añade un `.gitignore` con
`symptom_index.json` si prefieres lo primero.

## 🌐 Desplegar gratis en Streamlit Community Cloud

1. Entra en [share.streamlit.io](https://share.streamlit.io) con tu cuenta
   de GitHub.
2. "New app" → selecciona el repositorio.
3. Archivo principal: `app.py`.
4. Si no subiste `symptom_index.json`, tendrás que ejecutar
   `build_index.py` en algún sitio con acceso de red completo (tu propio
   ordenador) y subir el archivo resultante al repo, ya que Streamlit
   Community Cloud puede tener restricciones de red para procesos largos
   de scraping en el arranque.

## 💡 Posibles mejoras futuras

- Ejecutar `build_index.py` en GitHub Actions de forma periódica (ej.
  mensual) para mantener `symptom_index.json` actualizado automáticamente.
- Sustituir o combinar la fuente MSD con MedlinePlus (licencia más clara).
- Mejorar la comparación de hallazgos con lematización en español (ej.
  `spaCy`) en vez de solapamiento de palabras.
- Permitir seleccionar el síntoma de una lista desplegable (autocompletado)
  en vez de solo texto libre, ahora que el índice es grande.
