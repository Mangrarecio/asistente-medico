"""
build_index.py
Construcción del índice de síntomas del Manual MSD.

Recorre el sitemap oficial del sitio (un archivo XML estático que NO
depende de JavaScript, a diferencia de la búsqueda del propio sitio) para
descubrir todas las páginas de síntoma de la versión profesional, comprueba
cuáles tienen una tabla de "causas" reconocible (ver scraper.obtener_tabla_causas)
y guarda el resultado en symptom_index.json.

Este módulo se puede usar de dos formas:
1. Como script de línea de comandos: `python build_index.py`
2. Importado desde app.py, usando construir_indice_generador() para mostrar
   progreso dentro de la interfaz de Streamlit (útil si despliegas en
   Streamlit Cloud, donde no puedes ejecutar un script suelto por separado).

IMPORTANTE:
- Este proceso tarda varios minutos porque respeta una pausa de cortesía
  entre peticiones (ver scraper.REQUEST_DELAY) y puede recorrer varios
  cientos de páginas.
- No ha sido posible ejecutar ni probar este script contra el sitio real
  desde el entorno donde se escribió este código (sin acceso a
  msdmanuals.com). Si PATRON_SINTOMA no encuentra ninguna URL, imprime un
  fragmento del sitemap descargado y ajusta la expresión regular.
"""

import json
import os
import re
from urllib.parse import unquote

import requests
from bs4 import BeautifulSoup

import scraper

SITEMAP_URL = "https://www.msdmanuals.com/es/sitemap.ashx"
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "symptom_index.json")

# Solo nos interesan páginas de la versión profesional cuya carpeta
# contenga "síntomas" (donde viven las tablas de causas por síntoma), por
# ejemplo:
#   /es/professional/trastornos-cardiovasculares/síntomas-de-las-enfermedades-cardiovasculares/dolor-torácico
PATRON_SINTOMA = re.compile(r"/es/professional/[^/]+/s[ií]ntomas[^/]*/([^/]+)/?$")

# Páginas de introducción al capítulo (no tienen tabla de causas, son solo
# el índice del capítulo de síntomas de esa especialidad).
PREFIJOS_A_EXCLUIR = ("introducción-a-", "introduccion-a-")


def descargar_sitemap():
    resp = requests.get(SITEMAP_URL, headers=scraper.HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.text


def extraer_urls_sintomas(xml_text):
    soup = BeautifulSoup(xml_text, "xml")
    urls = []
    for loc in soup.find_all("loc"):
        url = loc.get_text(strip=True)
        m = PATRON_SINTOMA.search(url)
        if not m:
            continue
        slug = unquote(m.group(1))
        if slug.startswith(PREFIJOS_A_EXCLUIR):
            continue
        urls.append(url)
    return sorted(set(urls))


def slug_a_clave(url):
    """Convierte el último segmento de la URL en una clave de síntoma legible."""
    slug = unquote(url.rstrip("/").split("/")[-1])
    return slug.replace("-", " ")


def construir_indice_generador():
    """
    Generador que hace todo el trabajo de construir el índice, produciendo
    (paso_actual, total_pasos, mensaje) en cada avance. Al terminar, dentro
    de mensaje viene también el número final de síntomas guardados.

    Se usa tanto desde la CLI (construir_indice) como desde app.py con una
    barra de progreso de Streamlit.
    """
    yield (0, 1, "Descargando el sitemap del Manual MSD...")
    xml_text = descargar_sitemap()
    urls = extraer_urls_sintomas(xml_text)
    total = len(urls)
    yield (0, max(total, 1), f"Encontradas {total} páginas candidatas de síntoma.")

    if not urls:
        yield (
            1,
            1,
            "AVISO: no se encontró ninguna URL con el patrón esperado. "
            "Puede que el Manual MSD haya cambiado el formato de sus URLs; "
            "revisa PATRON_SINTOMA en build_index.py.",
        )
        return

    indice = {}
    for i, url in enumerate(urls, 1):
        clave = slug_a_clave(url)
        try:
            filas = scraper.obtener_tabla_causas(url)
        except Exception as e:
            yield (i, total, f"[{i}/{total}] {clave}: error al procesar ({e})")
            continue

        if filas:
            indice[clave] = url
            yield (i, total, f"[{i}/{total}] {clave}: OK ({len(filas)} causas)")
        else:
            yield (i, total, f"[{i}/{total}] {clave}: sin tabla de causas, se omite")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(indice, f, ensure_ascii=False, indent=2)

    yield (total, total, f"Listo. Índice guardado con {len(indice)} síntomas.")


def construir_indice():
    """Versión de línea de comandos: imprime el progreso en la terminal."""
    for _, _, mensaje in construir_indice_generador():
        print(mensaje)


if __name__ == "__main__":
    construir_indice()
