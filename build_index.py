"""
build_index.py
Script de construcción del índice de síntomas del Manual MSD.

Recorre el sitemap oficial del sitio (un archivo XML estático que NO
depende de JavaScript, a diferencia de la búsqueda del propio sitio) para
descubrir todas las páginas de síntoma de la versión profesional, comprueba
cuáles tienen una tabla de "causas" reconocible (ver scraper.obtener_tabla_causas)
y guarda el resultado en symptom_index.json.

IMPORTANTE:
- Este script se ejecuta UNA VEZ (o de forma periódica para actualizar el
  índice), no en cada consulta del usuario. La propia app (app.py) solo LEE
  el archivo symptom_index.json que este script genera.
- Tarda varios minutos porque respeta una pausa de cortesía entre peticiones
  (ver scraper.REQUEST_DELAY) y puede recorrer varios cientos de páginas.
- No ha sido posible ejecutar ni probar este script contra el sitio real
  desde este entorno de desarrollo (sin acceso a msdmanuals.com). Dos cosas
  a revisar la primera vez que lo ejecutes:
    1. Si el patrón PATRON_SINTOMA no encuentra URLs, imprime alguna URL de
       ejemplo del sitemap (con un print) y ajusta la expresión regular.
    2. El sitemap puede venir en varias partes (un índice de sitemaps con
       <sitemap><loc>...</loc></sitemap> en vez de <url><loc>...</loc></url>
       directamente) si el sitio decide paginarlo en el futuro; en ese caso
       habría que descargar cada sub-sitemap listado.

Uso:
    python build_index.py
"""

import json
import re
import time
from urllib.parse import unquote

import requests
from bs4 import BeautifulSoup

import scraper

SITEMAP_URL = "https://www.msdmanuals.com/es/sitemap.ashx"
OUTPUT_FILE = "symptom_index.json"

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


def construir_indice():
    print("Descargando sitemap...")
    xml_text = descargar_sitemap()
    urls = extraer_urls_sintomas(xml_text)
    print(f"Encontradas {len(urls)} páginas candidatas de síntoma.")

    if not urls:
        print(
            "AVISO: no se encontró ninguna URL con el patrón esperado. "
            "Revisa PATRON_SINTOMA en este script contra el contenido real "
            "del sitemap (imprime xml_text o busca 'síntomas' a mano)."
        )
        return

    indice = {}
    for i, url in enumerate(urls, 1):
        clave = slug_a_clave(url)
        print(f"[{i}/{len(urls)}] {clave} -> {url}")
        try:
            filas = scraper.obtener_tabla_causas(url)
        except Exception as e:
            print(f"   ! error al procesar esta página: {e}")
            continue

        if filas:
            indice[clave] = url
        else:
            print("   (sin tabla de causas reconocible; se omite)")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(indice, f, ensure_ascii=False, indent=2)

    print(f"\nÍndice guardado en {OUTPUT_FILE} con {len(indice)} síntomas.")


if __name__ == "__main__":
    construir_indice()
