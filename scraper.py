"""
scraper.py (v2)
Módulo de scraping puntual y en vivo contra el Manual MSD (versión profesional).

CAMBIO CLAVE respecto a la v1:
Muchas páginas de síntoma del Manual MSD (ej. "Dolor torácico", "Disnea",
"Síncope") incluyen una tabla ya curada del tipo "Algunas causas de <síntoma>"
con columnas Causa / Hallazgos sugestivos / Abordaje diagnóstico. En vez de
comparar texto libre entre páginas, extraemos esa tabla directamente: es un
dato mucho más limpio y clínicamente más útil para hacer preguntas de sí/no.

IMPORTANTE (léase antes de desplegar):
- Este scraper consulta únicamente la página de síntoma necesaria en cada
  momento, no descarga el manual completo.
- El Manual MSD no publica una licencia abierta para su contenido y su
  página de "Permisos" (https://www.msdmanuals.com/es/professional/content/permissions)
  pide explícitamente solicitar autorización por correo para reutilizar
  contenido. Este código está pensado para uso personal/educativo; si vas a
  darle un uso más amplio, escribe primero a msdmanualpermissions@msd.com.
- Los selectores de tabla están escritos de forma robusta (buscan cualquier
  <table> con al menos 3 columnas y descartan filas de subgrupo vacías),
  pero no ha sido posible probarlos contra el HTML real en vivo desde este
  entorno de desarrollo. Pruébalos en tu máquina; si una tabla no se
  reconoce bien, ajusta obtener_tabla_causas().
"""

import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.msdmanuals.com"
HEADERS = {
    "User-Agent": "AsistenteDiagnosticoTES/1.0 (uso personal y educativo)"
}
REQUEST_DELAY = 1.5  # segundos entre peticiones, para no saturar el servidor

_page_cache = {}


def _get(url):
    """Descarga una URL con caché simple y una pausa de cortesía entre peticiones."""
    if url in _page_cache:
        return _page_cache[url]
    time.sleep(REQUEST_DELAY)
    response = requests.get(url, headers=HEADERS, timeout=10)
    response.raise_for_status()
    _page_cache[url] = response.text
    return response.text


def _dividir_por_saltos(celda):
    """
    Divide el contenido de una celda de tabla en una lista de líneas,
    usando los <br> como separador (los hallazgos suelen venir así, uno
    por línea, dentro de la misma celda).
    """
    for br in celda.find_all("br"):
        br.replace_with("\n")
    texto = celda.get_text()
    return [linea.strip() for linea in texto.split("\n") if linea.strip()]


def obtener_tabla_causas(url):
    """
    Extrae la tabla de causas de una página de síntoma del Manual MSD
    (del tipo "Algunas causas de <síntoma>").

    Devuelve una lista de dicts:
    [{"causa": str, "url_causa": str|None, "hallazgos": [str, ...], "abordaje": [str, ...]}, ...]

    Devuelve lista vacía si no encuentra ninguna tabla reconocible en la página.
    """
    html = _get(url)
    soup = BeautifulSoup(html, "html.parser")
    resultados = []

    for tabla in soup.find_all("table"):
        filas = tabla.find_all("tr")
        if len(filas) < 2:
            continue

        primera_fila_celdas = filas[0].find_all(["th", "td"])
        if len(primera_fila_celdas) < 3:
            continue  # no parece la tabla de 3 columnas que buscamos

        for fila in filas[1:]:
            celdas = fila.find_all("td")
            if len(celdas) < 3:
                continue

            celda_causa, celda_hallazgos, celda_abordaje = celdas[0], celdas[1], celdas[2]
            hallazgos = _dividir_por_saltos(celda_hallazgos)
            abordaje = _dividir_por_saltos(celda_abordaje)

            if not hallazgos and not abordaje:
                # fila de encabezado de subgrupo (ej. "Cardiovascular"), sin datos propios
                continue

            causa_texto = celda_causa.get_text(" ", strip=True)
            if not causa_texto:
                continue

            enlace = celda_causa.find("a")
            causa_url = urljoin(BASE_URL, enlace["href"]) if enlace else None

            resultados.append(
                {
                    "causa": causa_texto,
                    "url_causa": causa_url,
                    "hallazgos": hallazgos,
                    "abordaje": abordaje,
                }
            )

    return resultados
