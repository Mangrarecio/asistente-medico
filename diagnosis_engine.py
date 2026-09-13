"""
diagnosis_engine.py (v2)
Motor de diagnóstico basado en las tablas de "causas de <síntoma>" del
Manual MSD, en vez de comparación de texto libre. Cada enfermedad candidata
trae ya su lista de "hallazgos sugestivos" curados por la fuente, que se
usan directamente como preguntas de sí/no.
"""

import difflib
import json
import os
import unicodedata

import scraper

# Diccionario de síntomas de RESPALDO (se usa si aún no has ejecutado
# build_index.py, o para los síntomas que ese script no haya podido
# clasificar). Una vez generes symptom_index.json, ese archivo tiene
# prioridad y amplía esta lista automáticamente con todo lo encontrado
# en el sitemap del Manual MSD.
SYMPTOM_PAGES_RESPALDO = {
    "dolor toracico": "https://www.msdmanuals.com/es/professional/trastornos-cardiovasculares/s%C3%ADntomas-de-las-enfermedades-cardiovasculares/dolor-tor%C3%A1cico",
    "disnea": "https://www.msdmanuals.com/es/professional/trastornos-pulmonares/s%C3%ADntomas-de-los-trastornos-pulmonares/disnea",
    "palpitaciones": "https://www.msdmanuals.com/es/professional/trastornos-cardiovasculares/s%C3%ADntomas-de-las-enfermedades-cardiovasculares/palpitaciones",
    "sincope": "https://www.msdmanuals.com/es/professional/trastornos-cardiovasculares/s%C3%ADntomas-de-las-enfermedades-cardiovasculares/s%C3%ADncope",
    "dolor abdominal": "https://www.msdmanuals.com/es/professional/trastornos-gastrointestinales/abdomen-agudo-y-gastroenterolog%C3%ADa-quir%C3%BArgica/dolor-abdominal-agudo",
}

_RUTA_INDICE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "symptom_index.json")


def _cargar_symptom_pages():
    """
    Combina la lista de respaldo con el índice completo generado por
    build_index.py (si existe). El índice generado tiene prioridad.
    """
    paginas = dict(SYMPTOM_PAGES_RESPALDO)
    if os.path.exists(_RUTA_INDICE):
        try:
            with open(_RUTA_INDICE, encoding="utf-8") as f:
                paginas.update(json.load(f))
        except (json.JSONDecodeError, OSError):
            pass  # si el archivo está corrupto, seguimos solo con el respaldo
    return paginas


SYMPTOM_PAGES = _cargar_symptom_pages()


def _normalizar(texto):
    """minusculas + sin acentos, para comparar de forma tolerante."""
    texto = texto.lower().strip()
    texto = "".join(
        c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn"
    )
    return texto


def buscar_url_por_sintoma(sintoma_usuario):
    """Empareja el síntoma escrito por el usuario con una entrada de SYMPTOM_PAGES."""
    clave = _normalizar(sintoma_usuario)
    if clave in SYMPTOM_PAGES:
        return SYMPTOM_PAGES[clave]
    coincidencias = difflib.get_close_matches(clave, SYMPTOM_PAGES.keys(), n=1, cutoff=0.6)
    if coincidencias:
        return SYMPTOM_PAGES[coincidencias[0]]
    return None


class SintomaNoDisponible(Exception):
    pass


class MotorDiagnostico:
    def __init__(self, sintoma_inicial):
        self.url_sintoma = buscar_url_por_sintoma(sintoma_inicial)
        if not self.url_sintoma:
            disponibles = ", ".join(sorted(SYMPTOM_PAGES.keys()))
            raise SintomaNoDisponible(
                f"No tengo registrada una página de síntoma para '{sintoma_inicial}'. "
                f"Síntomas disponibles ahora mismo: {disponibles}."
            )

        filas = scraper.obtener_tabla_causas(self.url_sintoma)
        self.candidatas = [
            {
                "titulo": f["causa"],
                "url": f["url_causa"] or self.url_sintoma,
                "hallazgos": f["hallazgos"],
                "abordaje": f["abordaje"],
            }
            for f in filas
        ]
        self.hallazgos_confirmados = set()
        self.hallazgos_descartados = set()

    def _hallazgo_presente_en(self, hallazgo, candidata):
        """
        Comparación flexible por solapamiento de palabras clave: no exigimos
        coincidencia exacta de la frase porque el usuario puede confirmar un
        hallazgo con matices distintos a como está redactado en la tabla.
        """
        palabras_h = set(_normalizar(hallazgo).split())
        for h in candidata["hallazgos"]:
            palabras_c = set(_normalizar(h).split())
            comunes = palabras_h & palabras_c
            if len(comunes) >= max(2, len(palabras_h) // 2):
                return True
        return False

    def puntuar_candidatas(self):
        puntuaciones = []
        for c in self.candidatas:
            score = 0
            for h in self.hallazgos_confirmados:
                if self._hallazgo_presente_en(h, c):
                    score += 1
            for h in self.hallazgos_descartados:
                if self._hallazgo_presente_en(h, c):
                    score -= 1
            puntuaciones.append((c, score))
        puntuaciones.sort(key=lambda x: x[1], reverse=True)
        return puntuaciones

    def siguiente_pregunta(self):
        """
        Elige un hallazgo LITERAL (tal cual redactado en la fuente) que
        distinga mejor entre las candidatas mejor puntuadas actualmente.
        """
        puntuaciones = self.puntuar_candidatas()
        top = [c for c, score in puntuaciones[:4]]
        if len(top) < 2:
            return None

        ya_preguntados = self.hallazgos_confirmados | self.hallazgos_descartados
        conteo = {}
        for c in top:
            for h in c["hallazgos"]:
                if h in ya_preguntados:
                    continue
                conteo[h] = conteo.get(h, 0) + 1

        candidatos_pregunta = [(h, n) for h, n in conteo.items() if 0 < n < len(top)]
        if not candidatos_pregunta:
            return None

        # Preferimos el hallazgo que divide el grupo de candidatas de forma
        # más equilibrada (más informativo que uno que solo aísla una).
        candidatos_pregunta.sort(key=lambda x: abs(x[1] - len(top) / 2))
        return candidatos_pregunta[0][0]

    def responder(self, hallazgo, presente):
        if presente:
            self.hallazgos_confirmados.add(hallazgo)
        else:
            self.hallazgos_descartados.add(hallazgo)

    def resultado_final(self, umbral_diferencia=2, minimo_preguntas=2):
        if len(self.hallazgos_confirmados) + len(self.hallazgos_descartados) < minimo_preguntas:
            return None
        puntuaciones = self.puntuar_candidatas()
        if not puntuaciones:
            return None
        mejor, score_mejor = puntuaciones[0]
        if len(puntuaciones) > 1:
            _, score_segundo = puntuaciones[1]
            if score_mejor - score_segundo < umbral_diferencia:
                return None
        return mejor
