"""Ficha de etimología, referencias del mundo real y resumen de Wikipedia para cada entidad.

Se guarda como nodo :Etimologia asociado por entidad_id (sin relación, para sobrevivir a un re-crawl).
Los campos de origen del nombre/mito los escribe el administrador; el resumen de Wikipedia real
(es.wikipedia.org / en.wikipedia.org) se busca mediante la API oficial de MediaWiki/REST y se guarda
en Neo4j para consultarlo directamente sin tener que consultar Wikipedia cada vez.
"""
import json
from datetime import datetime, timezone
from urllib.parse import quote

import requests

CAMPOS_TEXTO = {
    "origen_nombre": 3000,
    "idioma_origen": 100,
    "referencia_real": 1000,
    "cultura": 200,
    "resumen_wiki": 3000,
    "wikipedia_titulo": 300,
    "wikipedia_resumen": 3000,
    "wikipedia_url": 500,
    "wikipedia_idioma": 10,
}
MAX_REFERENCIAS = 10
WIKIPEDIA_HEADERS = {"User-Agent": "ArchivoIrminsulGenshinLore/1.0 (https://github.com/MZapicoCabeza/genshin-lore)"}


class ErrorValidacion(ValueError):
    pass


def _consultar_resumen_wikipedia(titulo: str, lang: str) -> dict | None:
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(titulo.replace(' ', '_'), safe='')}"
    try:
        r = requests.get(url, headers=WIKIPEDIA_HEADERS, timeout=8)
        if r.status_code != 200:
            return None
        d = r.json()
        if d.get("type") == "disambiguation" or not d.get("extract"):
            return None
        return {
            "wikipedia_titulo": (d.get("title") or titulo)[:300],
            "wikipedia_resumen": (d.get("extract") or "").strip()[:3000],
            "wikipedia_url": (d.get("content_urls", {}).get("desktop", {}).get("page") or "")[:500],
            "wikipedia_idioma": lang,
        }
    except Exception:
        return None


def _buscar_titulo_wikipedia(query: str, lang: str) -> str | None:
    try:
        r = requests.get(
            f"https://{lang}.wikipedia.org/w/api.php",
            params={"action": "opensearch", "search": query, "limit": 4, "namespace": 0, "format": "json"},
            headers=WIKIPEDIA_HEADERS,
            timeout=8,
        )
        if r.status_code != 200:
            return None
        datos = r.json()
        titulos = datos[1] if isinstance(datos, list) and len(datos) > 1 else []
        return titulos[0] if titulos else None
    except Exception:
        return None


def buscar_en_wikipedia(termino: str) -> dict | None:
    """Busca en Wikipedia ES (y fallback EN) devolviendo título, resumen y enlace oficial."""
    q = (termino or "").strip()
    if not q:
        return None
    for lang in ("es", "en"):
        res = _consultar_resumen_wikipedia(q, lang)
        if res:
            return res
    for lang in ("es", "en"):
        t = _buscar_titulo_wikipedia(q, lang)
        if t:
            res = _consultar_resumen_wikipedia(t, lang)
            if res:
                return res
    return None


def guardar_wikipedia(driver, entidad_id: str, wiki_data: dict) -> dict:
    """Guarda el resumen de Wikipedia en la ficha :Etimologia de la entidad dentro de Neo4j."""
    campos = {
        "wikipedia_titulo": (wiki_data.get("wikipedia_titulo") or "")[:300],
        "wikipedia_resumen": (wiki_data.get("wikipedia_resumen") or "")[:3000],
        "wikipedia_url": (wiki_data.get("wikipedia_url") or "")[:500],
        "wikipedia_idioma": (wiki_data.get("wikipedia_idioma") or "")[:10],
        "fecha_actualizacion": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with driver.session() as s:
        rec = s.run(
            """
            MATCH (e:Entity {id: $id})
            MERGE (n:Etimologia {entidad_id: $id})
            SET n += $campos
            RETURN n
            """,
            id=entidad_id,
            campos=campos,
        ).single()
    if not rec:
        raise LookupError(f"No existe la entidad '{entidad_id}'")
    return _a_dict(rec["n"])


def validar(datos: dict) -> dict:
    limpio = {}
    for campo, max_len in CAMPOS_TEXTO.items():
        limpio[campo] = (datos.get(campo) or "").strip()[:max_len]

    if limpio.get("wikipedia_url") and not limpio["wikipedia_url"].lower().startswith(("http://", "https://")):
        raise ErrorValidacion("La URL de Wikipedia debe empezar por http:// o https://")

    refs = []
    for r in (datos.get("referencias") or [])[:MAX_REFERENCIAS]:
        if not isinstance(r, dict):
            continue
        titulo = (r.get("titulo") or "").strip()[:200]
        url = (r.get("url") or "").strip()[:500]
        if url and not url.lower().startswith(("http://", "https://")):
            raise ErrorValidacion("Las URLs de referencias deben empezar por http:// o https://")
        if titulo or url:
            refs.append({"titulo": titulo, "url": url})
    limpio["referencias_json"] = json.dumps(refs, ensure_ascii=False)
    limpio["resumen_auto"] = bool(datos.get("resumen_auto"))

    if not any(limpio[c] for c in CAMPOS_TEXTO) and not refs:
        raise ErrorValidacion("La ficha no puede estar vacía")
    return limpio


def _a_dict(n) -> dict:
    try:
        refs = json.loads(n.get("referencias_json") or "[]")
    except ValueError:
        refs = []
    d = {c: n.get(c) or "" for c in CAMPOS_TEXTO}
    d.update(
        entidad_id=n.get("entidad_id"),
        referencias=refs,
        resumen_auto=bool(n.get("resumen_auto")),
        fecha_actualizacion=n.get("fecha_actualizacion"),
    )
    return d


def obtener(driver, entidad_id: str) -> dict | None:
    with driver.session() as s:
        rec = s.run("MATCH (n:Etimologia {entidad_id: $id}) RETURN n", id=entidad_id).single()
    return _a_dict(rec["n"]) if rec else None


def guardar(driver, entidad_id: str, datos: dict) -> dict:
    campos = validar(datos)
    campos["fecha_actualizacion"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with driver.session() as s:
        rec = s.run(
            """
            MATCH (e:Entity {id: $id})
            MERGE (n:Etimologia {entidad_id: $id})
            SET n += $campos
            RETURN n
            """,
            id=entidad_id,
            campos=campos,
        ).single()
    if not rec:
        raise LookupError(f"No existe la entidad '{entidad_id}'")
    return _a_dict(rec["n"])


def borrar(driver, entidad_id: str) -> bool:
    with driver.session() as s:
        rec = s.run(
            "MATCH (n:Etimologia {entidad_id: $id}) DETACH DELETE n RETURN count(n) AS c", id=entidad_id
        ).single()
    return bool(rec and rec["c"])


def asegurar_indices(driver) -> None:
    with driver.session() as s:
        s.run("CREATE CONSTRAINT etimologia_entidad IF NOT EXISTS FOR (n:Etimologia) REQUIRE n.entidad_id IS UNIQUE")
