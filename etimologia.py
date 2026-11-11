"""Ficha de etimología y referencias del mundo real para cada entidad.

Se guarda como nodo :Etimologia asociado por entidad_id (sin relación, para sobrevivir a un re-crawl).
Los campos de origen del nombre/mito los escribe el administrador; el resumen de la Wiki puede
generarse como borrador con un LLM y queda marcado como automático hasta que se edite a mano.
"""
import json
from datetime import datetime, timezone

CAMPOS_TEXTO = {
    "origen_nombre": 3000,
    "idioma_origen": 100,
    "referencia_real": 1000,
    "cultura": 200,
    "resumen_wiki": 3000,
}
MAX_REFERENCIAS = 10


class ErrorValidacion(ValueError):
    pass


def validar(datos: dict) -> dict:
    limpio = {}
    for campo, max_len in CAMPOS_TEXTO.items():
        limpio[campo] = (datos.get(campo) or "").strip()[:max_len]

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
