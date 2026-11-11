"""Capa de procedencia y certeza para el lore.

Cada dato que NO viene directamente de la Wiki (vídeos, Reddit, notas propias...) se guarda como un
nodo :Afirmacion asociado a la entidad por su id (propiedad entidad_id, sin relación, para que sobreviva a un re-crawl), con fuente, fecha, versión del juego y nivel de certeza.
El texto de la Wiki (canon) nunca se sobrescribe: las afirmaciones son una capa aparte.
"""
import uuid
from datetime import datetime, timezone

CERTEZAS = ("canon", "inferido", "teoria")
ESTADOS = ("vigente", "en_revision", "refutada")
TIPOS_FUENTE = ("wiki", "juego", "youtube", "reddit", "nota_propia", "otro")


class ErrorValidacion(ValueError):
    pass


def _limpiar(valor, max_len: int) -> str:
    return (valor or "").strip()[:max_len]


def validar_afirmacion(datos: dict, parcial: bool = False) -> dict:
    """Valida y normaliza los campos de una afirmación. Con parcial=True solo valida los presentes."""
    limpio: dict = {}
    campos_texto = {
        "texto": 2000,
        "fuente_titulo": 300,
        "fuente_url": 500,
        "fuente_ref": 100,
        "version_juego": 20,
        "nota": 1000,
    }
    for campo, max_len in campos_texto.items():
        if campo in datos:
            limpio[campo] = _limpiar(datos[campo], max_len)

    for campo, validos in (("certeza", CERTEZAS), ("estado", ESTADOS), ("tipo_fuente", TIPOS_FUENTE)):
        if campo in datos:
            v = _limpiar(datos[campo], 30).lower()
            if v not in validos:
                raise ErrorValidacion(f"'{campo}' debe ser uno de: {', '.join(validos)}")
            limpio[campo] = v

    if "fuente_url" in limpio and limpio["fuente_url"] and not limpio["fuente_url"].lower().startswith(("http://", "https://")):
        raise ErrorValidacion("'fuente_url' debe empezar por http:// o https://")

    if not parcial:
        if not limpio.get("texto"):
            raise ErrorValidacion("'texto' es obligatorio")
        limpio.setdefault("certeza", "teoria")
        limpio.setdefault("estado", "vigente")
        limpio.setdefault("tipo_fuente", "otro")
        for campo in campos_texto:
            limpio.setdefault(campo, "")
    elif "texto" in limpio and not limpio["texto"]:
        raise ErrorValidacion("'texto' no puede quedar vacío")
    return limpio


def _registro_a_dict(a) -> dict:
    return {k: a.get(k) for k in (
        "id", "entidad_id", "texto", "certeza", "estado", "tipo_fuente", "fuente_titulo",
        "fuente_url", "fuente_ref", "version_juego", "nota", "fecha_captura", "fecha_actualizacion",
    )}


def listar_afirmaciones(driver, entidad_id: str) -> list[dict]:
    with driver.session() as s:
        res = s.run(
            """
            MATCH (a:Afirmacion {entidad_id: $id})
            RETURN a ORDER BY a.fecha_captura DESC
            """,
            id=entidad_id,
        )
        return [_registro_a_dict(r["a"]) for r in res]


def crear_afirmacion(driver, entidad_id: str, datos: dict) -> dict:
    campos = validar_afirmacion(datos)
    ahora = datetime.now(timezone.utc).isoformat(timespec="seconds")
    campos.update(id=uuid.uuid4().hex, entidad_id=entidad_id, fecha_captura=ahora, fecha_actualizacion=ahora)
    with driver.session() as s:
        rec = s.run(
            """
            MATCH (e:Entity {id: $entidad_id})
            CREATE (a:Afirmacion)
            SET a = $campos
            RETURN a
            """,
            entidad_id=entidad_id,
            campos=campos,
        ).single()
    if not rec:
        raise LookupError(f"No existe la entidad '{entidad_id}'")
    return _registro_a_dict(rec["a"])


def actualizar_afirmacion(driver, afirmacion_id: str, datos: dict) -> dict:
    campos = validar_afirmacion(datos, parcial=True)
    if not campos:
        raise ErrorValidacion("No hay campos que actualizar")
    campos["fecha_actualizacion"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with driver.session() as s:
        rec = s.run(
            "MATCH (a:Afirmacion {id: $id}) SET a += $campos RETURN a",
            id=afirmacion_id,
            campos=campos,
        ).single()
    if not rec:
        raise LookupError("Afirmación no encontrada")
    return _registro_a_dict(rec["a"])


def borrar_afirmacion(driver, afirmacion_id: str) -> bool:
    with driver.session() as s:
        rec = s.run(
            "MATCH (a:Afirmacion {id: $id}) DETACH DELETE a RETURN count(a) AS n",
            id=afirmacion_id,
        ).single()
    return bool(rec and rec["n"])


def asegurar_indices(driver) -> None:
    with driver.session() as s:
        s.run("CREATE CONSTRAINT afirmacion_id IF NOT EXISTS FOR (a:Afirmacion) REQUIRE a.id IS UNIQUE")
