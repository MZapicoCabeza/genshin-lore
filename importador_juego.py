"""Importador de textos canon del juego (libros, historias de armas y artefactos) desde Project Amber.

Uso:
    python importador_juego.py                      # importa todo
    python importador_juego.py --tipos libros       # solo libros (libros,armas,artefactos)
    python importador_juego.py --limite 20          # prueba rápida con 20 elementos por tipo

Guarda nodos :TextoJuego en Neo4j (idempotente, MERGE por id), en inglés (el idioma original, que
coincide con los títulos de la Wiki). Se asocian a entidades de dos formas, ambas por propiedad
(sobreviven a un re-crawl):
  - entidad_id: si el título del texto coincide con una entidad.
  - menciona:   entidades cuyo nombre aparece literalmente en el texto (heurística conservadora).
"""
import argparse
import asyncio
import re
from datetime import datetime, timezone

import ambr
from loguru import logger

import crawler_entidades
import enlaces_juego
CONCURRENCIA = 8
MIN_LONG_NOMBRE_MENCION = 6
FUENTE_URL = "https://gi.yatta.moe"
# Los changelogs de Amber empiezan en la 3.0: lo que no aparece en ellos es anterior.
VERSION_ANTERIOR = "<3.0"


def construir_regex_menciones(nombres: list[str]):
    """Regex que detecta nombres de entidad completos (sensible a mayúsculas, con límites de palabra)."""
    candidatos = sorted(
        {n for n in nombres if len(n) >= MIN_LONG_NOMBRE_MENCION and n[0].isupper()}, key=len, reverse=True
    )
    if not candidatos:
        return None
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(n) for n in candidatos) + r")(?!\w)")


def detectar_menciones(regex, texto: str, propio: str = "") -> list[str]:
    if not regex or not texto:
        return []
    return sorted({m for m in regex.findall(texto) if m.lower() != propio.lower()})


def mapa_versiones(changelogs) -> dict[tuple[str, str], str]:
    """(categoría, id) -> primera versión estable en la que apareció el elemento."""
    mapa: dict[tuple[str, str], str] = {}
    for ch in sorted(changelogs, key=lambda c: c.id):
        if getattr(ch, "beta", False):
            continue
        for it in ch.items:
            for i in it.ids:
                mapa.setdefault((it.category, str(i)), ch.version)
    return mapa


def limpiar(texto: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", (texto or "").replace("\r", "")).strip()


async def _acotar(sem, coro):
    async with sem:
        return await coro


async def recolectar_libros(api, versiones, limite):
    sem = asyncio.Semaphore(CONCURRENCIA)
    libros = (await api.fetch_books())[:limite]

    async def uno(libro):
        det = await api.fetch_book_detail(libro.id)
        textos = []
        for vol in det.volumes:
            try:
                cuerpo = limpiar(await api.fetch_readable(f"Book{vol.story_id}"))
            except Exception as e:
                logger.warning(f"Libro {vol.name}: sin texto ({e})")
                continue
            if cuerpo:
                textos.append({
                    "id": f"libro:{vol.story_id}",
                    "tipo": "libro",
                    "titulo": vol.name,
                    "titulo_obra": det.name,
                    "descripcion": limpiar(vol.description),
                    "texto": cuerpo,
                    "version_juego": versiones.get(("book", str(libro.id)), VERSION_ANTERIOR),
                })
        return textos

    res = await asyncio.gather(*[_acotar(sem, uno(l)) for l in libros], return_exceptions=True)
    return [t for r in res if isinstance(r, list) for t in r]


async def recolectar_armas(api, versiones, limite):
    sem = asyncio.Semaphore(CONCURRENCIA)
    armas = (await api.fetch_weapons())[:limite]

    async def uno(arma):
        det = await api.fetch_weapon_detail(arma.id)
        try:
            cuerpo = limpiar(await api.fetch_readable(f"Weapon{det.story_id}"))
        except Exception:
            cuerpo = ""
        if not cuerpo:
            return []
        return [{
            "id": f"arma:{det.id}",
            "tipo": "arma",
            "titulo": det.route or det.name,
            "titulo_obra": det.route or det.name,
            "descripcion": limpiar(det.description),
            "texto": cuerpo,
            "version_juego": versiones.get(("weapon", str(det.id)), VERSION_ANTERIOR),
        }]

    res = await asyncio.gather(*[_acotar(sem, uno(a)) for a in armas], return_exceptions=True)
    return [t for r in res if isinstance(r, list) for t in r]


async def recolectar_artefactos(api, versiones, limite):
    sem = asyncio.Semaphore(CONCURRENCIA)
    sets = (await api.fetch_artifact_sets())[:limite]

    async def uno(s):
        det = await api.fetch_artifact_set_detail(s.id, fetch_story=True)
        nombre_set = det.route or det.name
        out = []
        for p in det.artifacts:
            cuerpo = limpiar(p.story or "")
            if not cuerpo:
                continue
            out.append({
                "id": f"artefacto:{det.id}:{p.pos}",
                "tipo": "artefacto",
                "titulo": p.name,
                "titulo_obra": nombre_set,
                "descripcion": limpiar(p.description),
                "texto": cuerpo,
                "version_juego": versiones.get(("reliquary", str(det.id)), VERSION_ANTERIOR),
            })
        return out

    res = await asyncio.gather(*[_acotar(sem, uno(s)) for s in sets], return_exceptions=True)
    return [t for r in res if isinstance(r, list) for t in r]


def guardar(driver, textos: list[dict]) -> int:
    ahora = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for t in textos:
        t["fuente"] = FUENTE_URL
        t["fecha_importacion"] = ahora
    with driver.session() as s:
        s.run("CREATE CONSTRAINT textojuego_id IF NOT EXISTS FOR (t:TextoJuego) REQUIRE t.id IS UNIQUE")
        for i in range(0, len(textos), 200):
            s.run(
                """
                UNWIND $lote AS t
                MERGE (n:TextoJuego {id: t.id})
                SET n += t
                """,
                lote=textos[i:i + 200],
            )
    return len(textos)


def asociar_entidades(driver, textos: list[dict]) -> None:
    """Rellena entidad_id (título = entidad) y menciona (nombres citados) en cada texto."""
    with driver.session() as s:
        nombres = [r["id"] for r in s.run("MATCH (e:Entity) RETURN e.id AS id")]
    por_minus = {n.lower(): n for n in nombres}
    regex = construir_regex_menciones(nombres)
    for t in textos:
        entidad = por_minus.get(t["titulo_obra"].lower()) or por_minus.get(t["titulo"].lower())
        t["entidad_id"] = entidad or ""
        t["menciona"] = detectar_menciones(regex, t["texto"], propio=entidad or "")


async def main(tipos: list[str], limite: int):
    driver = crawler_entidades.crear_driver_neo4j()
    try:
        async with ambr.AmbrAPI(lang=ambr.Language.EN) as api:
            versiones = mapa_versiones(await api.fetch_changelogs())
            todos: list[dict] = []
            if "libros" in tipos:
                todos += await recolectar_libros(api, versiones, limite)
            if "armas" in tipos:
                todos += await recolectar_armas(api, versiones, limite)
            if "artefactos" in tipos:
                todos += await recolectar_artefactos(api, versiones, limite)
        print(f"Recolectados {len(todos)} textos; asociando entidades...")
        asociar_entidades(driver, todos)
        n = guardar(driver, todos)
        con_ent = sum(1 for t in todos if t["entidad_id"])
        print(f"Guardados {n} textos ({con_ent} enlazados a una entidad por título).")
        print("Relaciones creadas:", enlaces_juego.reenlazar(driver))
    finally:
        driver.close()


if __name__ == "__main__":
    logger.remove()
    ap = argparse.ArgumentParser()
    ap.add_argument("--tipos", default="libros,armas,artefactos")
    ap.add_argument("--limite", type=int, default=10**9)
    args = ap.parse_args()
    asyncio.run(main([t.strip() for t in args.tipos.split(",")], args.limite))
