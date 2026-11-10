import hashlib
import html
import json
import os
import re
import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
from neo4j import GraphDatabase
import uvicorn
import crawler_entidades

NEO4J_URI = os.getenv("NEO4J_URI", crawler_entidades.NEO4J_URI).strip()
NEO4J_USER = os.getenv("NEO4J_USER", crawler_entidades.NEO4J_USER).strip()
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", crawler_entidades.NEO4J_PASSWORD).strip()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODELS_FALLBACK = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3-flash-preview",
    "gemma-4-31b-it",
    "gemini-2.5-flash",
]
RUTA_CACHE_TRADUCCIONES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_traducciones.json")

WIKI_API_URL = "https://genshin-impact.fandom.com/api.php"
WIKI_BASE_URL = "https://genshin-impact.fandom.com/wiki/"
FANDOM_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Referer": "https://genshin-impact.fandom.com/",
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
}

app = FastAPI()
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
http_session = requests.Session()
http_session.headers.update(FANDOM_HEADERS)

# Cachés en memoria y en disco para imágenes, redirecciones canónicas y traducciones de Gemini
MAPA_URLS_IMAGEN: dict[str, str | None] = {}
MAPA_CANONICO: dict[str, str] = {}
ALIAS_CONOCIDOS: dict[str, set[str]] = {}
CACHE_AVATARES: dict[str, tuple[bytes, str]] = {}
CACHE_TRADUCCIONES: dict[str, str] = {}
CACHE_TIMELINE_TEYVAT: list[dict] = []

if os.path.exists(RUTA_CACHE_TRADUCCIONES):
    try:
        with open(RUTA_CACHE_TRADUCCIONES, "r", encoding="utf-8") as f_cache:
            CACHE_TRADUCCIONES.update(json.load(f_cache))
    except Exception:
        pass


def guardar_cache_traducciones_disco():
    try:
        with open(RUTA_CACHE_TRADUCCIONES, "w", encoding="utf-8") as f_cache:
            json.dump(CACHE_TRADUCCIONES, f_cache, ensure_ascii=False)
    except Exception:
        pass


def llamar_gemini_con_fallback(instruccion: str, contents: list[dict], temperature: float = 0.2) -> tuple[str | None, str | None]:
    """
    Llama a la API de Gemini rotando automáticamente entre varios modelos (2.5-flash -> 2.5-flash-lite -> 2.0-flash -> 2.0-flash-lite)
    si alguno devuelve 429 (cuota por minuto/día agotada), multiplicando la cuota disponible por 4.
    """
    ultimo_error = ""
    for modelo in GEMINI_MODELS_FALLBACK:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{modelo}:generateContent?key={GEMINI_API_KEY}"
        try:
            resp = requests.post(
                url,
                json={
                    "systemInstruction": {"parts": [{"text": instruccion}]},
                    "contents": contents,
                    "generationConfig": {"temperature": temperature},
                },
                timeout=50,
            )
            if resp.status_code == 200:
                data = resp.json()
                partes = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                texto = "".join(p.get("text", "") for p in partes).strip()
                if texto:
                    return texto, None
            elif resp.status_code in (429, 503):
                ultimo_error = f"Cuota agotada en {modelo} (429)"
                continue
            else:
                ultimo_error = f"Error Gemini ({modelo}, {resp.status_code}): {resp.text[:180]}"
        except Exception as e:
            ultimo_error = f"Error de red ({modelo}): {e}"
            continue
    return None, ultimo_error or "Se agotó la cuota en todos los modelos de respaldo."


class PeticionTraduccion(BaseModel):
    texto: str
    contexto: str = ""


class PeticionChat(BaseModel):
    pregunta: str
    entidad_foco: str = ""
    historial: list[dict] = []

ETIQUETAS_INFOBOX = {
    "realname": "Nombre Real",
    "name": "Nombre",
    "title": "Títulos",
    "alias": "Alias",
    "aliases": "Alias",
    "aka": "También conocido como",
    "element": "Elemento",
    "weapon": "Arma",
    "rarity": "Rareza",
    "quality": "Calidad",
    "constellation": "Constelación",
    "region": "Región",
    "nation": "Nación",
    "subregion": "Subregión",
    "area": "Área",
    "location": "Ubicación",
    "affiliation": "Afiliaciones",
    "group": "Grupo / Facción",
    "organization": "Organización",
    "faction": "Facción",
    "leader": "Líder",
    "founder": "Fundador",
    "archon": "Arconte",
    "ideal": "Ideal",
    "governing_body": "Órgano de Gobierno",
    "base": "Base / Sede",
    "ancestry": "Linaje / Origen",
    "species": "Especie / Raza",
    "race": "Raza",
    "gender": "Género",
    "sex": "Sexo",
    "family": "Familia",
    "father": "Padre",
    "mother": "Madre",
    "sibling": "Hermanos",
    "siblings": "Hermanos",
    "spouse": "Cónyuge",
    "child": "Hijos",
    "children": "Hijos",
    "relative": "Parientes",
    "relatives": "Parientes",
    "deceased": "Estado / Fallecimiento",
    "status": "Estado",
    "birthday": "Cumpleaños",
    "dish": "Plato Especial",
    "author": "Autor",
    "volumes": "Volúmenes",
    "type": "Categoría / Tipo",
    "category": "Categoría",
    "chapter": "Capítulo",
    "act": "Acto",
    "prev": "Anterior",
    "next": "Siguiente",
    "characters": "Personajes",
    "source": "Fuente",
}

CLAVES_IDIOMA_EXCLUIDAS = {
    "en", "zh", "zhs", "zht", "chs", "cht", "ja", "jp", "ko", "kr",
    "es", "fr", "ru", "th", "vi", "vn", "de", "id", "pt", "tr", "it",
    "tl", "rm", "lit", "default", "header", "footer", "nocat", "sort",
}


def normalizar_titulo_canonico(titulo: str) -> str:
    """Si una redirección de la Wiki apunta a 'Nahida/Storyline' o 'Zhongli/Lore', devuelve 'Nahida' o 'Zhongli'."""
    if not titulo:
        return ""
    base = titulo.split("#")[0].strip().replace("_", " ")
    if "/" in base:
        raiz, sub = base.split("/", 1)
        if sub.lower().startswith(("lore", "storyline", "history", "profile", "voice-overs", "gallery", "outfits", "combat", "companion")):
            base = raiz.strip()
    return (base[0].upper() + base[1:]) if base else ""


def limpiar_valor_infobox(val: str) -> str:
    v = re.sub(r"<ref.*?>.*?</ref>", "", val, flags=re.DOTALL)
    v = re.sub(r"<ref[^>]*/>", "", v)
    prev = None
    while prev != v:
        prev = v
        v = re.sub(r"\{\{[^{}]*\}\}", "", v)
    v = re.sub(r"\[\[[^\]|]+\|([^\]]+)\]\]", r"\1", v)
    v = re.sub(r"\[\[([^\]]+)\]\]", r"\1", v)
    v = re.sub(r"<br\s*/?>", ", ", v, flags=re.IGNORECASE)
    v = re.sub(r"<[^>]+>", "", v)
    return v.strip(" ,;|-")


def parsear_texto_y_atributos_wiki(texto_bruto: str | None) -> tuple[str, list[dict[str, str]], list[dict[str, str]]]:
    """
    Extrae TODOS los metadatos de la Infobox sin truncar y divide el artículo completo
    en capítulos/secciones (Introducción, Profile, Personality, Appearance, Story, Character Stories, Trivia, etc.).
    """
    if not texto_bruto:
        return "", [], []

    texto = texto_bruto.strip()
    atributos: list[dict[str, str]] = []
    valores_por_clave: dict[str, list[str]] = {}
    orden_claves: list[str] = []

    # Aislar únicamente el bloque inicial de la Infobox (antes del primer encabezado == o de Other Languages)
    bloque_infobox = texto
    if "\n}}\n" in bloque_infobox:
        bloque_infobox = bloque_infobox.split("\n}}\n", 1)[0]
    bloque_infobox = re.split(r"(?m)^==+|\{\{\s*Other\s+Languages", bloque_infobox, maxsplit=1, flags=re.IGNORECASE)[0]

    for match in re.finditer(r"^\s*\|\s*([a-zA-Z0-9_]+)\s*=\s*(.+)$", bloque_infobox, flags=re.MULTILINE):
        clave_raw = match.group(1).strip().lower()
        valor_raw = match.group(2).strip()
        if not valor_raw or "note" in clave_raw or clave_raw.startswith(("image", "caption", "voice", "obtain", "release")):
            continue
        clave_base = re.sub(r"\d+$", "", clave_raw).strip("_")
        # Excluir cualquier clave de traducción/idioma (ja_rm, ko, es, fr, ru, zh_tl, etc.) o ajena a la Ficha Técnica
        if (
            clave_base in CLAVES_IDIOMA_EXCLUIDAS
            or clave_raw in CLAVES_IDIOMA_EXCLUIDAS
            or clave_base.endswith(("_rm", "_tl", "_lit", "_note", "_ref", "_lang"))
            or any(clave_base.startswith(pref + "_") for pref in ("en", "zh", "zhs", "zht", "ja", "ko", "es", "fr", "ru", "th", "vi", "de", "id", "pt", "tr", "it"))
            or clave_base not in ETIQUETAS_INFOBOX
        ):
            continue
        val_limpio = limpiar_valor_infobox(valor_raw)
        if val_limpio and not val_limpio.lower().endswith((".png", ".jpg", ".webp", ".ogg", ".mp3", ".wav")):
            if clave_base not in valores_por_clave:
                valores_por_clave[clave_base] = []
                orden_claves.append(clave_base)
            if val_limpio not in valores_por_clave[clave_base]:
                valores_por_clave[clave_base].append(val_limpio)

    for clave_b in orden_claves:
        etiqueta = ETIQUETAS_INFOBOX[clave_b]
        atributos.append({
            "etiqueta": etiqueta,
            "valor": " · ".join(valores_por_clave[clave_b]),
        })

    # Eliminar plantillas anidadas {{ ... }} que hayan podido quedar
    prev = None
    while prev != texto:
        prev = texto
        texto = re.sub(r"\{\{[^{}]*\}\}", "", texto, flags=re.DOTALL)

    if texto.lstrip().startswith("{{") and "\n}}\n" in texto:
        texto = texto.split("\n}}\n", 1)[1]
    elif "\n}}\n" in texto and "|" in texto.split("\n}}\n", 1)[0]:
        texto = texto.split("\n}}\n", 1)[1]

    lineas_limpias = []
    for linea in texto.splitlines():
        l_strip = linea.strip()
        if not l_strip:
            lineas_limpias.append("")
            continue
        if l_strip.startswith(("{{", "}}", "|")) or re.match(r"^[^|]+\.(?:png|jpg|webp)\|", l_strip, re.IGNORECASE):
            continue
        lineas_limpias.append(l_strip)

    cuerpo_total = "\n".join(lineas_limpias)
    cuerpo_total = re.sub(r"\n{3,}", "\n\n", cuerpo_total).strip()

    # Dividir por encabezados == Sección == para ofrecer todas las secciones de la página
    partes_sec = re.split(r"^==+\s*(.*?)\s*==+\s*$", cuerpo_total, flags=re.MULTILINE)
    secciones: list[dict[str, str]] = []

    if partes_sec:
        intro = partes_sec[0].strip()
        if intro:
            secciones.append({"titulo": "Introducción / Resumen", "contenido": intro})
        idx = 1
        while idx + 1 < len(partes_sec):
            t_sec = partes_sec[idx].strip()
            c_sec = partes_sec[idx + 1].strip()
            if c_sec:
                secciones.append({"titulo": t_sec, "contenido": c_sec})
            idx += 2

    if not secciones and cuerpo_total:
        secciones.append({"titulo": "Información General", "contenido": cuerpo_total})

    return cuerpo_total, atributos, secciones


def asegurar_articulo_completo_wiki(session_neo4j, id_entidad: str, texto_actual: str) -> str:
    """
    Si el nodo en Neo4j solo tenía la introducción corta (sin secciones '==') o llaves desbalanceadas,
    descarga al momento toda la página de la Wiki (+ /Storyline + /Lore) y actualiza Neo4j.
    """
    if (
        texto_actual
        and "\n== " in texto_actual
        and len(texto_actual) > 1500
        and texto_actual.count("{{") == texto_actual.count("}}")
    ):
        return texto_actual

    try:
        wikitext_completo = crawler_entidades.obtener_wikitexto_lore(id_entidad)
        if wikitext_completo:
            texto_completo, _, _ = crawler_entidades.procesar_pagina(id_entidad, wikitext_completo)
            if texto_completo:
                session_neo4j.run(
                    """
                    MERGE (e:Entity {id: $id})
                    SET e.texto_web = $texto
                    """,
                    id=id_entidad,
                    texto=texto_completo,
                )
                # En local (con RAM de sobra) indexamos también los embeddings; en Render Free (512 MB RAM) evitamos cargar el modelo ONNX pesado
                if not os.getenv("RENDER"):
                    chunks = crawler_entidades.generar_chunks_jerarquicos_small_to_big(id_entidad, texto_completo)
                    if chunks:
                        vecs = crawler_entidades.calcular_embeddings_lote([c["texto_embedding"] for c in chunks])
                        if len(vecs) == len(chunks):
                            for c, v in zip(chunks, vecs):
                                c["embedding"] = v
                            session_neo4j.run(
                                """
                                MATCH (e:Entity {id: $entidad_id})
                                OPTIONAL MATCH (e)-[:HAS_CHUNK]->(old_c:Chunk)
                                DETACH DELETE old_c
                                WITH e
                                UNWIND $chunks AS ch
                                MERGE (c:Chunk {id: ch.id})
                                SET c.entidad_id = ch.entidad_id,
                                    c.seccion = ch.seccion,
                                    c.orden = ch.orden,
                                    c.texto = ch.texto,
                                    c.texto_padre = ch.texto_padre,
                                    c.url_wiki = ch.url_wiki,
                                    c.embedding = ch.embedding
                                MERGE (e)-[:HAS_CHUNK]->(c)
                                """,
                                entidad_id=id_entidad,
                                chunks=chunks,
                            )
                return texto_completo
    except Exception as e:
        print(f"Aviso descargando artículo completo de '{id_entidad}': {e}")

    return texto_actual or ""


def obtener_iniciales(nombre: str) -> str:
    palabras = [p for p in nombre.replace("[", "").replace("]", "").split() if p]
    if not palabras:
        return "?"
    if len(palabras) == 1:
        return palabras[0][:2].upper()
    return (palabras[0][0] + palabras[-1][0]).upper()


def generar_svg_fallback(nombre: str) -> bytes:
    """Genera un avatar SVG circular estilo constelación de Genshin cuando la entidad no tiene foto en la Wiki."""
    iniciales = html.escape(obtener_iniciales(nombre))
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" viewBox="0 0 200 200">
        <defs>
            <radialGradient id="g" cx="50%" cy="35%" r="70%">
                <stop offset="0%" stop-color="#2c344a" />
                <stop offset="100%" stop-color="#171b26" />
            </radialGradient>
        </defs>
        <circle cx="100" cy="100" r="98" fill="url(#g)" />
        <circle cx="100" cy="100" r="84" fill="none" stroke="#d3bc8e" stroke-width="2" stroke-opacity="0.45" />
        <polygon points="100,22 114,86 178,100 114,114 100,178 86,114 22,100 86,86" fill="#d3bc8e" fill-opacity="0.12" />
        <text x="100" y="116" text-anchor="middle" fill="#ece5d8" font-family="Cinzel, Georgia, serif" font-weight="bold" font-size="50" letter-spacing="2">{iniciales}</text>
    </svg>"""
    return svg.encode("utf-8")


def precargar_urls_imagenes(nombres: list[str]):
    """
    Consulta en lote (hasta 50 entidades por petición) las miniaturas y redirecciones en la API de Fandom.
    Registra tanto la imagen como el título canónico (ej. 'Buer' y 'Lesser Lord Kusanali' -> 'Nahida').
    """
    pendientes = []
    vistos = set()
    for n in nombres:
        if not n:
            continue
        k = n.strip().lower()
        if k not in MAPA_CANONICO and k not in vistos:
            vistos.add(k)
            pendientes.append(n.strip())

    par_raiz_extra = []

    for i in range(0, len(pendientes), 50):
        lote = pendientes[i : i + 50]
        try:
            res = http_session.get(
                WIKI_API_URL,
                params={
                    "action": "query",
                    "format": "json",
                    "prop": "pageimages",
                    "pithumbsize": 250,
                    "titles": "|".join(lote),
                    "redirects": 1,
                },
                timeout=6,
            )
            if res.status_code != 200:
                for orig in lote:
                    MAPA_CANONICO[orig.lower()] = orig
                continue
            qdata = res.json().get("query", {})

            alias_map: dict[str, str] = {}
            for item in qdata.get("normalized", []):
                alias_map[item["from"].lower()] = item["to"]
            for item in qdata.get("redirects", []):
                origen_l = item["from"].lower()
                dest_t = item["to"]
                alias_map[origen_l] = dest_t
                for k_orig, v_dest in list(alias_map.items()):
                    if v_dest.lower() == origen_l:
                        alias_map[k_orig] = dest_t

            urls_por_titulo: dict[str, str | None] = {}
            titulos_exactos: dict[str, str] = {}
            for page in qdata.get("pages", {}).values():
                t = page.get("title", "")
                titulo_l = t.lower()
                titulos_exactos[titulo_l] = t
                thumb = page.get("thumbnail", {}).get("source")
                if thumb and "/revision/latest" in thumb:
                    base = thumb.split("/revision/latest")[0]
                    urls_por_titulo[titulo_l] = f"{base}/revision/latest/top-crop/width/200/height/200"
                else:
                    urls_por_titulo[titulo_l] = thumb

            for nombre_orig in lote:
                k = nombre_orig.lower()
                destino_bruto = alias_map.get(k, titulos_exactos.get(k, nombre_orig))
                canonico = normalizar_titulo_canonico(destino_bruto) or nombre_orig
                canon_l = canonico.lower()

                MAPA_CANONICO[k] = canonico
                MAPA_CANONICO[canon_l] = canonico
                ALIAS_CONOCIDOS.setdefault(canon_l, set()).update([k, canon_l])

                img = urls_por_titulo.get(destino_bruto.lower()) or urls_por_titulo.get(canon_l)
                if img:
                    MAPA_URLS_IMAGEN[k] = img
                    MAPA_URLS_IMAGEN[canon_l] = img
                elif "/" in destino_bruto and canon_l not in MAPA_URLS_IMAGEN:
                    par_raiz_extra.append(canonico)
                else:
                    MAPA_URLS_IMAGEN.setdefault(k, None)
        except Exception as e:
            print(f"Error precargando lote de imágenes/redirecciones: {e}")
            for orig in lote:
                MAPA_CANONICO.setdefault(orig.lower(), orig)

    if par_raiz_extra:
        raices_unicas = list({r for r in par_raiz_extra if r.lower() not in MAPA_URLS_IMAGEN})
        if raices_unicas:
            try:
                res2 = http_session.get(
                    WIKI_API_URL,
                    params={
                        "action": "query",
                        "format": "json",
                        "prop": "pageimages",
                        "pithumbsize": 250,
                        "titles": "|".join(raices_unicas[:50]),
                        "redirects": 1,
                    },
                    timeout=6,
                )
                if res2.status_code == 200:
                    for page in res2.json().get("query", {}).get("pages", {}).values():
                        tl = page.get("title", "").lower()
                        thumb = page.get("thumbnail", {}).get("source")
                        if thumb and "/revision/latest" in thumb:
                            base = thumb.split("/revision/latest")[0]
                            url_crop = f"{base}/revision/latest/top-crop/width/200/height/200"
                            MAPA_URLS_IMAGEN[tl] = url_crop
                            for alias_k in ALIAS_CONOCIDOS.get(tl, set()):
                                MAPA_URLS_IMAGEN[alias_k] = url_crop
            except Exception:
                pass


def obtener_nombre_canonico(nombre: str) -> str:
    if not nombre:
        return ""
    k = nombre.strip().lower()
    if k not in MAPA_CANONICO:
        precargar_urls_imagenes([nombre.strip()])
    return MAPA_CANONICO.get(k, nombre.strip())


def obtener_aliases_completos(canon_id: str) -> list[str]:
    """Obtiene todos los alias de la Wiki que redirigen a canon_id (o a canon_id/Storyline y canon_id/Lore)."""
    key = canon_id.lower()
    aliases = ALIAS_CONOCIDOS.setdefault(key, {key})
    if len(aliases) == 1:
        try:
            res = http_session.get(
                WIKI_API_URL,
                params={
                    "action": "query",
                    "format": "json",
                    "prop": "redirects",
                    "rdlimit": 50,
                    "titles": f"{canon_id}|{canon_id}/Storyline|{canon_id}/Lore",
                },
                timeout=5,
            )
            if res.status_code == 200:
                pages = res.json().get("query", {}).get("pages", {})
                for p in pages.values():
                    for r in p.get("redirects", []):
                        t = r.get("title", "").strip()
                        if t:
                            tl = t.lower()
                            aliases.add(tl)
                            MAPA_CANONICO[tl] = canon_id
        except Exception:
            pass
    return list(aliases)


@app.get("/api/avatar/{entidad_id:path}")
def obtener_avatar_entidad(entidad_id: str):
    """Devuelve la imagen cuadrada de la entidad desde la Wiki (con caché) o un avatar SVG Irminsul."""
    limpio = entidad_id.strip()
    canon = obtener_nombre_canonico(limpio)
    key = canon.lower()

    if key in CACHE_AVATARES:
        contenido, mime = CACHE_AVATARES[key]
        return Response(content=contenido, media_type=mime, headers={"Cache-Control": "public, max-age=86400"})

    url_img = MAPA_URLS_IMAGEN.get(key) or MAPA_URLS_IMAGEN.get(limpio.lower())
    if url_img:
        try:
            r = http_session.get(url_img, timeout=6)
            if r.status_code == 200 and r.content:
                mime = r.headers.get("content-type", "image/webp")
                if len(CACHE_AVATARES) > 200:
                    CACHE_AVATARES.pop(next(iter(CACHE_AVATARES)), None)
                CACHE_AVATARES[key] = (r.content, mime)
                return Response(content=r.content, media_type=mime, headers={"Cache-Control": "public, max-age=86400"})
        except Exception:
            pass

    svg_bytes = generar_svg_fallback(canon)
    if len(CACHE_AVATARES) > 200:
        CACHE_AVATARES.pop(next(iter(CACHE_AVATARES)), None)
    CACHE_AVATARES[key] = (svg_bytes, "image/svg+xml")
    return Response(content=svg_bytes, media_type="image/svg+xml", headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/buscar")
def buscar_entidades(q: str = ""):
    with driver.session() as session:
        if not q.strip():
            res = session.run(
                """
                MATCH (e:Entity)
                WHERE NOT e.id STARTS WITH ':'
                  AND NOT toLower(e.id) STARTS WITH 'category:'
                  AND NOT toLower(e.id) STARTS WITH 'file:'
                  AND NOT toLower(e.id) STARTS WITH 'template:'
                RETURN e.id AS id
                ORDER BY e.id
                LIMIT 300
                """
            )
        else:
            res = session.run(
                """
                MATCH (e:Entity)
                WHERE toLower(e.id) CONTAINS toLower($q)
                  AND NOT e.id STARTS WITH ':'
                  AND NOT toLower(e.id) STARTS WITH 'category:'
                  AND NOT toLower(e.id) STARTS WITH 'file:'
                  AND NOT toLower(e.id) STARTS WITH 'template:'
                RETURN e.id AS id
                ORDER BY CASE WHEN toLower(e.id) = toLower($q) THEN 0 ELSE 1 END, e.id
                LIMIT 50
                """,
                q=q.strip(),
            )
        brutos = [r["id"] for r in res if crawler_entidades.es_entidad_valida(r["id"])]
        if len(brutos) <= 50:
            precargar_urls_imagenes(brutos)
        vistos = set()
        unicos = []
        for b in brutos:
            c = MAPA_CANONICO.get(b.lower(), b)
            if crawler_entidades.es_entidad_valida(c) and c.lower() not in vistos:
                vistos.add(c.lower())
                unicos.append(c)
        return unicos


@app.post("/api/traducir")
def traducir_texto_gemini(req: PeticionTraduccion):
    """Traduce al español textos de lore de Genshin Impact usando rotación de modelos Gemini y caché persistente en disco."""
    texto_orig = (req.texto or "").strip()
    if not texto_orig:
        return {"traduccion": ""}

    clave_hash = hashlib.md5(texto_orig.encode("utf-8")).hexdigest()
    if clave_hash in CACHE_TRADUCCIONES:
        return {"traduccion": CACHE_TRADUCCIONES[clave_hash]}

    instruccion = (
        "Eres un traductor experto y redactor de una enciclopedia de lore de Genshin Impact en español. "
        "Traduce el siguiente texto del inglés al español con estilo literario, enciclopédico y riguroso, "
        "utilizando la terminología oficial en español de Genshin Impact (por ejemplo: 'Los Once de los Fatui', "
        "'El Capitán / Capitano', 'Reina Menor Kusanali', 'Los Siete', 'Principios Celestiales', 'Orden del Abismo', "
        "'Irminsul', 'Líneas Ley', 'Cataclismo', 'Reino de la Noche', 'Engaño' para Delusion, 'Gnosis', 'Visión', "
        "'Misión de Arconte', 'Historia del Personaje'). "
        "Si el texto incluye separadores exactos como '===SEPARADOR_GI===', consérvalos exactamente igual en su posición. "
        "Conserva exactamente la división en párrafos y devuelve ÚNICAMENTE la traducción en español, sin comentarios previos ni finales."
    )
    if req.contexto:
        instruccion += f" Contexto de la entidad/sección: {req.contexto}."

    traduccion, err = llamar_gemini_con_fallback(
        instruccion,
        [{"role": "user", "parts": [{"text": texto_orig}]}],
        temperature=0.2,
    )
    if traduccion:
        CACHE_TRADUCCIONES[clave_hash] = traduccion
        guardar_cache_traducciones_disco()
        return {"traduccion": traduccion}
    return {"error": err or "No se pudo traducir con Gemini."}


@app.post("/api/chat")
def consultar_chatbot_graphrag(req: PeticionChat):
    """
    Chatbot GraphRAG Híbrido (Small-to-Big Vector Search + 1-Hop Graph Traversal en Neo4j):
    1. Vectoriza la pregunta con el modelo multilingüe gratuito (FastEmbed 384d).
    2. Recupera los Chunks hijos más relevantes (Small) junto con el texto de su sección Padre (Big).
    3. Combina búsqueda vectorial con detección de entidades mencionadas y recorre 1 salto en el grafo
       (:Entity)-[:RELATED_TO]-(:Entity) para extraer relaciones verificadas y sus enlaces de referencia.
    4. Genera la respuesta en español con Gemini 2.5 Flash y devuelve las fuentes de la Wiki y relaciones.
    """
    pregunta = (req.pregunta or "").strip()
    if not pregunta:
        return {"error": "Escribe una pregunta para consultar el archivo."}

    chunks_recuperados = []
    entidades_clave = []

    palabras_clave = [
        p.strip("¿?¡!.,;:()\"'")
        for p in pregunta.split()
        if len(p.strip("¿?¡!.,;:()\"'")) >= 4
    ]

    with driver.session() as session:
        # 1. Búsqueda vectorial sobre el índice Neo4j 'chunk_embeddings' (en local) o búsqueda léxica/semántica ligera (en Render 512MB)
        if not os.getenv("RENDER"):
            try:
                vectores_q = crawler_entidades.calcular_embeddings_lote([pregunta])
                if vectores_q:
                    res_vec = session.run(
                        """
                        CALL db.index.vector.queryNodes('chunk_embeddings', 10, $vec)
                        YIELD node AS c, score
                        RETURN c.id AS chunk_id,
                               c.entidad_id AS entidad_id,
                               c.seccion AS seccion,
                               c.texto AS texto_hijo,
                               c.texto_padre AS texto_padre,
                               c.url_wiki AS url_wiki,
                               score
                        ORDER BY score DESC
                        """,
                        vec=vectores_q[0],
                    ).data()
                    chunks_recuperados.extend(res_vec)
            except Exception as e:
                print(f"Aviso en consulta vectorial Neo4j: {e}")
        elif palabras_clave:
            try:
                res_kw = session.run(
                    """
                    MATCH (c:Chunk)
                    WITH c, size([w IN $palabras WHERE toLower(c.texto) CONTAINS toLower(w) OR toLower(c.entidad_id) CONTAINS toLower(w)]) AS coincidencias
                    WHERE coincidencias > 0
                    RETURN c.id AS chunk_id,
                           c.entidad_id AS entidad_id,
                           c.seccion AS seccion,
                           c.texto AS texto_hijo,
                           c.texto_padre AS texto_padre,
                           c.url_wiki AS url_wiki,
                           toFloat(coincidencias) / toFloat(size($palabras)) AS score
                    ORDER BY coincidencias DESC
                    LIMIT 10
                    """,
                    palabras=palabras_clave,
                ).data()
                chunks_recuperados.extend(res_kw)
            except Exception as e:
                print(f"Aviso en consulta de chunks en Render: {e}")

        # 2. Detectar entidades mencionadas explícitamente en la pregunta (o la entidad foco actual)
        entidades_mencionadas = []
        if palabras_clave:
            res_ent = session.run(
                """
                MATCH (e:Entity)
                WHERE any(w IN $palabras WHERE toLower(e.id) CONTAINS toLower(w))
                RETURN e.id AS id
                LIMIT 6
                """,
                palabras=palabras_clave,
            ).data()
            entidades_mencionadas = [r["id"] for r in res_ent]

        if req.entidad_foco and req.entidad_foco not in entidades_mencionadas:
            entidades_mencionadas.append(req.entidad_foco)

        # Si alguna entidad mencionada directamente solo tenía stub sin chunks, asegurar su artículo e indexarla
        for ent_m in entidades_mencionadas[:2]:
            canon_m = obtener_nombre_canonico(ent_m)
            chk_exist = session.run(
                "MATCH (e:Entity {id: $id})-[:HAS_CHUNK]->(c:Chunk) RETURN count(c) AS cnt",
                id=canon_m,
            ).single()
            if not chk_exist or chk_exist["cnt"] == 0:
                reg_t = session.run("MATCH (e:Entity {id: $id}) RETURN e.texto_web AS t", id=canon_m).single()
                texto_act = reg_t["t"] if reg_t else ""
                asegurar_articulo_completo_wiki(session, canon_m, texto_act)

            # Añadir también los mejores chunks de esa entidad mencionada si no estaban ya
            res_extra = session.run(
                """
                MATCH (e:Entity {id: $id})-[:HAS_CHUNK]->(c:Chunk)
                RETURN c.id AS chunk_id,
                       c.entidad_id AS entidad_id,
                       c.seccion AS seccion,
                       c.texto AS texto_hijo,
                       c.texto_padre AS texto_padre,
                       c.url_wiki AS url_wiki,
                       0.88 AS score
                ORDER BY c.orden ASC
                LIMIT 3
                """,
                id=canon_m,
            ).data()
            ids_ya = {c["chunk_id"] for c in chunks_recuperados}
            for rx in res_extra:
                if rx["chunk_id"] not in ids_ya:
                    chunks_recuperados.append(rx)
                    ids_ya.add(rx["chunk_id"])

        # Construir lista única de entidades relevantes para expandir el grafo (1-Hop GraphRAG)
        vistas_ent = set()
        for em in entidades_mencionadas:
            cm = obtener_nombre_canonico(em)
            if cm and cm.lower() not in vistas_ent:
                vistas_ent.add(cm.lower())
                entidades_clave.append(cm)
        for ch in chunks_recuperados:
            ce = obtener_nombre_canonico(ch.get("entidad_id") or "")
            if ce and ce.lower() not in vistas_ent and len(entidades_clave) < 7:
                vistas_ent.add(ce.lower())
                entidades_clave.append(ce)

        # 3. Recorrido de Grafo (1 salto RELATED_TO) desde las entidades clave
        relaciones_recuperadas = []
        if entidades_clave:
            res_rels = session.run(
                """
                MATCH (o:Entity)-[r:RELATED_TO]->(d:Entity)
                WHERE o.id IN $ents OR d.id IN $ents
                RETURN o.id AS origen,
                       d.id AS destino,
                       r.epoca AS epoca,
                       r.parrafo AS parrafo,
                       r.ref_titulo AS ref_titulo,
                       r.ref_url AS ref_url,
                       CASE WHEN o.id IN $ents AND d.id IN $ents THEN 2 ELSE 1 END AS prioridad
                ORDER BY prioridad DESC
                LIMIT 35
                """,
                ents=entidades_clave,
            ).data()

            rels_vistas = set()
            for r in res_rels:
                orig_c = obtener_nombre_canonico(r["origen"])
                dest_c = obtener_nombre_canonico(r["destino"])
                if orig_c.lower() == dest_c.lower():
                    continue
                epoca = r.get("epoca") or "Overview"
                parrafo = (r.get("parrafo") or "").strip()
                clave_r = (orig_c.lower(), dest_c.lower(), epoca.lower(), parrafo[:120].lower())
                if clave_r in rels_vistas:
                    continue
                rels_vistas.add(clave_r)
                info_era = clasificar_era_historica(epoca, parrafo)
                relaciones_recuperadas.append({
                    "origen": orig_c,
                    "destino": dest_c,
                    "epoca": epoca,
                    "era_corta": info_era["era_corta"],
                    "parrafo": parrafo,
                    "ref_titulo": r.get("ref_titulo") or f"{orig_c} ({epoca})",
                    "ref_url": r.get("ref_url") or f"{WIKI_BASE_URL}{orig_c.replace(' ', '_')}",
                })
                if len(relaciones_recuperadas) >= 14:
                    break

    # Precargar avatares de las entidades implicadas
    precargar_urls_imagenes(entidades_clave)

    # 4. Construir contexto Small-to-Big (combinando pasaje Hijo + contexto de Sección Padre) y Relaciones
    fuentes_wiki = []
    bloques_contexto_chunks = []
    secciones_padre_vistas = set()

    for idx, ch in enumerate(chunks_recuperados[:8], start=1):
        ent_id = obtener_nombre_canonico(ch.get("entidad_id") or "")
        seccion = ch.get("seccion") or "Overview"
        url_w = ch.get("url_wiki") or f"{WIKI_BASE_URL}{ent_id.replace(' ', '_')}"
        texto_hijo = (ch.get("texto_hijo") or "").strip()
        texto_padre = (ch.get("texto_padre") or texto_hijo).strip()
        score = round(float(ch.get("score") or 0.0), 3)

        clave_sec = (ent_id.lower(), seccion.lower())
        # Estrategia Small-to-Big: si es la primera vez que vemos esta sección padre, damos el bloque Padre amplio
        cuerpo_para_llm = texto_padre[:1800] if clave_sec not in secciones_padre_vistas else texto_hijo
        secciones_padre_vistas.add(clave_sec)

        bloques_contexto_chunks.append(
            f"[Fuente F{idx} | Entidad: {ent_id} | Sección: {seccion} | Wiki: {url_w}]\n{cuerpo_para_llm}"
        )
        fuentes_wiki.append({
            "codigo": f"F{idx}",
            "entidad_id": ent_id,
            "seccion": seccion,
            "url_wiki": url_w,
            "extracto": texto_hijo[:420] + ("..." if len(texto_hijo) > 420 else ""),
            "score": score,
        })

    bloques_contexto_rels = []
    for idx, rel in enumerate(relaciones_recuperadas[:12], start=1):
        bloques_contexto_rels.append(
            f"[Relación R{idx} | {rel['origen']} ➔ {rel['destino']} | Época: {rel['epoca']} ({rel['era_corta']}) | "
            f"Fuente: {rel['ref_titulo']} ({rel['ref_url']})]\n\"{rel['parrafo']}\""
        )

    contexto_total = (
        "=== FRAGMENTOS DE PÁGINAS DE LA WIKI (CHUNKS JERÁRQUICOS SMALL-TO-BIG) ===\n"
        + ("\n\n".join(bloques_contexto_chunks) if bloques_contexto_chunks else "Sin fragmentos directos.")
        + "\n\n=== RELACIONES Y CITAS DEL GRAFO DE CONOCIMIENTO (NEO4J) ===\n"
        + ("\n\n".join(bloques_contexto_rels) if bloques_contexto_rels else "Sin relaciones adicionales.")
    )

    instruccion_sistema = (
        "Eres el 'Oráculo del Irminsul', un asistente experto en el lore de Genshin Impact que ayuda a una autora "
        "a escribir su libro enciclopedia de lore en español.\n"
        "REGLAS ESTRICTAS:\n"
        "1. Responde SIEMPRE en español con estilo enciclopédico, claro, bien estructurado y riguroso.\n"
        "2. Basa tu respuesta en las Fuentes [F1..Fn] y Relaciones del Grafo [R1..Rn] proporcionadas.\n"
        "3. Cita explícitamente en el texto de dónde sale cada dato usando corchetes, por ejemplo: "
        "[Capitano - The Cataclysm (F1)] o [Relación Capitano ➔ Mavuika (R2)].\n"
        "4. Utiliza la terminología oficial en español de Genshin Impact (Los Once de los Fatui, Reina Menor Kusanali, "
        "Principios Celestiales, Siete Soberanos, Cataclismo, Irminsul, Líneas Ley, Reino de la Noche)."
    )

    mensajes_gemini = []
    for h in (req.historial or [])[-4:]:
        rol_h = "user" if h.get("rol") == "user" else "model"
        txt_h = (h.get("texto") or "").strip()
        if txt_h:
            mensajes_gemini.append({"role": rol_h, "parts": [{"text": txt_h}]})

    prompt_usuario = (
        f"CONTEXTO RECUPERADO DE NEO4J Y LA WIKI:\n{contexto_total}\n\n"
        f"PREGUNTA DEL USUARIO (Foco actual en el grafo: {req.entidad_foco or 'General'}):\n{pregunta}"
    )
    mensajes_gemini.append({"role": "user", "parts": [{"text": prompt_usuario}]})

    respuesta_texto, err = llamar_gemini_con_fallback(
        instruccion_sistema,
        mensajes_gemini,
        temperature=0.25,
    )
    if respuesta_texto:
        return {
            "respuesta": respuesta_texto,
            "fuentes_wiki": fuentes_wiki,
            "relaciones_grafo": relaciones_recuperadas[:10],
            "entidades_clave": entidades_clave,
        }
    return {"error": err or "No se pudo generar una respuesta."}


def clasificar_era_historica(epoca: str, parrafo: str = "") -> dict:
    """
    Clasifica cada relación o suceso en una de las 6 Grandes Eras Cronológicas de Teyvat
    analizando tanto el título de la época/sección como los marcadores temporales del párrafo.
    """
    ep = (epoca or "Overview").strip().lower()
    txt = f"{ep} {(parrafo or '').lower()}"

    # I. Mundo Antiguo y Era Primordial
    if any(
        k in ep
        for k in (
            "old world",
            "ancient history",
            "primordial",
            "unified civilization",
            "seven sovereigns",
            "three moons",
            "hyperborea",
            "before the archon war",
        )
    ) or any(
        k in txt
        for k in (
            "primordial one",
            "phanes",
            "seven sovereigns",
            "dragon king nibelung",
            "nibelung",
            "unified civilization",
            "moon sisters",
            "three moons",
            "second who came",
            "war of vengeance",
        )
    ):
        return {
            "era_id": "era_1_antigua",
            "era_orden": 1,
            "era_corta": "I. Mundo Antiguo",
            "era_titulo": "I. Mundo Antiguo y Era Primordial",
        }

    # II. Guerra de los Arcontes y Era de los Dioses
    if any(
        k in ep
        for k in (
            "archon war",
            "remuria",
            "gurabad",
            "deshret",
            "decarabian",
            "guili",
            "enkanomiya",
        )
    ) or any(
        k in txt
        for k in (
            "archon war",
            "guili assembly",
            "king deshret",
            "remuria",
            "2,000 years ago",
            "2,600 years ago",
            "3,700 years ago",
            "thousand winds",
        )
    ):
        return {
            "era_id": "era_2_arcontes",
            "era_orden": 2,
            "era_corta": "II. Guerra de Arcontes",
            "era_titulo": "II. Guerra de los Arcontes y Era de los Dioses",
        }

    # III. El Cataclismo (Hace 500 años)
    if any(
        k in ep
        for k in (
            "cataclysm",
            "aftermath",
            "fall of khaenri",
            "eclipse dynasty",
        )
    ) or any(
        k in txt
        for k in (
            "cataclysm",
            "500 years ago",
            "five hundred years ago",
            "fall of khaenri'ah",
            "abyssal invasion",
            "eclipse dynasty",
        )
    ):
        return {
            "era_id": "era_3_cataclismo",
            "era_orden": 3,
            "era_corta": "III. El Cataclismo",
            "era_titulo": "III. El Cataclismo y Caída de Khaenri'ah (Hace 500 años)",
        }

    # IV. Historia y Pasado Previo
    if any(
        k in ep
        for k in (
            "history",
            "past",
            "background",
            "pre-",
            "historical",
            "legacy",
            "origin",
            "youth",
            "early life",
            "before",
            "founding",
            "creation",
            "fatui involvement",
        )
    ):
        return {
            "era_id": "era_4_pasado",
            "era_orden": 4,
            "era_corta": "IV. Historia y Pasado",
            "era_titulo": "IV. Historia Previa, Orígenes y Legado",
        }

    # V. Era Presente y Trama Actual
    if any(
        k in ep
        for k in (
            "present",
            "current",
            "story",
            "mondstadt",
            "liyue",
            "inazuma",
            "sumeru",
            "fontaine",
            "natlan",
            "snezhnaya",
            "nod-krai",
            "welkin moon",
            "archon quest",
            "chapter",
            "prologue",
            "interlude",
        )
    ):
        return {
            "era_id": "era_5_presente",
            "era_orden": 5,
            "era_corta": "V. Era Presente",
            "era_titulo": "V. Era Presente y Trama Actual en Teyvat",
        }

    # VI. Archivo General y Perfil
    return {
        "era_id": "era_6_archivo",
        "era_orden": 6,
        "era_corta": "VI. Archivo y Perfil",
        "era_titulo": "VI. Perfil General, Historias de Personaje y Lore",
    }


METADATOS_ERAS_TEYVAT = {
    "old world": {
        "orden": 1,
        "titulo_es": "I. El Mundo Antiguo (Old World)",
        "subtitulo": "Era de los Siete Soberanos Dragón, Nibelung y las Tres Lunas",
    },
    "when the doves held branches": {
        "orden": 2,
        "titulo_es": "II. Cuando las Palomas Portaban Ramas",
        "subtitulo": "Descenso de los Principios Celestiales y creación de las Cuatro Sombras",
    },
    "unified civilization": {
        "orden": 3,
        "titulo_es": "III. La Civilización Unificada",
        "subtitulo": "Primera Era de la Humanidad, los Seelie e Hiperbórea",
    },
    "war of funerary flame": {
        "orden": 4,
        "titulo_es": "IV. La Guerra de la Llama Funeraria",
        "subtitulo": "El Segundo Trono, la Guerra de la Venganza y los Clavos Celestiales",
    },
    "interim period": {
        "orden": 5,
        "titulo_es": "V. Periodo Intermedio",
        "subtitulo": "Caída de Sal Vindagnyr, Enkanomiya, Isla Tsurumi y Antiguo Mondstadt",
    },
    "archon war": {
        "orden": 6,
        "titulo_es": "VI. La Guerra de los Arcontes (Archon War)",
        "subtitulo": "Disputa por los Siete Tronos Divinos (~3.700 – 2.000 años atrás)",
    },
    "post-archon war": {
        "orden": 7,
        "titulo_es": "VII. Posguerra de los Arcontes",
        "subtitulo": "Consolidación de Los Siete, Remuria, el Rey Deshret y las Naciones",
    },
    "pre-cataclysm": {
        "orden": 8,
        "titulo_es": "VIII. Pre-Cataclismo",
        "subtitulo": "Auge de la Dinastía Eclipse en Khaenri'ah (~1.000 – 500 años atrás)",
    },
    "cataclysm": {
        "orden": 9,
        "titulo_es": "IX. El Cataclismo (Hace 500 años)",
        "subtitulo": "Caída de Khaenri'ah, invasión abismal y tragedia en las Siete Naciones",
    },
    "post-cataclysm": {
        "orden": 10,
        "titulo_es": "X. Pos-Cataclismo",
        "subtitulo": "Fundación de los Fatui, la Orden del Abismo y reconstrucción de Teyvat",
    },
    "recent past": {
        "orden": 11,
        "titulo_es": "XI. Pasado Reciente",
        "subtitulo": "Sucesos previos al despertar del Viajero en Teyvat",
    },
    "the beginning": {
        "orden": 12,
        "titulo_es": "XII. El Comienzo y Era Actual",
        "subtitulo": "El despertar del Viajero y la cronología del presente",
    },
}


@app.get("/api/timeline_teyvat")
def obtener_linea_temporal_teyvat():
    """
    Construye la Línea del Tiempo Completa de Teyvat combinando la página oficial 'Timeline'
    y las 8 subpáginas 'Nación/History' de la Wiki junto con las entidades de la base de datos.
    """
    if CACHE_TIMELINE_TEYVAT:
        return {"eras": CACHE_TIMELINE_TEYVAT}

    titulos_lote = (
        "Timeline|Mondstadt/History|Liyue/History|Inazuma/History|"
        "Sumeru/History|Fontaine/History|Natlan/History|Snezhnaya/History|Khaenri'ah/History"
    )
    paginas_wikitext: dict[str, str] = {}
    try:
        res = http_session.get(
            WIKI_API_URL,
            params={
                "action": "query",
                "format": "json",
                "prop": "revisions",
                "rvprop": "content",
                "titles": titulos_lote,
                "redirects": 1,
            },
            timeout=15,
        )
        if res.status_code == 200:
            for p in res.json().get("query", {}).get("pages", {}).values():
                if "revisions" in p:
                    paginas_wikitext[p["title"]] = p["revisions"][0].get("*", "")
    except Exception as e:
        return {"error": f"No se pudo descargar la Línea del Tiempo de Teyvat: {e}"}

    wikitext_timeline = paginas_wikitext.get("Timeline", "")
    if not wikitext_timeline:
        return {"error": "No se encontró el contenido de Timeline."}

    # Indexar las secciones == Era == de cada página regional (ej. Sumeru/History -> Old World)
    secciones_por_pagina: dict[tuple[str, str], str] = {}
    mapa_refs_global: dict[str, str] = {}

    for titulo_pag, wt_pag in paginas_wikitext.items():
        for m_ref in re.finditer(r'<ref\s+name=["\']?([^"\'>\s]+)["\']?\s*>(.*?)</ref>', wt_pag, flags=re.DOTALL):
            mapa_refs_global[m_ref.group(1).strip()] = m_ref.group(2).strip()
        if titulo_pag == "Timeline":
            continue
        partes_p = re.split(r"^==\s*([^=]+?)\s*==\s*$", wt_pag, flags=re.MULTILINE)
        idx_p = 1
        while idx_p + 1 < len(partes_p):
            nom_s = partes_p[idx_p].strip().lower()
            cuerpo_s = partes_p[idx_p + 1].strip()
            secciones_por_pagina[(titulo_pag.lower(), nom_s)] = cuerpo_s
            idx_p += 2

    # Expandir {{Transclude|Pagina|Seccion}} dentro de Timeline
    def repl_transclude(m):
        pag_t = m.group(1).strip().lower()
        sec_t = m.group(2).strip().lower()
        return secciones_por_pagina.get((pag_t, sec_t), "")

    wikitext_expandido = re.sub(
        r"\{\{Transclude\|([^|}]+)\|([^|}]+)[^}]*\}\}",
        repl_transclude,
        wikitext_timeline,
        flags=re.IGNORECASE,
    )

    # Dividir Timeline por sus grandes épocas (== Época ==)
    partes_eras = re.split(r"^==\s*([^=]+?)\s*==\s*$", wikitext_expandido, flags=re.MULTILINE)
    eras_resultado = []
    idx_e = 1

    while idx_e + 1 < len(partes_eras):
        nombre_era_en = partes_eras[idx_e].strip()
        cuerpo_era = partes_eras[idx_e + 1]
        idx_e += 2

        clave_era = nombre_era_en.lower()
        if clave_era in ("notes", "references", "navigation", "other languages", "change history"):
            continue

        meta = METADATOS_ERAS_TEYVAT.get(
            clave_era,
            {
                "orden": 99,
                "titulo_es": nombre_era_en,
                "subtitulo": "Registros históricos de Teyvat",
            },
        )

        # Dividir la época en bloques regionales (=== Región ===)
        partes_reg = re.split(r"^===+\s*(.+?)\s*===+\s*$", cuerpo_era, flags=re.MULTILINE)
        bloques_reg = [("Teyvat (General)", partes_reg[0])]
        ir = 1
        while ir + 1 < len(partes_reg):
            bloques_reg.append((partes_reg[ir].strip(), partes_reg[ir + 1]))
            ir += 2

        hechos_era = []
        conteo_entidades_era: dict[str, int] = {}
        regiones_presentes = []

        for nombre_region, texto_reg in bloques_reg:
            for linea in texto_reg.split("\n"):
                l_strip = linea.strip()
                if not l_strip.startswith("*"):
                    continue
                nivel_sub = len(l_strip) - len(l_strip.lstrip("*"))
                contenido_linea = l_strip.lstrip("*").strip()
                if len(contenido_linea) < 15:
                    continue

                # Extraer primera referencia si existe
                refs_inline = re.findall(r"<ref(?:\s+[^>]*)?>(.*?)</ref>", contenido_linea, flags=re.DOTALL)
                for n_ref in re.findall(r'<ref\s+name=["\']?([^"\'>\s/]+)["\']?\s*/>', contenido_linea):
                    if n_ref in mapa_refs_global:
                        refs_inline.append(mapa_refs_global[n_ref])

                if refs_inline:
                    ref_titulo, ref_url = crawler_entidades.formatear_cita_ref(refs_inline[0])
                else:
                    ancla_era = nombre_era_en.replace(" ", "_")
                    ref_titulo = f"Timeline ({nombre_era_en})"
                    ref_url = f"{WIKI_BASE_URL}Timeline#{ancla_era}"

                entidades_linea = crawler_entidades.extraer_entidades_brutas_de_fragmento(contenido_linea)
                texto_limpio = crawler_entidades.limpiar_texto_plano(contenido_linea)
                if not texto_limpio or len(texto_limpio) < 12:
                    continue

                for ent_n in entidades_linea:
                    conteo_entidades_era[ent_n] = conteo_entidades_era.get(ent_n, 0) + 1

                hechos_era.append({
                    "id": f"era_{meta['orden']}_{len(hechos_era)}",
                    "region": nombre_region,
                    "nivel": min(nivel_sub, 3),
                    "texto": texto_limpio,
                    "entidades": entidades_linea[:8],
                    "ref_titulo": ref_titulo[:100],
                    "ref_url": ref_url or f"{WIKI_BASE_URL}Timeline",
                })

            if any(h["region"] == nombre_region for h in hechos_era) and nombre_region not in regiones_presentes:
                regiones_presentes.append(nombre_region)

        entidades_destacadas = [
            {"id": k, "menciones": v}
            for k, v in sorted(conteo_entidades_era.items(), key=lambda x: (-x[1], x[0]))[:24]
        ]

        eras_resultado.append({
            "id": f"teyvat_era_{meta['orden']}",
            "orden": meta["orden"],
            "nombre_en": nombre_era_en,
            "titulo_es": meta["titulo_es"],
            "subtitulo": meta["subtitulo"],
            "url_wiki": f"{WIKI_BASE_URL}Timeline#{nombre_era_en.replace(' ', '_')}",
            "regiones": regiones_presentes,
            "entidades_destacadas": entidades_destacadas,
            "hechos": hechos_era,
        })

    eras_resultado.sort(key=lambda x: x["orden"])
    CACHE_TIMELINE_TEYVAT.extend(eras_resultado)
    return {"eras": CACHE_TIMELINE_TEYVAT}


@app.get("/api/nodo/{entidad_id:path}")
def obtener_padres_e_hijos(entidad_id: str):
    """
    Busca la entidad (y todos sus alias que redirigen a la misma página de la Wiki),
    asegura que tenga TODAS las secciones de la página descargadas para la enciclopedia,
    y unifica todas sus relaciones.
    """
    canon_id = obtener_nombre_canonico(entidad_id.strip())
    aliases_busqueda = obtener_aliases_completos(canon_id)
    if entidad_id.strip().lower() not in aliases_busqueda:
        aliases_busqueda.append(entidad_id.strip().lower())

    with driver.session() as session:
        query = """
        MATCH (e:Entity)
        WHERE toLower(e.id) IN $aliases
        OPTIONAL MATCH (e)-[r:RELATED_TO]-(v:Entity)
        RETURN collect(DISTINCT e.id) AS ids_encontrados,
               collect(DISTINCT e.url_wiki) AS urls,
               collect(DISTINCT e.texto_web) AS textos,
               collect({
                   rel_id: r.id,
                   origen: startNode(r).id,
                   destino: endNode(r).id,
                   rol_vecino: CASE WHEN startNode(r) = e THEN 'hijo' ELSE 'padre' END,
                   epoca: r.epoca,
                   parrafo: r.parrafo,
                   ref_titulo: r.ref_titulo,
                   ref_url: r.ref_url,
                   ref_texto: r.ref_texto,
                   vecino_id: v.id,
                   vecino_url: v.url_wiki,
                   vecino_texto: v.texto_web
               }) AS conexiones
        """
        record = session.run(query, aliases=aliases_busqueda).single()
        if not record or not record["ids_encontrados"]:
            return {"error": f"No se encontró la entidad '{entidad_id}' en Neo4j."}

        id_principal = canon_id

        textos_validos = [t for t in (record["textos"] or []) if t and t.strip()]
        textos_validos.sort(key=len, reverse=True)
        texto_bruto_mejor = textos_validos[0] if textos_validos else ""

        # Asegurar que tenemos el artículo enciclopédico COMPLETO (con todas sus secciones, Storyline y Lore)
        texto_bruto_completo = asegurar_articulo_completo_wiki(session, id_principal, texto_bruto_mejor)
        texto_limpio, atributos, secciones = parsear_texto_y_atributos_wiki(texto_bruto_completo)

        conexiones_brutas = [c for c in (record["conexiones"] or []) if c.get("vecino_id")]

        vecinos_distintos = list({c["vecino_id"] for c in conexiones_brutas})
        precargar_urls_imagenes([id_principal] + vecinos_distintos)

        conexiones_unificadas = []
        citas_vistas = set()

        for c in conexiones_brutas:
            vec_canon = MAPA_CANONICO.get(c["vecino_id"].lower(), c["vecino_id"])
            if vec_canon.lower() == id_principal.lower():
                continue

            rol = c.get("rol_vecino", "hijo")
            origen_canon = id_principal if rol == "hijo" else vec_canon
            destino_canon = vec_canon if rol == "hijo" else id_principal
            epoca = c.get("epoca") or "Overview"
            parrafo = (c.get("parrafo") or "").strip()

            clave_cita = (origen_canon.lower(), destino_canon.lower(), epoca.lower(), parrafo[:160].lower())
            if clave_cita in citas_vistas:
                continue
            citas_vistas.add(clave_cita)

            ref_texto_limpio = ""
            if c.get("ref_texto"):
                ref_texto_limpio, _, _ = parsear_texto_y_atributos_wiki(c["ref_texto"])

            info_era = clasificar_era_historica(epoca, parrafo)

            conexiones_unificadas.append({
                "rel_id": c.get("rel_id") or f"{origen_canon}->{destino_canon}#{epoca}",
                "origen": origen_canon,
                "destino": destino_canon,
                "rol_vecino": rol,
                "epoca": epoca,
                "era_id": info_era["era_id"],
                "era_orden": info_era["era_orden"],
                "era_corta": info_era["era_corta"],
                "era_titulo": info_era["era_titulo"],
                "parrafo": parrafo,
                "ref_titulo": c.get("ref_titulo"),
                "ref_url": c.get("ref_url"),
                "ref_texto": ref_texto_limpio,
                "vecino_id": vec_canon,
                "vecino_url": f"{WIKI_BASE_URL}{vec_canon.replace(' ', '_')}",
            })

        return {
            "id": id_principal,
            "url_wiki": f"{WIKI_BASE_URL}{id_principal.replace(' ', '_')}",
            "texto_web": texto_limpio,
            "atributos": atributos,
            "secciones": secciones,
            "conexiones": conexiones_unificadas,
        }


@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
@app.get("/irminsul", response_class=HTMLResponse)
def interfaz_web():
    return r"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Archivo de Teyvat · Enciclopedia de Lore</title>
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- vis-network -->
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <!-- Font Awesome Icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    fontFamily: {
                        sans: ['Plus Jakarta Sans', 'Segoe UI', 'sans-serif'],
                        serif: ['Cinzel', 'Georgia', 'serif']
                    },
                    colors: {
                        gi: {
                            bgDark: '#161b26',
                            panel: '#1f2535',
                            panelLight: '#283044',
                            cream: '#ece5d8',
                            creamDark: '#dcd2c0',
                            gold: '#d3bc8e',
                            goldBright: '#f0dcaf',
                            goldDim: '#9c8964',
                            ink: '#3b4255',
                            anemo: '#74c2a8',
                            cryo: '#9fd6e3',
                            dendro: '#a5c83b',
                            pyro: '#ec7938'
                        }
                    }
                }
            }
        }
    </script>
    <style>
        body {
            background-color: #161b26;
            color: #ece5d8;
            overflow: hidden;
            height: 100vh;
            display: flex;
            flex-direction: column;
            user-select: none;
        }

        /* Fondo estilo Constelación / Astrolabio In-Game de Genshin */
        .gi-constellation-canvas {
            background:
                radial-gradient(circle at 50% 48%, rgba(66, 89, 138, 0.38) 0%, rgba(30, 41, 68, 0.65) 38%, rgba(17, 22, 36, 0.94) 72%, #0c101a 100%);
            position: relative;
        }
        .gi-constellation-canvas::before {
            content: "";
            position: absolute;
            inset: 0;
            pointer-events: none;
            background-image:
                radial-gradient(rgba(255, 244, 214, 0.55) 1px, transparent 1.5px),
                radial-gradient(rgba(211, 188, 142, 0.32) 1px, transparent 1.5px),
                radial-gradient(rgba(159, 214, 227, 0.28) 1px, transparent 1.5px);
            background-size: 95px 95px, 140px 140px, 190px 190px;
            background-position: 0 0, 35px 50px, 70px 20px;
            opacity: 0.75;
        }
        .gi-constellation-canvas::after {
            content: "";
            position: absolute;
            inset: 0;
            pointer-events: none;
            background: radial-gradient(circle at 50% 50%, transparent 55%, rgba(8, 11, 18, 0.65) 100%);
        }

        .astrolabe-ring {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            border-radius: 9999px;
            border: 1px dashed rgba(211, 188, 142, 0.14);
            box-shadow: 0 0 25px rgba(211, 188, 142, 0.05), inset 0 0 25px rgba(211, 188, 142, 0.05);
            pointer-events: none;
        }

        /* Panel lateral estilo Menú de Archivo / Personaje de Genshin */
        .gi-menu-panel {
            background: linear-gradient(180deg, rgba(34, 40, 56, 0.98) 0%, rgba(25, 30, 43, 0.99) 100%);
            border-left: 2px solid rgba(211, 188, 142, 0.45);
            box-shadow: -12px 0 35px rgba(0, 0, 0, 0.65), inset 3px 0 0 rgba(211, 188, 142, 0.12);
            transition: width 0.25s ease;
        }

        /* Botón cápsula crema icónico de Genshin Impact */
        .gi-btn-cream {
            background: linear-gradient(180deg, #f4efe6 0%, #e5dec9 100%);
            color: #3b4255;
            border: 2px solid #d3bc8e;
            border-radius: 9999px;
            font-weight: 700;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.35), inset 0 0 0 1px rgba(255, 255, 255, 0.6);
            transition: all 0.15s ease;
        }
        .gi-btn-cream:hover {
            background: #ffffff;
            border-color: #f0dcaf;
            transform: translateY(-1px);
            box-shadow: 0 4px 14px rgba(211, 188, 142, 0.4);
        }

        /* Botón cápsula oscuro translúcido de los menús superiores de Genshin */
        .gi-btn-dark {
            background: rgba(23, 28, 41, 0.85);
            color: #ece5d8;
            border: 1.5px solid rgba(211, 188, 142, 0.42);
            border-radius: 9999px;
            font-weight: 600;
            transition: all 0.15s ease;
        }
        .gi-btn-dark:hover {
            background: rgba(45, 54, 77, 0.95);
            border-color: #d3bc8e;
            color: #ffffff;
        }

        /* Caja pergamino / tarjeta interior estilo Archivo de Genshin */
        .gi-card-inner {
            background: rgba(22, 26, 38, 0.78);
            border: 1px solid rgba(211, 188, 142, 0.32);
            box-shadow: inset 0 0 0 1px rgba(211, 188, 142, 0.06);
            position: relative;
        }

        /* Caja clara tipo libro de historia de Genshin para lectura enciclopédica */
        .gi-parchment-box {
            background: linear-gradient(180deg, #f5f0e6 0%, #eae2d2 100%);
            color: #333a4d;
            border: 1.5px solid #c8b082;
            box-shadow: inset 0 0 0 2px rgba(255, 255, 255, 0.5), 0 4px 12px rgba(0, 0, 0, 0.25);
            user-select: text;
        }

        /* Filas de estadísticas / atributos estilo pantalla de Atributos de Personaje */
        .gi-stat-row {
            background: rgba(42, 49, 68, 0.65);
            border-left: 3px solid #d3bc8e;
            user-select: text;
        }
        .gi-stat-row:hover {
            background: rgba(56, 65, 88, 0.85);
        }

        ::-webkit-scrollbar { width: 7px; height: 7px; }
        ::-webkit-scrollbar-track { background: rgba(20, 24, 34, 0.6); }
        ::-webkit-scrollbar-thumb { background: #6e6555; border-radius: 9999px; }
        ::-webkit-scrollbar-thumb:hover { background: #d3bc8e; }
    </style>
</head>
<body>

    <!-- CABECERA ESTILO MENÚ SUPERIOR IN-GAME DE GENSHIN IMPACT -->
    <header class="bg-[#1b202e]/95 border-b-2 border-gi-gold/40 px-5 py-2.5 shrink-0 z-30 shadow-[0_4px_20px_rgba(0,0,0,0.6)]">
        <div class="max-w-[1800px] mx-auto flex flex-wrap items-center justify-between gap-4">
            <div class="flex items-center gap-3.5">
                <div class="w-10 h-10 rounded-full bg-[#262d40] border-2 border-gi-gold flex items-center justify-center shadow-[0_0_12px_rgba(211,188,142,0.3)]">
                    <span class="text-gi-goldBright text-lg leading-none">✦</span>
                </div>
                <div>
                    <div class="flex items-center gap-2.5">
                        <h1 class="font-serif text-lg md:text-xl font-bold text-gi-cream tracking-wide">
                            ARCHIVO DE TEYVAT
                        </h1>
                        <span class="text-[11px] px-2.5 py-0.5 rounded-full bg-gi-gold/15 text-gi-goldBright border border-gi-gold/40 font-semibold">
                            Enciclopedia Completa
                        </span>
                    </div>
                    <p class="text-[11px] text-[#9aa3b8]">Todos los capítulos, historias de personaje, atributos y referencias integradas para tu libro</p>
                </div>
            </div>

            <!-- Controles estilo Píldoras de Genshin Impact -->
            <div class="flex items-center gap-2.5 flex-wrap">
                <div class="relative flex items-center">
                    <i class="fa-solid fa-magnifying-glass absolute left-3.5 text-gi-gold text-xs"></i>
                    <input type="text" id="searchInput" list="entitiesList" placeholder="Buscar en la Enciclopedia..." value="" oninput="actualizarSugerencias(this.value)" class="bg-[#131722]/90 border-[1.5px] border-gi-gold/45 rounded-full pl-9 pr-4 py-1.5 text-xs text-gi-cream placeholder-[#7c8599] focus:outline-none focus:border-gi-goldBright w-60 transition">
                    <datalist id="entitiesList"></datalist>
                </div>

                <button onclick="buscarEntidadInput()" class="gi-btn-cream px-4 py-1.5 text-xs flex items-center gap-2">
                    <span class="w-4 h-4 rounded-full bg-[#3b4255] text-gi-goldBright flex items-center justify-center text-[9px]">✦</span>
                    Buscar
                </button>

                <!-- Conmutador Principal: Grafo de Constelación vs. Línea del Tiempo de Teyvat -->
                <button onclick="alternarModoPrincipal()" id="btnModoTimelineTeyvat" class="gi-btn-cream px-3.5 py-1.5 text-xs flex items-center gap-1.5" title="Alternar entre el Grafo de Constelación y la Línea del Tiempo de Teyvat">
                    <i class="fa-solid fa-hourglass-half text-[#3b4255] text-[11px]"></i>
                    <span id="labelModoTimelineTeyvat">Línea del Tiempo de Teyvat</span>
                </button>

                <button onclick="alternarModoArbol()" id="btnLayout" class="gi-btn-dark px-3.5 py-1.5 text-xs flex items-center gap-1.5">
                    <i class="fa-solid fa-sitemap text-gi-gold text-[11px]"></i>
                    <span>Vista: Constelación</span>
                </button>

                <select id="maxHijosSelect" onchange="recargarNodoActual()" class="gi-btn-dark px-3.5 py-1.5 text-xs focus:outline-none cursor-pointer">
                    <option value="15" class="bg-[#1b202e] text-gi-cream">Máx. 15 conexiones</option>
                    <option value="25" selected class="bg-[#1b202e] text-gi-cream">Máx. 25 conexiones</option>
                    <option value="45" class="bg-[#1b202e] text-gi-cream">Máx. 45 conexiones</option>
                    <option value="100" class="bg-[#1b202e] text-gi-cream">Todas las conexiones</option>
                </select>

                <button onclick="reiniciarLienzo()" class="gi-btn-dark px-3.5 py-1.5 text-xs flex items-center gap-1.5">
                    <i class="fa-solid fa-rotate-left text-gi-gold text-[11px]"></i>
                    <span>Reiniciar</span>
                </button>
            </div>
        </div>
    </header>

    <!-- CONTENIDO PRINCIPAL: MAPA DE CONSTELACIÓN + ENCICLOPEDIA -->
    <main class="flex-1 min-h-0 flex flex-col lg:flex-row relative overflow-hidden">

        <!-- LIENZO DE CONSTELACIÓN (vis.js) -->
        <div class="flex-1 min-h-0 gi-constellation-canvas overflow-hidden" id="graphViewport">
            <div class="astrolabe-ring w-[420px] h-[420px]"></div>
            <div class="astrolabe-ring w-[760px] h-[760px]"></div>

            <div id="network" class="w-full h-full absolute inset-0 z-10" oncontextmenu="return false;"></div>

            <!-- Guía rápida de controles del camino -->
            <div class="absolute bottom-3 right-3 z-20 pointer-events-none bg-[#141824]/85 border border-gi-gold/30 rounded-full px-3.5 py-1 text-[11px] text-gi-cream/80 flex items-center gap-3 shadow">
                <span><strong class="text-gi-goldBright">Clic derecho en nodo:</strong> Cerrar entidad</span>
                <span class="text-gi-gold/40">|</span>
                <span><strong class="text-gi-goldBright">Clic en Vínculos (derecha):</strong> Añadir solo esa entidad al camino</span>
            </div>

            <!-- Indicador de carga discreto estilo Genshin -->
            <div id="loadingOverlay" class="hidden absolute bottom-5 left-5 z-20 bg-[#1b202e]/95 rounded-full px-4 py-2 text-xs text-gi-cream flex items-center gap-2.5 border border-gi-gold/50 shadow-lg">
                <i class="fa-solid fa-spinner fa-spin text-gi-goldBright"></i>
                <span>Descargando capítulos completos del Archivo...</span>
            </div>
        </div>

        <!-- VISTA ALTERNATIVA: LÍNEA DEL TIEMPO GENERAL DE TEYVAT -->
        <div id="teyvatTimelineViewport" class="hidden flex-1 min-h-0 gi-constellation-canvas flex flex-col md:flex-row overflow-hidden relative">
            <!-- Columna izquierda: Las 12 Grandes Épocas de Teyvat -->
            <div class="w-full md:w-72 lg:w-80 bg-[#161b28]/95 border-b md:border-b-0 md:border-r border-gi-gold/35 flex flex-col shrink-0 z-10">
                <div class="px-4 py-3 border-b border-gi-gold/30 flex items-center justify-between">
                    <div class="flex items-center gap-2">
                        <span class="text-gi-goldBright text-sm">⏳</span>
                        <span class="font-serif font-bold text-xs uppercase tracking-wider text-gi-goldBright">Épocas de Teyvat</span>
                    </div>
                    <button onclick="alternarModoPrincipal('grafo')" class="px-2.5 py-1 rounded-full bg-[#232b3e] hover:bg-gi-gold/25 border border-gi-gold/40 text-[10px] text-gi-cream transition flex items-center gap-1">
                        <i class="fa-solid fa-arrow-left text-[9px] text-gi-gold"></i> Volver al Grafo
                    </button>
                </div>
                <div id="listaErasTeyvatSidebar" class="p-3 overflow-y-auto space-y-1.5 flex-1 min-h-0"></div>
            </div>

            <!-- Área central: Entidades protagonistas y Hechos Históricos de la Época seleccionada -->
            <div id="detalleEraTeyvatContent" class="flex-1 min-h-0 overflow-y-auto p-5 space-y-5 z-10">
                <div class="text-center py-16 text-gi-cream/70 space-y-2">
                    <i class="fa-solid fa-spinner fa-spin text-2xl text-gi-goldBright"></i>
                    <p class="text-sm">Cargando la Crónica Histórica de Teyvat...</p>
                </div>
            </div>
        </div>

        <!-- PANEL DERECHO: MODO ENCICLOPEDIA COMPLETA -->
        <aside id="inspectorPanel" class="w-full lg:w-[580px] xl:w-[660px] min-h-0 gi-menu-panel flex flex-col z-20 shrink-0">
            <!-- Barra superior ornamental del panel con botón para ampliar el Libro -->
            <div class="px-5 py-3 border-b border-gi-gold/30 flex items-center justify-between bg-[#181d2a]/90 shrink-0 gap-2">
                <div class="flex items-center gap-2 min-w-0">
                    <span class="text-gi-gold text-sm">✦</span>
                    <span id="inspectorTypeBadge" class="text-xs font-serif font-bold uppercase tracking-widest text-gi-goldBright shrink-0">
                        ENCICLOPEDIA
                    </span>
                    <span class="text-gi-gold/40">·</span>
                    <span id="inspectorIdLabel" class="text-xs text-gi-cream/80 truncate font-medium">Listo</span>
                </div>
                <div class="flex items-center gap-2 shrink-0">
                    <button onclick="alternarAnchoEnciclopedia()" id="btnExpandReader" class="gi-btn-dark px-3 py-1 text-[11px] flex items-center gap-1.5" title="Ampliar el panel de lectura para redactar o leer toda la página">
                        <i class="fa-solid fa-book-open text-gi-gold text-[10px]"></i>
                        <span id="labelExpandReader">Ampliar Lectura</span>
                    </button>
                    <a id="inspectorWikiLink" href="#" target="_blank" rel="noopener noreferrer" class="gi-btn-dark px-3 py-1 text-[11px] flex items-center gap-1.5">
                        <span>Wiki</span>
                        <i class="fa-solid fa-arrow-up-right-from-square text-[9px] text-gi-gold"></i>
                    </a>
                </div>
            </div>

            <!-- Contenido desplazable del Archivo -->
            <div id="inspectorContent" class="p-5 overflow-y-auto space-y-5 flex-1 min-h-0">
                <div class="text-center py-14 text-gi-cream/60 space-y-2">
                    <div class="text-2xl text-gi-gold">✦</div>
                    <p class="text-sm">Selecciona un nodo en la constelación para abrir todos los capítulos de su página.</p>
                </div>
            </div>
        </aside>
    </main>

    <!-- VENTANA FLOTANTE DEL CHATBOT (Estilo Telegram / Facebook Messenger) -->
    <div class="fixed bottom-4 left-4 z-50 flex flex-col items-start pointer-events-none">
        <!-- Ventana de conversación desplegada -->
        <div id="ventanaFlotanteChat" class="hidden pointer-events-auto w-[370px] sm:w-[430px] h-[530px] max-h-[82vh] max-w-[94vw] rounded-2xl bg-[#161b28]/98 border-2 border-gi-gold/70 shadow-[0_12px_42px_rgba(0,0,0,0.85)] flex flex-col overflow-hidden mb-2.5 transition-all duration-200">
            <!-- Cabecera de la ventana de chat -->
            <div class="px-4 py-2.5 bg-gradient-to-r from-[#242d42] via-[#1c2335] to-[#161b28] border-b border-gi-gold/40 flex items-center justify-between gap-2 shrink-0 select-none">
                <div class="flex items-center gap-2.5 min-w-0">
                    <div class="w-7 h-7 rounded-full bg-gi-gold/20 border border-gi-gold flex items-center justify-center text-gi-goldBright text-xs shadow-[0_0_10px_rgba(240,220,175,0.4)] shrink-0">
                        ✦
                    </div>
                    <div class="min-w-0">
                        <div class="font-serif font-bold text-xs text-gi-goldBright tracking-wide truncate">
                            Oráculo del Irminsul
                        </div>
                        <div class="text-[10px] text-[#9fd6e3] flex items-center gap-1">
                            <span class="w-1.5 h-1.5 rounded-full bg-emerald-400 inline-block"></span>
                            <span>GraphRAG · Fuentes Wiki</span>
                        </div>
                    </div>
                </div>
                <div class="flex items-center gap-1 shrink-0">
                    <button onclick="alternarTamanoVentanaChat()" id="btnTamanoChat" class="w-6 h-6 rounded-full hover:bg-white/10 text-gi-cream/80 hover:text-gi-goldBright flex items-center justify-center text-[11px] transition" title="Ampliar / Reducir ventana de chat">
                        <i class="fa-solid fa-up-right-and-down-left-from-center"></i>
                    </button>
                    <button onclick="limpiarHistorialChat()" class="w-6 h-6 rounded-full hover:bg-white/10 text-gi-cream/80 hover:text-gi-goldBright flex items-center justify-center text-[11px] transition" title="Limpiar conversación">
                        <i class="fa-solid fa-trash-can"></i>
                    </button>
                    <button onclick="alternarVentanaChat(false)" class="w-6 h-6 rounded-full hover:bg-white/10 text-gi-cream/80 hover:text-gi-goldBright flex items-center justify-center text-xs transition" title="Minimizar chat">
                        <i class="fa-solid fa-minus"></i>
                    </button>
                </div>
            </div>

            <!-- Cuerpo de mensajes -->
            <div id="contenedorMensajesChat" class="flex-1 min-h-0 overflow-y-auto p-3.5 space-y-3 bg-[#121622]/95"></div>

            <!-- Barra inferior de escritura -->
            <div class="p-2.5 bg-[#181d2b] border-t border-gi-gold/35 flex items-center gap-2 shrink-0">
                <input type="text" id="chatInputPregunta" onkeydown="if(event.key==='Enter') enviarPreguntaChat()" placeholder="Escribe tu pregunta al Oráculo..." class="flex-1 bg-[#111520] border border-gi-gold/40 rounded-full px-3.5 py-1.5 text-xs text-gi-cream placeholder-[#7c8599] focus:outline-none focus:border-gi-goldBright">
                <button id="btnEnviarChat" onclick="enviarPreguntaChat()" class="gi-btn-cream px-3.5 py-1.5 text-xs flex items-center gap-1 shrink-0">
                    <i class="fa-solid fa-paper-plane text-[10px] text-[#3b4255]"></i>
                    <span>Enviar</span>
                </button>
            </div>
        </div>

        <!-- Burbuja flotante para abrir/minimizar el chat -->
        <button id="btnBurbujaChat" onclick="alternarVentanaChat()" class="pointer-events-auto gi-btn-cream px-4 py-2.5 rounded-full shadow-[0_6px_24px_rgba(0,0,0,0.75)] flex items-center gap-2 text-xs font-serif font-bold border-2 border-gi-gold">
            <i class="fa-solid fa-comments text-[#3b4255] text-sm"></i>
            <span>Oráculo IA (Chat)</span>
        </button>
    </div>

    <script>
        const nodes = new vis.DataSet([]);
        const edges = new vis.DataSet([]);
        let cacheNodos = {};
        let cacheAristas = {};
        let expandidos = new Set();
        let nodoActualId = null;
        let modoJerarquico = false;
        let mostrarEtiquetasSiempre = false;
        let modoLecturaAncho = 0; // 0: normal, 1: 68% pantalla, 2: pantalla completa
        let seccionActivaIdx = -1; // -1: mostrar todas las secciones
        let chatVentanaAbierta = false;
        let chatVentanaGrande = false;
        let modoPrincipalIzquierdo = "grafo"; // 'grafo' | 'timeline'
        let datosTimelineTeyvat = [];
        let eraTeyvatActivaIdx = 0;
        let regionTeyvatFiltro = "TODAS";
        let historialChat = [];

        const CATALOGO_ERAS = [
            { id: "era_1_antigua", orden: 1, corta: "I. Mundo Antiguo", titulo: "I. Mundo Antiguo y Era Primordial" },
            { id: "era_2_arcontes", orden: 2, corta: "II. Guerra de Arcontes", titulo: "II. Guerra de los Arcontes y Era de los Dioses" },
            { id: "era_3_cataclismo", orden: 3, corta: "III. El Cataclismo", titulo: "III. El Cataclismo y Caída de Khaenri'ah (Hace 500 años)" },
            { id: "era_4_pasado", orden: 4, corta: "IV. Historia y Pasado", titulo: "IV. Historia Previa, Orígenes y Legado" },
            { id: "era_5_presente", orden: 5, corta: "V. Era Presente", titulo: "V. Era Presente y Trama Actual en Teyvat" },
            { id: "era_6_archivo", orden: 6, corta: "VI. Archivo y Perfil", titulo: "VI. Perfil General, Historias de Personaje y Lore" }
        ];

        const container = document.getElementById('network');
        const network = new vis.Network(container, { nodes, edges }, obtenerOpcionesVis(false));

        function urlAvatar(id) {
            return `/api/avatar/${encodeURIComponent(id)}`;
        }

        function idAristaUnica(origen, destino) {
            return `${origen}➔${destino}`;
        }

        function escaparComillas(str) {
            return (str || '').replace(/\\/g, '\\\\').replace(/'/g, "\\'");
        }

        function escaparHtml(str) {
            return (str || '')
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;');
        }

        function alternarAnchoEnciclopedia() {
            const panel = document.getElementById('inspectorPanel');
            const viewport = document.getElementById('graphViewport');
            const label = document.getElementById('labelExpandReader');
            modoLecturaAncho = (modoLecturaAncho + 1) % 3;

            if (modoLecturaAncho === 1) {
                viewport.classList.remove('hidden');
                panel.className = "w-full lg:w-[68%] xl:w-[72%] min-h-0 gi-menu-panel flex flex-col z-20 shrink-0";
                label.textContent = "Pantalla Completa";
            } else if (modoLecturaAncho === 2) {
                viewport.classList.add('hidden');
                panel.className = "w-full min-h-0 gi-menu-panel flex flex-col z-20 shrink-0";
                label.textContent = "Volver al Grafo";
            } else {
                viewport.classList.remove('hidden');
                panel.className = "w-full lg:w-[580px] xl:w-[660px] min-h-0 gi-menu-panel flex flex-col z-20 shrink-0";
                label.textContent = "Ampliar Lectura";
            }
            setTimeout(() => { network.redraw(); network.fit(); }, 260);
        }

        function copiarTextoAlPortapapeles(texto, btnElem) {
            navigator.clipboard.writeText(texto).then(() => {
                const orig = btnElem.innerHTML;
                btnElem.innerHTML = '<i class="fa-solid fa-check text-emerald-700"></i> Copiado';
                setTimeout(() => { btnElem.innerHTML = orig; }, 1600);
            });
        }

        function etiquetaResumenArista(aristaObj) {
            if (!aristaObj || !aristaObj.relaciones || !aristaObj.relaciones.length) return 'Relación';
            if (aristaObj.relaciones.length === 1) {
                return aristaObj.relaciones[0].epoca || 'Relación';
            }
            return `${aristaObj.relaciones.length} citas`;
        }

        function obtenerOpcionesVis(jerarquico) {
            return {
                layout: jerarquico ? {
                    hierarchical: {
                        enabled: true,
                        direction: 'UD',
                        sortMethod: 'directed',
                        levelSeparation: 195,
                        nodeSpacing: 175,
                        treeSpacing: 225
                    }
                } : { hierarchical: false },
                nodes: {
                    shape: 'circularImage',
                    size: 28,
                    borderWidth: 3,
                    borderWidthSelected: 4.5,
                    font: {
                        color: '#f4efe6',
                        size: 12,
                        face: 'Plus Jakarta Sans, Segoe UI, sans-serif',
                        strokeWidth: 4,
                        strokeColor: '#0e1320',
                        vadjust: 8
                    },
                    shadow: {
                        enabled: true,
                        color: 'rgba(240, 220, 175, 0.45)',
                        size: 18,
                        x: 0,
                        y: 0
                    }
                },
                edges: {
                    arrows: {
                        to: { enabled: false }
                    },
                    color: {
                        color: 'rgba(240, 220, 175, 0.45)',
                        hover: '#fff6dc',
                        highlight: '#ffffff'
                    },
                    width: 1.1,
                    selectionWidth: 1.8,
                    hoverWidth: 1.5,
                    shadow: {
                        enabled: true,
                        color: 'rgba(240, 220, 175, 0.28)',
                        size: 5,
                        x: 0,
                        y: 0
                    },
                    font: {
                        color: '#f0dcaf',
                        size: 10,
                        face: 'Cinzel, Plus Jakarta Sans, sans-serif',
                        background: 'rgba(18, 23, 36, 0.92)',
                        strokeWidth: 2,
                        strokeColor: '#0e1320',
                        align: 'horizontal'
                    },
                    smooth: jerarquico
                        ? { type: 'cubicBezier', roundness: 0.25 }
                        : false
                },
                interaction: {
                    hover: true,
                    tooltipDelay: 120,
                    hideEdgesOnDrag: false
                },
                physics: jerarquico ? false : {
                    barnesHut: {
                        gravitationalConstant: -8200,
                        centralGravity: 0.24,
                        springLength: 205,
                        springConstant: 0.035,
                        damping: 0.16,
                        avoidOverlap: 0.72
                    },
                    stabilization: { iterations: 120, updateInterval: 25 }
                }
            };
        }

        function estiloNodo(id, rol) {
            const esFocoPrincipal = (id.toLowerCase() === (nodoActualId || '').toLowerCase());
            const fueExpandido = expandidos.has(id);

            if (esFocoPrincipal) {
                return {
                    rolConstelacion: 'central',
                    size: 46,
                    borderWidth: 4,
                    borderWidthSelected: 5,
                    color: {
                        background: '#222a3d',
                        border: '#fff3d1',
                        highlight: { background: '#2d3750', border: '#ffffff' },
                        hover: { background: '#2d3750', border: '#ffffff' }
                    },
                    font: {
                        color: '#fff3d1',
                        size: 15,
                        face: 'Cinzel, serif',
                        bold: true,
                        strokeWidth: 5,
                        strokeColor: '#0c101a',
                        vadjust: 10
                    },
                    shadow: {
                        enabled: true,
                        color: 'rgba(245, 224, 175, 0.85)',
                        size: 32,
                        x: 0,
                        y: 0
                    }
                };
            }

            if (fueExpandido) {
                return {
                    rolConstelacion: 'expandido',
                    size: 35,
                    borderWidth: 3.5,
                    color: {
                        background: '#222a3d',
                        border: '#f0dcaf',
                        highlight: { background: '#2d3750', border: '#ffffff' },
                        hover: { background: '#2d3750', border: '#fff3d1' }
                    },
                    font: {
                        color: '#f0dcaf',
                        size: 13,
                        face: 'Cinzel, serif',
                        strokeWidth: 4,
                        strokeColor: '#0c101a',
                        vadjust: 8
                    },
                    shadow: {
                        enabled: true,
                        color: 'rgba(240, 220, 175, 0.6)',
                        size: 22,
                        x: 0,
                        y: 0
                    }
                };
            }

            if (rol === 'padre') {
                return {
                    rolConstelacion: 'padre',
                    size: 27,
                    borderWidth: 2.8,
                    color: {
                        background: '#192636',
                        border: '#9fd6e3',
                        highlight: { background: '#23354a', border: '#e6f9ff' },
                        hover: { background: '#23354a', border: '#c9f0fa' }
                    },
                    font: {
                        color: '#e6f4f8',
                        size: 12,
                        face: 'Plus Jakarta Sans, sans-serif',
                        strokeWidth: 4,
                        strokeColor: '#0c101a',
                        vadjust: 7
                    },
                    shadow: {
                        enabled: true,
                        color: 'rgba(159, 214, 227, 0.5)',
                        size: 18,
                        x: 0,
                        y: 0
                    }
                };
            }

            return {
                rolConstelacion: 'hijo',
                size: 27,
                borderWidth: 2.8,
                color: {
                    background: '#26221c',
                    border: '#e5cf9e',
                    highlight: { background: '#363026', border: '#fff8e7' },
                    hover: { background: '#363026', border: '#f5e4be' }
                },
                font: {
                    color: '#f4efe6',
                    size: 12,
                    face: 'Plus Jakarta Sans, sans-serif',
                    strokeWidth: 4,
                    strokeColor: '#0c101a',
                    vadjust: 7
                },
                shadow: {
                    enabled: true,
                    color: 'rgba(229, 207, 158, 0.48)',
                    size: 18,
                    x: 0,
                    y: 0
                }
            };
        }

        function alternarModoArbol() {
            modoJerarquico = !modoJerarquico;
            document.getElementById('btnLayout').innerHTML = modoJerarquico
                ? '<i class="fa-solid fa-sitemap text-gi-gold text-[11px]"></i> <span>Vista: Jerarquía</span>'
                : '<i class="fa-solid fa-sitemap text-gi-gold text-[11px]"></i> <span>Vista: Constelación</span>';
            network.setOptions(obtenerOpcionesVis(modoJerarquico));
            setTimeout(() => { network.fit({ animation: true }); }, 200);
        }

        function alternarEtiquetasAristas() {
            mostrarEtiquetasSiempre = !mostrarEtiquetasSiempre;
            document.getElementById('btnEdgeLabels').innerHTML = mostrarEtiquetasSiempre
                ? '<i class="fa-solid fa-tag text-gi-gold text-[11px]"></i> <span>Épocas: Siempre</span>'
                : '<i class="fa-solid fa-tag text-gi-gold text-[11px]"></i> <span>Épocas: Al seleccionar</span>';
            const actualizaciones = edges.get().map(e => {
                const arista = cacheAristas[e.id];
                return {
                    id: e.id,
                    label: mostrarEtiquetasSiempre ? etiquetaResumenArista(arista) : ''
                };
            });
            edges.update(actualizaciones);
        }

        async function actualizarSugerencias(texto) {
            try {
                const res = await fetch(`/api/buscar?q=${encodeURIComponent(texto)}`);
                const lista = await res.json();
                document.getElementById('entitiesList').innerHTML = lista.map(id => `<option value="${id}">`).join('');
            } catch (e) {
                console.error("Error al actualizar sugerencias:", e);
            }
        }

        function registrarRelacionesEnCacheAristas(listaRelaciones) {
            listaRelaciones.forEach(c => {
                const edgeId = idAristaUnica(c.origen, c.destino);
                if (!cacheAristas[edgeId]) {
                    cacheAristas[edgeId] = {
                        id: edgeId,
                        origen: c.origen,
                        destino: c.destino,
                        relaciones: []
                    };
                }
                const yaExiste = cacheAristas[edgeId].relaciones.some(
                    r => (r.epoca === c.epoca && r.parrafo === c.parrafo)
                );
                if (!yaExiste) {
                    cacheAristas[edgeId].relaciones.push(c);
                }
            });
        }

        function guardarDatosNodoEnCache(data) {
            const mapaPadres = new Map();
            const mapaHijos = new Map();

            const conexiones = data.conexiones || [];
            registrarRelacionesEnCacheAristas(conexiones);

            conexiones.forEach(c => {
                if (!c.vecino_id) return;
                const mapaDestino = (c.rol_vecino === 'padre') ? mapaPadres : mapaHijos;
                const key = c.vecino_id.toLowerCase();
                if (!mapaDestino.has(key)) {
                    mapaDestino.set(key, {
                        vecino_id: c.vecino_id,
                        vecino_url: c.vecino_url,
                        rol_vecino: c.rol_vecino,
                        origen: c.origen,
                        destino: c.destino,
                        relaciones: []
                    });
                }
                mapaDestino.get(key).relaciones.push(c);
            });

            const padresUnicos = Array.from(mapaPadres.values()).sort((a, b) => a.vecino_id.localeCompare(b.vecino_id));
            const hijosUnicos = Array.from(mapaHijos.values()).sort((a, b) => a.vecino_id.localeCompare(b.vecino_id));

            cacheNodos[data.id] = {
                id: data.id,
                url: data.url_wiki,
                texto: data.texto_web,
                atributos: data.atributos || [],
                secciones: data.secciones || [],
                padres: padresUnicos,
                hijos: hijosUnicos,
                cargadoCompleto: true
            };
            return { padres: padresUnicos, hijos: hijosUnicos };
        }

        function crearPropsArista(grupo) {
            const edgeId = idAristaUnica(grupo.origen, grupo.destino);
            const aristaObj = cacheAristas[edgeId];
            const numCitas = aristaObj ? aristaObj.relaciones.length : grupo.relaciones.length;
            const epocasLista = (aristaObj ? aristaObj.relaciones : grupo.relaciones)
                .map(r => r.epoca || 'Overview')
                .filter((v, idx, a) => a.indexOf(v) === idx)
                .join(', ');

            const esEntrante = (grupo.rol_vecino === 'padre');
            const colorArista = esEntrante
                ? {
                    color: 'rgba(159, 214, 227, 0.44)',
                    hover: '#d4f6ff',
                    highlight: '#ffffff'
                }
                : {
                    color: 'rgba(240, 220, 175, 0.48)',
                    hover: '#fff6dc',
                    highlight: '#ffffff'
                };
            const sombraArista = esEntrante
                ? {
                    enabled: true,
                    color: 'rgba(142, 209, 230, 0.25)',
                    size: 5,
                    x: 0,
                    y: 0
                }
                : {
                    enabled: true,
                    color: 'rgba(240, 220, 175, 0.28)',
                    size: 5,
                    x: 0,
                    y: 0
                };

            return {
                id: edgeId,
                from: grupo.origen,
                to: grupo.destino,
                esEntrante: esEntrante,
                width: Math.min(1.0 + (numCitas - 1) * 0.16, 1.9),
                color: colorArista,
                shadow: sombraArista,
                label: mostrarEtiquetasSiempre ? etiquetaResumenArista(aristaObj) : '',
                title: `${grupo.origen} ➔ ${grupo.destino} · ${numCitas} ${numCitas === 1 ? 'registro' : 'registros'} (${epocasLista})`
            };
        }

        function cerrarNodoDelGrafo(nodoId) {
            if (!nodoId || !nodes.get(nodoId)) return;
            const aristasConectadas = edges.get({
                filter: e => e.from === nodoId || e.to === nodoId
            });
            if (aristasConectadas.length) {
                edges.remove(aristasConectadas.map(e => e.id));
            }
            nodes.remove(nodoId);
            expandidos.delete(nodoId);

            if (nodoActualId === nodoId) {
                const restantes = nodes.getIds();
                if (restantes.length > 0) {
                    nodoActualId = restantes[restantes.length - 1];
                    mostrarFichaNodo(nodoActualId);
                } else {
                    nodoActualId = null;
                }
            }
            network.redraw();
        }

        // Abre y añade SOLO esta entidad al grafo (conectándola únicamente con los nodos que ya están visibles en tu camino)
        async function abrirSoloEntidadEnCamino(entidadId, limpiarAntes = false) {
            if (!entidadId) return;
            const prevNodoActual = nodoActualId;
            nodoActualId = entidadId;
            document.getElementById('loadingOverlay').classList.remove('hidden');

            try {
                const res = await fetch(`/api/nodo/${encodeURIComponent(entidadId)}`);
                const data = await res.json();
                if (data.error) {
                    alert(data.error);
                    return;
                }

                nodoActualId = data.id;
                document.getElementById('searchInput').value = data.id;

                if (limpiarAntes) {
                    nodes.clear();
                    edges.clear();
                    expandidos.clear();
                } else if (prevNodoActual && prevNodoActual !== data.id && nodes.get(prevNodoActual)) {
                    nodes.update({
                        id: prevNodoActual,
                        ...estiloNodo(prevNodoActual, 'expandido')
                    });
                }

                expandidos.add(data.id);
                const { padres, hijos } = guardarDatosNodoEnCache(data);

                nodes.update({
                    id: data.id,
                    label: data.id,
                    image: urlAvatar(data.id),
                    ...estiloNodo(data.id, 'central')
                });

                // Conectar únicamente con los nodos que YA están visibles en el lienzo
                const todosVecinos = [...padres, ...hijos];
                todosVecinos.forEach(grupo => {
                    const vId = grupo.vecino_id;
                    if (!cacheNodos[vId]) {
                        cacheNodos[vId] = {
                            id: vId,
                            url: grupo.vecino_url,
                            texto: "",
                            atributos: [],
                            secciones: [],
                            padres: [],
                            hijos: [],
                            cargadoCompleto: false
                        };
                    }
                    if (nodes.get(vId)) {
                        const propsArista = crearPropsArista(grupo);
                        if (!edges.get(propsArista.id)) {
                            edges.add(propsArista);
                        } else {
                            edges.update(propsArista);
                        }
                    }
                });

                await mostrarFichaNodo(data.id);

                setTimeout(() => {
                    network.redraw();
                    network.focus(data.id, { scale: Math.max(network.getScale(), 0.85), animation: { duration: 400, easingFunction: 'easeInOutQuad' } });
                }, 60);
            } catch (e) {
                console.error("Error al abrir entidad en el camino:", e);
            } finally {
                document.getElementById('loadingOverlay').classList.add('hidden');
            }
        }

        async function expandirEntidad(entidadId, limpiarAntes = false) {
            if (!entidadId) return;
            const prevNodoActual = nodoActualId;
            nodoActualId = entidadId;
            document.getElementById('loadingOverlay').classList.remove('hidden');

            try {
                const res = await fetch(`/api/nodo/${encodeURIComponent(entidadId)}`);
                const data = await res.json();
                if (data.error) {
                    alert(data.error);
                    return;
                }

                nodoActualId = data.id;
                document.getElementById('searchInput').value = data.id;

                if (limpiarAntes) {
                    nodes.clear();
                    edges.clear();
                    expandidos.clear();
                } else if (prevNodoActual && prevNodoActual !== data.id && nodes.get(prevNodoActual)) {
                    nodes.update({
                        id: prevNodoActual,
                        ...estiloNodo(prevNodoActual, 'central')
                    });
                }

                expandidos.add(data.id);
                const maxHijosPermitidos = parseInt(document.getElementById('maxHijosSelect').value) || 25;

                const { padres, hijos } = guardarDatosNodoEnCache(data);
                const hijosLimitados = hijos.slice(0, maxHijosPermitidos);

                nodes.update({
                    id: data.id,
                    label: data.id,
                    image: urlAvatar(data.id),
                    ...estiloNodo(data.id, 'central')
                });

                const gruposVecinosAPintar = [...padres, ...hijosLimitados];

                gruposVecinosAPintar.forEach(grupo => {
                    const vId = grupo.vecino_id;
                    if (!cacheNodos[vId]) {
                        cacheNodos[vId] = {
                            id: vId,
                            url: grupo.vecino_url,
                            texto: "",
                            atributos: [],
                            secciones: [],
                            padres: [],
                            hijos: [],
                            cargadoCompleto: false
                        };
                    }

                    if (!nodes.get(vId)) {
                        nodes.add({
                            id: vId,
                            label: vId,
                            image: urlAvatar(vId),
                            ...estiloNodo(vId, grupo.rol_vecino)
                        });
                    } else if (expandidos.has(vId)) {
                        nodes.update({
                            id: vId,
                            ...estiloNodo(vId, 'central')
                        });
                    }

                    const propsArista = crearPropsArista(grupo);
                    if (!edges.get(propsArista.id)) {
                        edges.add(propsArista);
                    } else {
                        edges.update(propsArista);
                    }
                });

                await mostrarFichaNodo(data.id);

                setTimeout(() => {
                    network.redraw();
                    network.fit({ animation: { duration: 450, easingFunction: 'easeInOutQuad' } });
                }, 80);

            } catch (e) {
                console.error("Error al conectar con la API de Neo4j:", e);
            } finally {
                document.getElementById('loadingOverlay').classList.add('hidden');
            }
        }

        function recargarNodoActual() {
            if (nodoActualId) expandirEntidad(nodoActualId, true);
        }

        async function alternarModoPrincipal(forzarModo = null) {
            const nuevoModo = forzarModo || (modoPrincipalIzquierdo === 'grafo' ? 'timeline' : 'grafo');
            modoPrincipalIzquierdo = nuevoModo;

            const vpGrafo = document.getElementById('graphViewport');
            const vpTimeline = document.getElementById('teyvatTimelineViewport');
            const lblBtn = document.getElementById('labelModoTimelineTeyvat');

            if (nuevoModo === 'timeline') {
                vpGrafo.classList.add('hidden');
                vpTimeline.classList.remove('hidden');
                if (lblBtn) lblBtn.textContent = "Ver Grafo de Constelación";
                await cargarLineaTemporalTeyvat();
            } else {
                vpTimeline.classList.add('hidden');
                vpGrafo.classList.remove('hidden');
                if (lblBtn) lblBtn.textContent = "Línea del Tiempo de Teyvat";
                setTimeout(() => { network.redraw(); network.fit(); }, 100);
            }
        }

        async function cargarLineaTemporalTeyvat() {
            if (datosTimelineTeyvat.length > 0) {
                renderizarSidebarErasTeyvat();
                renderizarDetalleEraTeyvat();
                return;
            }
            const contDetalle = document.getElementById('detalleEraTeyvatContent');
            try {
                const res = await fetch('/api/timeline_teyvat');
                const data = await res.json();
                if (data.error) {
                    if (contDetalle) contDetalle.innerHTML = `<div class="p-4 rounded-xl bg-[#181d2b] border border-gi-gold/30 text-xs text-red-300">${escaparHtml(data.error)}</div>`;
                    return;
                }
                datosTimelineTeyvat = data.eras || [];
                eraTeyvatActivaIdx = 0;
                regionTeyvatFiltro = "TODAS";
                renderizarSidebarErasTeyvat();
                renderizarDetalleEraTeyvat();
            } catch (e) {
                if (contDetalle) contDetalle.innerHTML = `<div class="p-4 rounded-xl bg-[#181d2b] border border-gi-gold/30 text-xs text-red-300">Error cargando la línea del tiempo: ${escaparHtml(e.message)}</div>`;
            }
        }

        function seleccionarEraTeyvat(idx) {
            eraTeyvatActivaIdx = idx;
            regionTeyvatFiltro = "TODAS";
            renderizarSidebarErasTeyvat();
            renderizarDetalleEraTeyvat();
            const cont = document.getElementById('detalleEraTeyvatContent');
            if (cont) cont.scrollTop = 0;
        }

        function seleccionarRegionEraTeyvat(region) {
            regionTeyvatFiltro = region;
            renderizarDetalleEraTeyvat();
        }

        function abrirEntidadEnGrafoDesdeTimeline(entidadId) {
            alternarModoPrincipal('grafo');
            expandirEntidad(entidadId, true);
        }

        function renderizarSidebarErasTeyvat() {
            const sidebar = document.getElementById('listaErasTeyvatSidebar');
            if (!sidebar) return;

            sidebar.innerHTML = datosTimelineTeyvat.map((era, idx) => {
                const activo = (idx === eraTeyvatActivaIdx);
                return `
                    <button onclick="seleccionarEraTeyvat(${idx})" class="w-full text-left p-2.5 rounded-xl border transition flex items-start gap-2.5 ${activo ? 'bg-gi-gold text-[#161b26] border-gi-goldBright shadow-[0_0_15px_rgba(240,220,175,0.4)] font-bold' : 'bg-[#1b2234]/85 text-gi-cream border-gi-gold/25 hover:border-gi-gold/60'}">
                        <span class="text-xs mt-0.5 ${activo ? 'text-[#161b26]' : 'text-gi-goldBright'}">✦</span>
                        <div class="min-w-0 flex-1">
                            <div class="font-serif text-xs leading-snug">${escaparHtml(era.titulo_es)}</div>
                            <div class="text-[10px] mt-0.5 ${activo ? 'text-[#2c3345]' : 'text-[#9aa3b8]'} flex items-center justify-between">
                                <span>${era.hechos.length} hechos</span>
                                <span>${era.entidades_destacadas.length} entidades</span>
                            </div>
                        </div>
                    </button>
                `;
            }).join('');
        }

        async function traducirHechoTeyvat(eraIdx, hechoIdx, btnElem) {
            const era = datosTimelineTeyvat[eraIdx];
            if (!era || !era.hechos || !era.hechos[hechoIdx]) return;
            const h = era.hechos[hechoIdx];

            if (h.texto_es) {
                h.mostrar_es = !h.mostrar_es;
                renderizarDetalleEraTeyvat();
                return;
            }

            if (btnElem) {
                btnElem.disabled = true;
                btnElem.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Traduciendo...';
            }

            try {
                h.texto_es = await llamarApiTraducir(h.texto, `${era.titulo_es} - ${h.region}`);
                h.mostrar_es = true;
                renderizarDetalleEraTeyvat();
            } catch (e) {
                alert("Error traduciendo hecho histórico: " + e.message);
                if (btnElem) {
                    btnElem.disabled = false;
                    btnElem.innerHTML = '🌐 Traducir';
                }
            }
        }

        async function traducirTodaEraTeyvat(eraIdx, btnElem) {
            const era = datosTimelineTeyvat[eraIdx];
            if (!era || !era.hechos) return;
            const hechosVisibles = era.hechos
                .map((h, i) => ({ h, i }))
                .filter(x => regionTeyvatFiltro === 'TODAS' || x.h.region === regionTeyvatFiltro);

            if (!hechosVisibles.length) return;
            if (btnElem) btnElem.disabled = true;

            // Traducir de 6 en 6 hechos en bloque para que sea mucho más rápido
            const LOTE = 6;
            for (let k = 0; k < hechosVisibles.length; k += LOTE) {
                const subLote = hechosVisibles.slice(k, k + LOTE).filter(x => !x.h.texto_es);
                if (!subLote.length) {
                    hechosVisibles.slice(k, k + LOTE).forEach(x => { x.h.mostrar_es = true; });
                    continue;
                }
                if (btnElem) {
                    btnElem.innerHTML = `<i class="fa-solid fa-spinner fa-spin text-[#3b4255]"></i> Traduciendo (${Math.min(k + LOTE, hechosVisibles.length)}/${hechosVisibles.length})...`;
                }
                try {
                    const unido = subLote.map((x, idx) => `[#${idx}#] ${x.h.texto}`).join('\n\n');
                    const tradUnida = await llamarApiTraducir(unido, `${era.titulo_es}`);
                    const partes = tradUnida.split(/\[#\d+#\]/).map(s => s.trim()).filter(Boolean);
                    subLote.forEach((x, idx) => {
                        x.h.texto_es = partes[idx] || x.h.texto;
                        x.h.mostrar_es = true;
                    });
                    renderizarDetalleEraTeyvat();
                } catch (e) {
                    console.error("Error traduciendo lote de era:", e);
                }
            }
            renderizarDetalleEraTeyvat();
        }

        function renderizarDetalleEraTeyvat() {
            const cont = document.getElementById('detalleEraTeyvatContent');
            if (!cont || !datosTimelineTeyvat.length) return;
            const era = datosTimelineTeyvat[eraTeyvatActivaIdx];
            if (!era) return;

            const filtroTxt = (document.getElementById('filtroTextoEraTeyvat')?.value || '').trim().toLowerCase();

            // Píldoras de regiones de esta época
            let pillsRegiones = `
                <button onclick="seleccionarRegionEraTeyvat('TODAS')" class="px-3 py-1 rounded-full text-xs font-semibold transition ${regionTeyvatFiltro === 'TODAS' ? 'bg-gi-gold text-[#161b26] font-bold shadow' : 'bg-[#181d2b] text-gi-cream/85 border border-gi-gold/30 hover:border-gi-gold'}">
                    ✦ Todas las Regiones (${era.hechos.length})
                </button>
            `;
            (era.regiones || []).forEach(reg => {
                const cnt = era.hechos.filter(h => h.region === reg).length;
                const act = (regionTeyvatFiltro === reg);
                pillsRegiones += `
                    <button onclick="seleccionarRegionEraTeyvat('${escaparComillas(reg)}')" class="px-3 py-1 rounded-full text-xs font-medium transition ${act ? 'bg-gi-gold text-[#161b26] font-bold shadow' : 'bg-[#181d2b] text-gi-cream/85 border border-gi-gold/30 hover:border-gi-gold'}">
                        ${escaparHtml(reg)} (${cnt})
                    </button>
                `;
            });

            // Galería de Entidades Protagonistas de esta época
            const htmlEntidadesEra = (era.entidades_destacadas && era.entidades_destacadas.length)
                ? era.entidades_destacadas.map(e => `
                    <div class="flex items-center gap-2 px-2.5 py-1.5 rounded-full bg-[#181d2b] hover:bg-[#242c40] border border-gi-gold/35 transition group">
                        <div class="flex items-center gap-2 cursor-pointer" onclick="mostrarFichaNodo('${escaparComillas(e.id)}')" title="Leer enciclopedia de ${escaparHtml(e.id)} a la derecha">
                            <img src="${urlAvatar(e.id)}" class="w-6 h-6 rounded-full object-cover border border-gi-gold shrink-0" alt="">
                            <span class="text-xs font-semibold text-gi-cream group-hover:text-gi-goldBright">${escaparHtml(e.id)}</span>
                            <span class="text-[10px] px-1.5 py-0.2 rounded-full bg-gi-gold/20 text-gi-goldBright">${e.menciones}</span>
                        </div>
                        <button onclick="abrirEntidadEnGrafoDesdeTimeline('${escaparComillas(e.id)}')" class="text-[10px] text-[#9fd6e3] hover:underline ml-1" title="Abrir en el Grafo de Constelación">
                            <i class="fa-solid fa-diagram-project"></i>
                        </button>
                    </div>
                `).join('')
                : '';

            // Hechos históricos filtrados
            const hechosFiltrados = era.hechos
                .map((h, idxOriginal) => ({ ...h, idxOriginal }))
                .filter(h => regionTeyvatFiltro === 'TODAS' || h.region === regionTeyvatFiltro)
                .filter(h => {
                    if (!filtroTxt) return true;
                    const t = ((h.mostrar_es && h.texto_es) ? h.texto_es : h.texto).toLowerCase();
                    const ents = (h.entidades || []).join(' ').toLowerCase();
                    return t.includes(filtroTxt) || ents.includes(filtroTxt) || h.region.toLowerCase().includes(filtroTxt);
                });

            const htmlHechos = hechosFiltrados.map(h => {
                const enEs = Boolean(h.mostrar_es && h.texto_es);
                const txtMostrar = enEs ? h.texto_es : h.texto;
                const etiqBtn = h.texto_es ? (enEs ? '🌐 Ver EN' : '🌐 Ver ES') : '🌐 Traducir';
                const margenSub = h.nivel === 2 ? 'ml-5' : (h.nivel >= 3 ? 'ml-10' : '');

                const pillsEnts = (h.entidades && h.entidades.length)
                    ? `<div class="flex flex-wrap items-center gap-1.5 pt-2">
                        ${h.entidades.map(ent => `
                            <button onclick="mostrarFichaNodo('${escaparComillas(ent)}')" class="px-2 py-0.5 rounded-full bg-[#3b4255]/15 hover:bg-[#3b4255]/30 border border-[#8c6d32]/40 text-[10px] font-semibold text-[#2c3345] flex items-center gap-1 transition">
                                <span class="text-[#8c6d32]">✦</span> ${escaparHtml(ent)}
                            </button>
                        `).join('')}
                       </div>`
                    : '';

                return `
                    <div class="relative pl-6 pb-3 ${margenSub}">
                        <div class="absolute -left-[9px] top-2 w-4 h-4 rounded-full bg-[#161b26] border-2 border-gi-gold flex items-center justify-center text-[8px] text-gi-goldBright shadow-[0_0_8px_rgba(240,220,175,0.6)]">✦</div>
                        <div class="gi-parchment-box rounded-xl overflow-hidden">
                            <div class="px-3.5 py-2 bg-[#e2d8c3] border-b border-[#c8b082] flex flex-wrap items-center justify-between gap-2">
                                <div class="flex items-center gap-2">
                                    <span class="px-2.5 py-0.5 rounded-full bg-[#3b4255] text-[#f0dcaf] text-[10px] font-bold">
                                        📍 ${escaparHtml(h.region)}
                                    </span>
                                    ${h.ref_url ? `<a href="${h.ref_url}" target="_blank" rel="noopener noreferrer" class="text-[11px] font-semibold text-[#5a4622] hover:underline flex items-center gap-1"><i class="fa-solid fa-book-bookmark text-[10px]"></i> ${escaparHtml(h.ref_titulo)} ↗</a>` : ''}
                                </div>
                                <div class="flex items-center gap-1.5">
                                    <button onclick="traducirHechoTeyvat(${eraTeyvatActivaIdx}, ${h.idxOriginal}, this)" class="px-2.5 py-0.5 rounded-full bg-[#3b4255] hover:bg-[#252b3b] border border-[#d3bc8e] text-[10px] font-semibold text-[#f4efe6] transition">
                                        ${etiqBtn}
                                    </button>
                                    <button onclick="copiarTextoAlPortapapeles(datosTimelineTeyvat[${eraTeyvatActivaIdx}].hechos[${h.idxOriginal}].mostrar_es ? datosTimelineTeyvat[${eraTeyvatActivaIdx}].hechos[${h.idxOriginal}].texto_es : datosTimelineTeyvat[${eraTeyvatActivaIdx}].hechos[${h.idxOriginal}].texto, this)" class="px-2.5 py-0.5 rounded-full bg-[#f5f0e6] hover:bg-white border border-[#b59d6e] text-[10px] font-semibold text-[#3b4255] transition">
                                        <i class="fa-regular fa-copy"></i> Copiar
                                    </button>
                                </div>
                            </div>
                            <div class="p-3.5 text-xs md:text-[13.5px] leading-relaxed font-medium text-[#2c3345]">
                                <div>${escaparHtml(txtMostrar)}</div>
                                ${pillsEnts}
                            </div>
                        </div>
                    </div>
                `;
            }).join('');

            cont.innerHTML = `
                <!-- Cabecera de la Época de Teyvat -->
                <div class="gi-card-inner p-5 rounded-2xl space-y-3 border border-gi-gold/50">
                    <div class="flex flex-wrap items-start justify-between gap-3">
                        <div>
                            <div class="text-[11px] font-serif uppercase tracking-widest text-gi-gold">CRÓNICA HISTÓRICA DE TEYVAT · ÉPOCA ${era.orden} DE ${datosTimelineTeyvat.length}</div>
                            <h2 class="font-serif text-xl md:text-2xl font-bold text-gi-goldBright mt-0.5">${escaparHtml(era.titulo_es)}</h2>
                            <p class="text-xs text-gi-cream/80 mt-1">${escaparHtml(era.subtitulo)}</p>
                        </div>
                        <div class="flex items-center gap-2 shrink-0">
                            <button onclick="traducirTodaEraTeyvat(${eraTeyvatActivaIdx}, this)" class="gi-btn-cream px-3.5 py-1.5 text-xs flex items-center gap-1.5">
                                <i class="fa-solid fa-language text-[#3b4255]"></i>
                                <span>Traducir Época con Gemini</span>
                            </button>
                            <a href="${era.url_wiki}" target="_blank" rel="noopener noreferrer" class="gi-btn-dark px-3 py-1.5 text-xs flex items-center gap-1.5">
                                <span>Wiki Timeline</span>
                                <i class="fa-solid fa-arrow-up-right-from-square text-[9px] text-gi-gold"></i>
                            </a>
                        </div>
                    </div>

                    <!-- Entidades Protagonistas de esta Época -->
                    <div class="pt-2 border-t border-gi-gold/25 space-y-2">
                        <div class="text-[11px] font-serif font-bold uppercase tracking-wider text-gi-gold flex items-center gap-1.5">
                            <span>✦</span>
                            <span>Entidades Protagonistas de esta Época (clic para abrir su enciclopedia a la derecha):</span>
                        </div>
                        <div class="flex flex-wrap gap-1.5 max-h-36 overflow-y-auto pr-1">${htmlEntidadesEra}</div>
                    </div>
                </div>

                <!-- Filtros de Región dentro de esta Época -->
                <div class="gi-card-inner p-3.5 rounded-xl space-y-2.5">
                    <div class="flex flex-wrap items-center gap-1.5">${pillsRegiones}</div>
                    <div class="relative flex items-center">
                        <i class="fa-solid fa-magnifying-glass absolute left-3 text-gi-gold/70 text-xs"></i>
                        <input type="text" id="filtroTextoEraTeyvat" value="${escaparHtml(filtroTxt)}" oninput="renderizarDetalleEraTeyvat()" placeholder="Buscar personaje, suceso o lugar dentro de esta época..." class="w-full bg-[#131722] border border-gi-gold/30 rounded-full pl-8 pr-4 py-1.5 text-xs text-gi-cream placeholder-[#7c8599] focus:outline-none focus:border-gi-goldBright">
                    </div>
                </div>

                <!-- Línea vertical de Hechos Históricos de la Época -->
                <div class="ml-3 border-l-2 border-gi-gold/50 space-y-1 pt-1">
                    ${htmlHechos || '<div class="p-4 text-xs text-gi-cream/70 italic">No hay hechos históricos que coincidan con el filtro.</div>'}
                </div>
            `;
        }

        function alternarVistaEnciclopedia(nodoId) {
            renderizarSeccionesEnciclopedia(nodoId);
            renderizarLineaTemporalEntidad(nodoId);
        }

        function alternarVentanaChat(forzarEstado = null) {
            const win = document.getElementById('ventanaFlotanteChat');
            if (!win) return;
            chatVentanaAbierta = (forzarEstado !== null) ? forzarEstado : !chatVentanaAbierta;
            if (chatVentanaAbierta) {
                win.classList.remove('hidden');
                renderizarHistorialChat();
                setTimeout(() => {
                    document.getElementById('chatInputPregunta')?.focus();
                }, 60);
            } else {
                win.classList.add('hidden');
            }
        }

        function alternarTamanoVentanaChat() {
            const win = document.getElementById('ventanaFlotanteChat');
            if (!win) return;
            chatVentanaGrande = !chatVentanaGrande;
            if (chatVentanaGrande) {
                win.className = "pointer-events-auto w-[92vw] sm:w-[620px] h-[720px] max-h-[85vh] max-w-[94vw] rounded-2xl bg-[#161b28]/98 border-2 border-gi-gold/70 shadow-[0_12px_42px_rgba(0,0,0,0.85)] flex flex-col overflow-hidden mb-2.5 transition-all duration-200";
            } else {
                win.className = "pointer-events-auto w-[370px] sm:w-[430px] h-[530px] max-h-[82vh] max-w-[94vw] rounded-2xl bg-[#161b28]/98 border-2 border-gi-gold/70 shadow-[0_12px_42px_rgba(0,0,0,0.85)] flex flex-col overflow-hidden mb-2.5 transition-all duration-200";
            }
        }

        function limpiarHistorialChat() {
            historialChat = [];
            renderizarHistorialChat();
        }

        function abrirOraculoChat() {
            alternarVentanaChat(true);
        }

        function formatearInlineMarkdown(lineaEscapada) {
            return lineaEscapada
                // Citas de fuentes [F1], [R1], [F1, R2]
                .replace(/\[((?:[FR]\d+(?:\s*,\s*)?)+)\]/g, '<span class="px-1.5 py-0.2 rounded bg-[#3b4255] text-[#f0dcaf] text-[10px] font-bold mx-0.5">[$1]</span>')
                // Código inline `...`
                .replace(/`([^`]+)`/g, '<code class="px-1.5 py-0.5 rounded bg-[#e2d8c3] text-[#2c3345] font-mono text-[11px]">$1</code>')
                // Negrita **...**
                .replace(/\*\*([^*]+)\*\*/g, '<strong class="font-bold text-[#1a2030]">$1</strong>')
                // Cursiva *...*
                .replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, '$1<em class="italic">$2</em>');
        }

        function renderizarMarkdownLore(texto) {
            if (!texto) return '';
            const lineas = escaparHtml(texto).split(/\r?\n/);
            let html = '';
            let enListaUl = false;
            let enListaOl = false;

            const cerrarListas = () => {
                if (enListaUl) { html += '</ul>'; enListaUl = false; }
                if (enListaOl) { html += '</ol>'; enListaOl = false; }
            };

            for (let i = 0; i < lineas.length; i++) {
                const raw = lineas[i];
                const l = raw.trim();

                if (!l) {
                    cerrarListas();
                    continue;
                }

                // Encabezados ###, ##, #
                const mHead = l.match(/^(#{1,4})\s+(.+)$/);
                if (mHead) {
                    cerrarListas();
                    const nivel = mHead[1].length;
                    const contenido = formatearInlineMarkdown(mHead[2]);
                    const tam = nivel === 1 ? 'text-sm md:text-base' : (nivel === 2 ? 'text-[13.5px]' : 'text-xs md:text-[13px]');
                    html += `<h4 class="font-serif font-bold ${tam} text-[#6e521f] mt-2.5 mb-1 flex items-center gap-1.5"><span>✦</span><span>${contenido}</span></h4>`;
                    continue;
                }

                // Viñetas (- o * )
                const mUl = l.match(/^[-*•]\s+(.+)$/);
                if (mUl) {
                    if (enListaOl) { html += '</ol>'; enListaOl = false; }
                    if (!enListaUl) {
                        html += '<ul class="list-disc pl-5 space-y-1 my-1.5">';
                        enListaUl = true;
                    }
                    html += `<li>${formatearInlineMarkdown(mUl[1])}</li>`;
                    continue;
                }

                // Listas numeradas (1. )
                const mOl = l.match(/^\d+\.\s+(.+)$/);
                if (mOl) {
                    if (enListaUl) { html += '</ul>'; enListaUl = false; }
                    if (!enListaOl) {
                        html += '<ol class="list-decimal pl-5 space-y-1 my-1.5">';
                        enListaOl = true;
                    }
                    html += `<li>${formatearInlineMarkdown(mOl[1])}</li>`;
                    continue;
                }

                // Cita >
                if (l.startsWith('&gt;')) {
                    cerrarListas();
                    const citaTxt = formatearInlineMarkdown(l.replace(/^&gt;\s*/, ''));
                    html += `<blockquote class="border-l-2 border-[#8c6d32] pl-3 py-0.5 my-1.5 italic text-[#424a5e]">${citaTxt}</blockquote>`;
                    continue;
                }

                cerrarListas();
                html += `<p class="my-1.5 leading-relaxed">${formatearInlineMarkdown(l)}</p>`;
            }

            cerrarListas();
            return html;
        }

        function usarPreguntaSugerida(preguntaTexto) {
            alternarVentanaChat(true);
            const inp = document.getElementById('chatInputPregunta');
            if (inp) {
                inp.value = preguntaTexto;
                enviarPreguntaChat();
            }
        }

        async function enviarPreguntaChat() {
            const inp = document.getElementById('chatInputPregunta');
            const btn = document.getElementById('btnEnviarChat');
            if (!inp) return;
            const pregunta = inp.value.trim();
            if (!pregunta) return;

            inp.value = '';
            historialChat.push({ rol: 'user', texto: pregunta });
            renderizarHistorialChat(true);

            if (btn) btn.disabled = true;

            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        pregunta,
                        entidad_foco: nodoActualId || '',
                        historial: historialChat.slice(-6).map(m => ({ rol: m.rol, texto: m.texto }))
                    })
                });
                const data = await res.json();
                if (data.error) {
                    historialChat.push({ rol: 'model', texto: `⚠️ ${data.error}`, fuentes_wiki: [], relaciones_grafo: [] });
                } else {
                    historialChat.push({
                        rol: 'model',
                        texto: data.respuesta || '',
                        fuentes_wiki: data.fuentes_wiki || [],
                        relaciones_grafo: data.relaciones_grafo || [],
                        entidades_clave: data.entidades_clave || []
                    });
                }
            } catch (e) {
                historialChat.push({ rol: 'model', texto: `⚠️ Error de red: ${e.message}`, fuentes_wiki: [], relaciones_grafo: [] });
            } finally {
                if (btn) btn.disabled = false;
                renderizarHistorialChat(false);
            }
        }

        function renderizarHistorialChat(cargando = false) {
            const cont = document.getElementById('contenedorMensajesChat');
            if (!cont) return;

            if (!historialChat.length && !cargando) {
                cont.innerHTML = `
                    <div class="gi-card-inner p-3.5 rounded-xl space-y-2.5 text-xs text-gi-cream/85">
                        <div class="flex items-center gap-2 text-gi-goldBright font-serif font-bold">
                            <span>✦</span>
                            <span>Oráculo del Irminsul</span>
                        </div>
                        <p class="leading-relaxed text-gi-cream/80 text-[11.5px]">
                            Pregunta en español sobre cualquier personaje, época o misterio de Teyvat. Respuestas con formato Markdown, enlaces a la Wiki y relaciones del grafo.
                        </p>
                        <div class="space-y-1.5 pt-1">
                            <div class="text-[10px] uppercase tracking-wider text-gi-gold font-bold">Preguntas de ejemplo:</div>
                            <button onclick="usarPreguntaSugerida('¿Qué hizo Capitano durante el Cataclismo hace 500 años y qué relación tiene con Natlan?')" class="w-full text-left px-3 py-2 rounded-lg bg-[#181d2b] hover:bg-[#232b40] border border-gi-gold/30 text-[11px] text-gi-cream transition">
                                ✦ ¿Qué hizo Capitano durante el Cataclismo hace 500 años y qué relación tiene con Natlan?
                            </button>
                            <button onclick="usarPreguntaSugerida('¿Cuál es la relación entre Nahida, Greater Lord Rukkhadevata y el Irminsul?')" class="w-full text-left px-3 py-2 rounded-lg bg-[#181d2b] hover:bg-[#232b40] border border-gi-gold/30 text-[11px] text-gi-cream transition">
                                ✦ ¿Cuál es la relación entre Nahida, Greater Lord Rukkhadevata y el Irminsul?
                            </button>
                            <button onclick="usarPreguntaSugerida('¿Qué sabemos sobre el Mundo Antiguo, los Siete Soberanos y el Orden Primordial?')" class="w-full text-left px-3 py-2 rounded-lg bg-[#181d2b] hover:bg-[#232b40] border border-gi-gold/30 text-[11px] text-gi-cream transition">
                                ✦ ¿Qué sabemos sobre el Mundo Antiguo, los Siete Soberanos y el Orden Primordial?
                            </button>
                        </div>
                    </div>
                `;
                return;
            }

            let htmlMsgs = historialChat.map((m, idx) => {
                if (m.rol === 'user') {
                    return `
                        <div class="flex justify-end">
                            <div class="max-w-[86%] px-3.5 py-2 rounded-2xl rounded-tr-sm bg-[#2b344b] border border-gi-gold/50 text-xs text-gi-cream font-medium shadow">
                                ${escaparHtml(m.texto)}
                            </div>
                        </div>
                    `;
                }

                const htmlFuentes = (m.fuentes_wiki && m.fuentes_wiki.length)
                    ? `
                        <details class="rounded-lg bg-[#151a27] border border-gi-gold/35 overflow-hidden">
                            <summary class="px-3 py-1.5 cursor-pointer select-none list-none flex items-center justify-between text-[10px] font-serif font-bold uppercase tracking-wider text-gi-goldBright hover:bg-[#1d2436] transition">
                                <span><i class="fa-solid fa-book-bookmark mr-1"></i> Fuentes de la Wiki (${m.fuentes_wiki.length})</span>
                                <span>▾</span>
                            </summary>
                            <div class="p-2 border-t border-gi-gold/25 space-y-1.5 max-h-48 overflow-y-auto">
                                ${m.fuentes_wiki.map(f => `
                                    <div class="p-2 rounded bg-[#101420] border border-gi-gold/25 space-y-1">
                                        <div class="flex items-center justify-between gap-1.5 flex-wrap">
                                            <div class="flex items-center gap-1.5 min-w-0 cursor-pointer hover:underline" onclick="abrirSoloEntidadEnCamino('${escaparComillas(f.entidad_id)}', false)" title="Añadir ${escaparHtml(f.entidad_id)} al camino">
                                                <img src="${urlAvatar(f.entidad_id)}" class="w-5 h-5 rounded-full object-cover border border-gi-gold shrink-0" alt="">
                                                <span class="px-1.5 py-0.2 rounded bg-gi-gold/20 text-gi-goldBright text-[9px] font-bold">${escaparHtml(f.codigo)}</span>
                                                <span class="text-[11px] font-bold text-gi-cream truncate">${escaparHtml(f.entidad_id)}</span>
                                                <span class="text-[10px] text-[#9fd6e3] truncate">· ${escaparHtml(f.seccion)}</span>
                                            </div>
                                            <a href="${f.url_wiki}" target="_blank" rel="noopener noreferrer" class="px-2 py-0.5 rounded-full bg-[#222b3e] hover:bg-[#2d3850] text-[9px] text-[#9fd6e3] border border-[#9fd6e3]/40 flex items-center gap-1 shrink-0">
                                                <span>Wiki</span>
                                                <i class="fa-solid fa-arrow-up-right-from-square text-[8px]"></i>
                                            </a>
                                        </div>
                                        <div class="text-[10px] text-gi-cream/70 italic line-clamp-2">${escaparHtml(f.extracto)}</div>
                                    </div>
                                `).join('')}
                            </div>
                        </details>
                    `
                    : '';

                const htmlRels = (m.relaciones_grafo && m.relaciones_grafo.length)
                    ? `
                        <details class="rounded-lg bg-[#151a27] border border-[#9fd6e3]/35 overflow-hidden">
                            <summary class="px-3 py-1.5 cursor-pointer select-none list-none flex items-center justify-between text-[10px] font-serif font-bold uppercase tracking-wider text-[#9fd6e3] hover:bg-[#1d2436] transition">
                                <span><i class="fa-solid fa-diagram-project mr-1"></i> Relaciones del Grafo (${m.relaciones_grafo.length})</span>
                                <span>▾</span>
                            </summary>
                            <div class="p-2 border-t border-[#9fd6e3]/25 space-y-1.5 max-h-48 overflow-y-auto">
                                ${m.relaciones_grafo.map((r, rIdx) => `
                                    <div class="p-2 rounded bg-[#101420] border border-[#9fd6e3]/25 space-y-1">
                                        <div class="flex items-center justify-between gap-1.5 flex-wrap">
                                            <div class="flex items-center gap-1 flex-wrap text-[11px]">
                                                <span class="px-1.5 py-0.2 rounded bg-[#9fd6e3]/15 text-[#9fd6e3] text-[9px] font-bold">R${rIdx + 1}</span>
                                                <button onclick="abrirSoloEntidadEnCamino('${escaparComillas(r.origen)}', false)" class="font-bold text-gi-goldBright hover:underline">${escaparHtml(r.origen)}</button>
                                                <span class="text-gi-gold text-[9px]">✦➔</span>
                                                <button onclick="abrirSoloEntidadEnCamino('${escaparComillas(r.destino)}', false)" class="font-bold text-[#9fd6e3] hover:underline">${escaparHtml(r.destino)}</button>
                                                <span class="text-[9px] px-1.5 py-0.2 rounded-full bg-gi-gold/15 text-gi-cream border border-gi-gold/30">${escaparHtml(r.epoca)}</span>
                                            </div>
                                            ${r.ref_url ? `<a href="${r.ref_url}" target="_blank" rel="noopener noreferrer" class="text-[9px] text-[#9fd6e3] hover:underline flex items-center gap-1"><span>${escaparHtml(r.ref_titulo)}</span> ↗</a>` : ''}
                                        </div>
                                        <div class="text-[10px] text-gi-cream/80 italic">"${escaparHtml(r.parrafo)}"</div>
                                    </div>
                                `).join('')}
                            </div>
                        </details>
                    `
                    : '';

                return `
                    <div class="gi-card-inner p-3 rounded-2xl rounded-tl-sm space-y-2.5">
                        <div class="flex items-center justify-between border-b border-gi-gold/25 pb-1.5">
                            <div class="flex items-center gap-1.5 text-[11px] font-serif font-bold text-gi-goldBright">
                                <span>✦</span>
                                <span>Oráculo del Irminsul</span>
                            </div>
                            <button onclick="copiarTextoAlPortapapeles(historialChat[${idx}].texto, this)" class="px-2 py-0.5 rounded-full bg-[#242c40] hover:bg-[#313b54] border border-gi-gold/40 text-[9px] text-gi-cream flex items-center gap-1">
                                <i class="fa-regular fa-copy text-gi-gold"></i> Copiar
                            </button>
                        </div>
                        <div class="gi-parchment-box p-3 rounded-lg text-xs leading-relaxed font-medium text-[#2c3345]">
                            ${renderizarMarkdownLore(m.texto)}
                        </div>
                        ${htmlFuentes}
                        ${htmlRels}
                    </div>
                `;
            }).join('');

            if (cargando) {
                htmlMsgs += `
                    <div class="gi-card-inner p-3 rounded-xl flex items-center gap-2.5 text-xs text-gi-goldBright">
                        <i class="fa-solid fa-spinner fa-spin text-sm"></i>
                        <span>Consultando el Irminsul y redactando respuesta...</span>
                    </div>
                `;
            }

            cont.innerHTML = htmlMsgs;
            cont.scrollTop = cont.scrollHeight;
        }

        function obtenerTodosLosHitosNodo(nodoId) {
            const n = cacheNodos[nodoId];
            if (!n) return [];
            const hitos = [];
            const vistos = new Set();

            [...(n.padres || []), ...(n.hijos || [])].forEach(grupo => {
                const edgeId = idAristaUnica(grupo.origen, grupo.destino);
                const arista = cacheAristas[edgeId];
                const listaRels = (arista && arista.relaciones) ? arista.relaciones : (grupo.relaciones || []);
                listaRels.forEach((r, relIdx) => {
                    const clave = `${edgeId}#${relIdx}`;
                    if (vistos.has(clave)) return;
                    vistos.add(clave);
                    hitos.push({
                        edgeId,
                        relIdx,
                        vecino_id: grupo.vecino_id,
                        rol_vecino: grupo.rol_vecino,
                        refObj: r
                    });
                });
            });

            return hitos.sort((a, b) => {
                const ordA = a.refObj.era_orden || 6;
                const ordB = b.refObj.era_orden || 6;
                if (ordA !== ordB) return ordA - ordB;
                return (a.refObj.epoca || '').localeCompare(b.refObj.epoca || '');
            });
        }

        async function traducirHitoCronologia(edgeId, relIdx, nodoId, btnElem) {
            const arista = cacheAristas[edgeId];
            if (!arista || !arista.relaciones || !arista.relaciones[relIdx]) return;
            const r = arista.relaciones[relIdx];

            if (r.parrafo_es) {
                r.mostrar_es = !r.mostrar_es;
                renderizarLineaTemporalEntidad(nodoId);
                return;
            }

            if (btnElem) {
                btnElem.disabled = true;
                btnElem.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Traduciendo...';
            }

            try {
                r.parrafo_es = await llamarApiTraducir(r.parrafo || '', `${arista.origen} -> ${arista.destino} (${r.epoca})`);
                r.mostrar_es = true;
                renderizarLineaTemporalEntidad(nodoId);
            } catch (e) {
                alert("Error al traducir hito: " + e.message);
                if (btnElem) {
                    btnElem.disabled = false;
                    btnElem.innerHTML = '🌐 Traducir';
                }
            }
        }

        let erasAbiertasEntidad = new Set();

        async function traducirTodaCronologia(nodoId, btnElem) {
            const hitos = obtenerTodosLosHitosNodo(nodoId);
            if (!hitos.length) return;
            if (btnElem) btnElem.disabled = true;

            try {
                const pendientes = [];
                hitos.forEach(h => {
                    erasAbiertasEntidad.add(h.refObj.era_id || 'era_6_archivo');
                    if (h.refObj.parrafo_es) {
                        h.refObj.mostrar_es = true;
                    } else if (h.refObj.parrafo) {
                        pendientes.push(h);
                    }
                });

                // Traducir en lotes unidos por ===SEPARADOR_GI=== para gastar 1 sola petición por cada 8 hitos
                const TAM_LOTE = 8;
                for (let i = 0; i < pendientes.length; i += TAM_LOTE) {
                    const lote = pendientes.slice(i, i + TAM_LOTE);
                    if (btnElem) {
                        btnElem.innerHTML = `<i class="fa-solid fa-spinner fa-spin text-gi-ink"></i> Traduciendo lote (${Math.min(i + TAM_LOTE, pendientes.length)}/${pendientes.length})...`;
                    }
                    const textoUnido = lote.map(h => h.refObj.parrafo).join('\n===SEPARADOR_GI===\n');
                    const tradUnida = await llamarApiTraducir(textoUnido, `Cronología de ${nodoId}`);
                    const partes = tradUnida.split(/\s*===SEPARADOR_GI===\s*/);
                    lote.forEach((h, idx) => {
                        h.refObj.parrafo_es = (partes[idx] || h.refObj.parrafo).trim();
                        h.refObj.mostrar_es = true;
                    });
                }

                renderizarLineaTemporalEntidad(nodoId);
                if (btnElem) {
                    btnElem.disabled = false;
                    btnElem.innerHTML = '<span>✦</span> Cronología Traducida (ES)';
                }
            } catch (e) {
                alert("Error traduciendo cronología: " + e.message);
                if (btnElem) {
                    btnElem.disabled = false;
                    btnElem.innerHTML = '<span>✦</span> Traducir Cronología con Gemini';
                }
            }
        }

        function alternarEraEntidadAbierta(eraId, abierto) {
            if (abierto) erasAbiertasEntidad.add(eraId);
            else erasAbiertasEntidad.delete(eraId);
        }

        function renderizarLineaTemporalEntidad(nodoId) {
            const contenedor = document.getElementById('contenedorCronologiaEpocas');
            if (!contenedor) return;

            const todosHitos = obtenerTodosLosHitosNodo(nodoId);

            if (!todosHitos.length) {
                contenedor.innerHTML = `<div class="p-4 rounded-xl bg-[#181d2b] border border-gi-gold/25 text-xs text-gi-cream/70 italic">No hay registros cronológicos asociados a esta entidad.</div>`;
                return;
            }

            let htmlCronologia = '';
            CATALOGO_ERAS.forEach(era => {
                const hitosDeEra = todosHitos.filter(h => (h.refObj.era_id || 'era_6_archivo') === era.id);
                if (!hitosDeEra.length) return;

                const estaAbierta = erasAbiertasEntidad.has(era.id);

                const tarjetasEra = hitosDeEra.map(h => {
                    const r = h.refObj;
                    const enEs = Boolean(r.mostrar_es && r.parrafo_es);
                    const textoMostrar = enEs ? r.parrafo_es : (r.parrafo || '');
                    const etiqTrad = r.parrafo_es ? (enEs ? '🌐 Ver EN' : '🌐 Ver ES') : '🌐 Traducir';
                    const badgeDireccion = h.rol_vecino === 'padre'
                        ? `<span class="text-[10px] px-2 py-0.5 rounded-full bg-[#192838] text-[#9fd6e3] border border-[#9fd6e3]/40">Mencionado por ${escaparHtml(h.vecino_id)}</span>`
                        : `<span class="text-[10px] px-2 py-0.5 rounded-full bg-[#2c261d] text-gi-goldBright border border-gi-gold/40">Vínculo con ${escaparHtml(h.vecino_id)}</span>`;

                    return `
                        <div class="relative pl-6 pb-4 group">
                            <!-- Nodo estelar en la línea vertical -->
                            <div class="absolute -left-[9px] top-1.5 w-4 h-4 rounded-full bg-[#161b26] border-2 border-gi-gold flex items-center justify-center text-[8px] text-gi-goldBright shadow-[0_0_8px_rgba(240,220,175,0.6)]">✦</div>

                            <div class="gi-card-inner p-3.5 rounded-xl space-y-2.5">
                                <div class="flex flex-wrap items-center justify-between gap-2">
                                    <div class="flex items-center gap-2 flex-wrap">
                                        <span class="px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-gi-gold/20 text-gi-goldBright border border-gi-gold/45">
                                            ⏳ ${escaparHtml(r.epoca || 'Overview')}
                                        </span>
                                        <div class="flex items-center gap-1.5 cursor-pointer hover:underline" onclick="abrirSoloEntidadEnCamino('${escaparComillas(h.vecino_id)}', false)" title="Añadir solo ${escaparHtml(h.vecino_id)} al camino">
                                            <img src="${urlAvatar(h.vecino_id)}" class="w-5 h-5 rounded-full object-cover border border-gi-gold" alt="">
                                            ${badgeDireccion}
                                        </div>
                                    </div>
                                    <div class="flex items-center gap-1.5">
                                        <button onclick="erasAbiertasEntidad.add('${era.id}'); traducirHitoCronologia('${escaparComillas(h.edgeId)}', ${h.relIdx}, '${escaparComillas(nodoId)}', this)" class="px-2 py-0.5 rounded-full bg-[#3b4255] hover:bg-[#252b3b] border border-gi-gold/60 text-[10px] font-semibold text-gi-cream transition">
                                            ${etiqTrad}
                                        </button>
                                        ${r.ref_url ? `<a href="${r.ref_url}" target="_blank" rel="noopener noreferrer" class="text-[10px] text-[#9fd6e3] hover:underline font-medium">${escaparHtml(r.ref_titulo || 'Fuente')} ↗</a>` : ''}
                                    </div>
                                </div>
                                <div class="gi-parchment-box p-3 rounded-lg text-xs md:text-[13px] leading-relaxed font-medium text-[#2c3345]">
                                    ${escaparHtml(textoMostrar)}
                                </div>
                            </div>
                        </div>
                    `;
                }).join('');

                htmlCronologia += `
                    <details class="rounded-xl border border-gi-gold/40 bg-[#1a2030]/80 overflow-hidden" ${estaAbierta ? 'open' : ''} ontoggle="alternarEraEntidadAbierta('${era.id}', this.open)">
                        <summary class="px-3.5 py-2.5 cursor-pointer select-none list-none bg-gradient-to-r from-[#262e42] to-[#1a2030] hover:from-[#2e3850] flex items-center justify-between shadow transition">
                            <div class="flex items-center gap-2">
                                <span class="text-gi-goldBright text-sm">✦</span>
                                <h4 class="font-serif font-bold text-xs md:text-sm text-gi-goldBright tracking-wide">${escaparHtml(era.titulo)}</h4>
                            </div>
                            <div class="flex items-center gap-2">
                                <span class="text-[10px] px-2.5 py-0.5 rounded-full bg-gi-gold/20 text-gi-cream border border-gi-gold/40 font-semibold">
                                    ${hitosDeEra.length} ${hitosDeEra.length === 1 ? 'hito' : 'hitos'} ▾
                                </span>
                            </div>
                        </summary>
                        <div class="p-3 pt-4 ml-3 border-l-2 border-gi-gold/45 space-y-1">
                            ${tarjetasEra}
                        </div>
                    </details>
                `;
            });

            contenedor.innerHTML = htmlCronologia;
        }

        function seleccionarSeccionArticulo(nodoId, idx) {
            seccionActivaIdx = idx;
            const n = cacheNodos[nodoId];
            if (n && n.secciones && idx >= 0 && n.secciones[idx]) {
                n.secciones[idx].abierto = true;
            }
            renderizarSeccionesEnciclopedia(nodoId);
        }

        function alternarTodasLasSeccionesEntidad(nodoId, abrir) {
            const n = cacheNodos[nodoId];
            if (!n || !n.secciones) return;
            n.secciones.forEach(s => { s.abierto = abrir; });
            renderizarSeccionesEnciclopedia(nodoId);
        }

        async function llamarApiTraducir(texto, contexto = "") {
            const res = await fetch('/api/traducir', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ texto, contexto })
            });
            const data = await res.json();
            if (data.error) throw new Error(data.error);
            return data.traduccion || texto;
        }

        async function traducirSeccionIndividual(nodoId, secIdx, btnElem) {
            const n = cacheNodos[nodoId];
            if (!n || !n.secciones || !n.secciones[secIdx]) return;
            const sec = n.secciones[secIdx];
            sec.abierto = true;

            // Si ya estaba traducida, permite alternar entre Español e Inglés original
            if (sec.contenido_es) {
                sec.mostrar_es = !sec.mostrar_es;
                renderizarSeccionesEnciclopedia(nodoId);
                return;
            }

            const htmlOriginalBtn = btnElem ? btnElem.innerHTML : '';
            if (btnElem) {
                btnElem.disabled = true;
                btnElem.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Traduciendo...';
            }

            try {
                const trad = await llamarApiTraducir(
                    `${sec.titulo}\n\n${sec.contenido}`,
                    `${n.id} - ${sec.titulo}`
                );
                const lineas = trad.split('\n');
                if (lineas.length > 1 && lineas[0].trim().length < 90) {
                    sec.titulo_es = lineas[0].trim().replace(/^#+\s*/, '');
                    sec.contenido_es = lineas.slice(1).join('\n').trim();
                } else {
                    sec.titulo_es = sec.titulo;
                    sec.contenido_es = trad;
                }
                sec.mostrar_es = true;
                sec.abierto = true;
                renderizarSeccionesEnciclopedia(nodoId);
            } catch (e) {
                alert("Error al traducir con Gemini: " + e.message);
                if (btnElem) {
                    btnElem.disabled = false;
                    btnElem.innerHTML = htmlOriginalBtn;
                }
            }
        }

        async function traducirTodoElArticulo(nodoId, btnElem) {
            const n = cacheNodos[nodoId];
            if (!n || !n.secciones) return;

            if (btnElem) btnElem.disabled = true;

            try {
                // Traducir primero los atributos de la ficha si aún no están traducidos
                if (n.atributos && n.atributos.length && !n.atributos_traducidos) {
                    if (btnElem) btnElem.innerHTML = '<i class="fa-solid fa-spinner fa-spin text-gi-ink"></i> Traduciendo ficha...';
                    const bloqueAttr = n.atributos.map(a => `${a.etiqueta}: ${a.valor}`).join('\n');
                    const tradAttr = await llamarApiTraducir(bloqueAttr, `${n.id} - Ficha técnica`);
                    const lineasAttr = tradAttr.split('\n').filter(l => l.trim());
                    lineasAttr.forEach((l, i) => {
                        if (n.atributos[i] && l.includes(':')) {
                            n.atributos[i].valor_es = l.split(':').slice(1).join(':').trim();
                        }
                    });
                    n.atributos_traducidos = true;
                    const contAttr = document.getElementById('contenedorAtributosFicha');
                    if (contAttr) {
                        contAttr.innerHTML = n.atributos.map(a => `
                            <div class="gi-stat-row flex items-start justify-between px-3 py-1.5 rounded-r-md text-xs gap-3">
                                <span class="text-[#b6afa0] font-medium flex items-center gap-1.5 shrink-0">
                                    <span class="text-gi-gold text-[10px]">✦</span> ${escaparHtml(a.etiqueta)}
                                </span>
                                <span class="text-gi-cream font-semibold text-right break-words">
                                    ${escaparHtml(a.valor_es || a.valor)}
                                </span>
                            </div>
                        `).join('');
                    }
                }

                // Agrupar secciones pendientes en lotes de hasta ~5500 caracteres para gastar muchas menos peticiones
                const indicesPendientes = [];
                for (let i = 0; i < n.secciones.length; i++) {
                    if (seccionActivaIdx !== -1 && seccionActivaIdx !== i) continue;
                    const sec = n.secciones[i];
                    if (sec.contenido_es) {
                        sec.mostrar_es = true;
                        sec.abierto = true;
                    } else {
                        indicesPendientes.push(i);
                    }
                }

                let loteActual = [];
                let charsActual = 0;
                const lotes = [];

                indicesPendientes.forEach(idx => {
                    const sec = n.secciones[idx];
                    const len = (sec.titulo.length + sec.contenido.length);
                    if (loteActual.length > 0 && (charsActual + len > 5500 || loteActual.length >= 5)) {
                        lotes.push(loteActual);
                        loteActual = [];
                        charsActual = 0;
                    }
                    loteActual.push(idx);
                    charsActual += len;
                });
                if (loteActual.length > 0) lotes.push(loteActual);

                for (let b = 0; b < lotes.length; b++) {
                    const grupoIdx = lotes[b];
                    if (btnElem) {
                        btnElem.innerHTML = `<i class="fa-solid fa-spinner fa-spin text-gi-ink"></i> Traduciendo lote (${b + 1}/${lotes.length})...`;
                    }
                    const payload = grupoIdx.map(idx => `${n.secciones[idx].titulo}\n\n${n.secciones[idx].contenido}`).join('\n===SEPARADOR_GI===\n');
                    const tradLote = await llamarApiTraducir(payload, `${n.id} - Lote de secciones`);
                    const partes = tradLote.split(/\s*===SEPARADOR_GI===\s*/);

                    grupoIdx.forEach((secIdx, pos) => {
                        const sec = n.secciones[secIdx];
                        const trad = (partes[pos] || '').trim();
                        if (trad) {
                            const lineas = trad.split('\n');
                            if (lineas.length > 1 && lineas[0].trim().length < 90) {
                                sec.titulo_es = lineas[0].trim().replace(/^#+\s*/, '');
                                sec.contenido_es = lineas.slice(1).join('\n').trim();
                            } else {
                                sec.titulo_es = sec.titulo;
                                sec.contenido_es = trad;
                            }
                            sec.mostrar_es = true;
                            sec.abierto = true;
                        }
                    });
                    renderizarSeccionesEnciclopedia(nodoId);
                }
                renderizarSeccionesEnciclopedia(nodoId);
                if (btnElem) {
                    btnElem.disabled = false;
                    btnElem.innerHTML = '<span>✦</span> Artículo Traducido (ES)';
                }
            } catch (e) {
                alert("Error traduciendo artículo: " + e.message);
                if (btnElem) {
                    btnElem.disabled = false;
                    btnElem.innerHTML = '<span>✦</span> Traducir Todo con Gemini';
                }
            }
        }

        async function traducirCitaRelacion(edgeId, relIdx, btnElem) {
            const arista = cacheAristas[edgeId];
            if (!arista || !arista.relaciones || !arista.relaciones[relIdx]) return;
            const r = arista.relaciones[relIdx];
            r.abierto = true;

            if (r.parrafo_es) {
                r.mostrar_es = !r.mostrar_es;
                mostrarFichaArista(edgeId);
                return;
            }

            if (btnElem) {
                btnElem.disabled = true;
                btnElem.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Traduciendo...';
            }

            try {
                r.parrafo_es = await llamarApiTraducir(r.parrafo || '', `${arista.origen} -> ${arista.destino} (${r.epoca})`);
                if (r.ref_texto && r.ref_texto !== r.parrafo && r.ref_texto !== 'Wiki Reference') {
                    r.ref_texto_es = await llamarApiTraducir(r.ref_texto, `Fuente: ${r.ref_titulo}`);
                }
                r.mostrar_es = true;
                r.abierto = true;
                mostrarFichaArista(edgeId);
            } catch (e) {
                alert("Error al traducir cita: " + e.message);
                if (btnElem) {
                    btnElem.disabled = false;
                    btnElem.innerHTML = '<i class="fa-solid fa-language"></i> Traducir';
                }
            }
        }

        async function traducirTodasLasCitasArista(edgeId, btnElem) {
            const arista = cacheAristas[edgeId];
            if (!arista || !arista.relaciones) return;
            if (btnElem) btnElem.disabled = true;

            for (let i = 0; i < arista.relaciones.length; i++) {
                const r = arista.relaciones[i];
                r.abierto = true;
                if (!r.parrafo_es) {
                    if (btnElem) {
                        btnElem.innerHTML = `<i class="fa-solid fa-spinner fa-spin text-gi-ink"></i> Traduciendo (${i + 1}/${arista.relaciones.length})...`;
                    }
                    await traducirCitaRelacion(edgeId, i, null);
                } else {
                    r.mostrar_es = true;
                }
            }
            mostrarFichaArista(edgeId);
        }

        function obtenerTextoCompletoParaCopiar(nodoId) {
            const n = cacheNodos[nodoId];
            if (!n) return '';
            if (n.secciones && n.secciones.length) {
                return n.secciones.map(s => {
                    const t = (s.mostrar_es && s.titulo_es) ? s.titulo_es : s.titulo;
                    const c = (s.mostrar_es && s.contenido_es) ? s.contenido_es : s.contenido;
                    return `== ${t} ==\n${c}`;
                }).join('\n\n');
            }
            return n.texto || '';
        }

        function actualizarEstadoAbiertoSeccion(nodoId, secIdx, abierto) {
            const n = cacheNodos[nodoId];
            if (n && n.secciones && n.secciones[secIdx]) {
                n.secciones[secIdx].abierto = abierto;
            }
        }

        function renderizarSeccionesEnciclopedia(nodoId) {
            const n = cacheNodos[nodoId];
            const contenedor = document.getElementById('contenedorCapitulosLore');
            const contenedorTabs = document.getElementById('contenedorIndiceSecciones');
            if (!n || !contenedor) return;

            const filtroTexto = (document.getElementById('filtroTextoArticulo')?.value || '').trim().toLowerCase();
            const secciones = n.secciones && n.secciones.length
                ? n.secciones
                : [{ titulo: 'Contenido General', contenido: n.texto || 'Sin contenido registrado.', abierto: false }];

            if (contenedorTabs) {
                let tabsHtml = `
                    <button onclick="seleccionarSeccionArticulo('${escaparComillas(n.id)}', -1)" class="px-2.5 py-1 rounded-full text-[11px] font-semibold transition ${seccionActivaIdx === -1 ? 'bg-gi-gold text-[#1b202e] font-bold shadow' : 'bg-[#181d2b] text-gi-cream/80 border border-gi-gold/30 hover:border-gi-gold'}">
                        ✦ Todas (${secciones.length})
                    </button>
                `;
                secciones.forEach((sec, idx) => {
                    const activo = (seccionActivaIdx === idx);
                    const tituloTab = (sec.mostrar_es && sec.titulo_es) ? sec.titulo_es : sec.titulo;
                    tabsHtml += `
                        <button onclick="seleccionarSeccionArticulo('${escaparComillas(n.id)}', ${idx})" class="px-2.5 py-1 rounded-full text-[11px] font-medium transition ${activo ? 'bg-gi-gold text-[#1b202e] font-bold shadow' : 'bg-[#181d2b] text-gi-cream/80 border border-gi-gold/30 hover:border-gi-gold'}">
                            ${idx + 1}. ${escaparHtml(tituloTab)}
                        </button>
                    `;
                });
                contenedorTabs.innerHTML = tabsHtml;
            }

            const seccionesAMostrar = secciones
                .map((s, idx) => ({ ...s, idx }))
                .filter(s => (seccionActivaIdx === -1 || s.idx === seccionActivaIdx))
                .filter(s => {
                    if (!filtroTexto) return true;
                    const t = ((s.mostrar_es && s.titulo_es) ? s.titulo_es : s.titulo).toLowerCase();
                    const c = ((s.mostrar_es && s.contenido_es) ? s.contenido_es : s.contenido).toLowerCase();
                    return t.includes(filtroTexto) || c.includes(filtroTexto);
                });

            if (!seccionesAMostrar.length) {
                contenedor.innerHTML = `<div class="p-4 rounded-xl bg-[#181d2b] border border-gi-gold/25 text-xs text-gi-cream/70 italic">No se encontraron pasajes que coincidan con "${escaparHtml(filtroTexto)}".</div>`;
                return;
            }

            // Cada sección de la entidad es un desplegable (<details>) cerrado por defecto salvo que el usuario lo abra o filtre texto
            contenedor.innerHTML = seccionesAMostrar.map(s => {
                const enEspanol = Boolean(s.mostrar_es && s.contenido_es);
                const tituloMostrar = enEspanol ? (s.titulo_es || s.titulo) : s.titulo;
                const contenidoMostrar = enEspanol ? s.contenido_es : s.contenido;
                const etiqBtnTrad = s.contenido_es
                    ? (enEspanol ? '🌐 Ver Original (EN)' : '🌐 Ver Español (ES)')
                    : 'Traducir';
                const estaAbierto = Boolean(s.abierto || filtroTexto);

                return `
                <details class="gi-parchment-box rounded-xl overflow-hidden group" ${estaAbierto ? 'open' : ''} ontoggle="actualizarEstadoAbiertoSeccion('${escaparComillas(n.id)}', ${s.idx}, this.open)">
                    <summary class="px-4 py-2.5 bg-[#e2d8c3] hover:bg-[#d8ccb4] cursor-pointer select-none list-none flex flex-wrap items-center justify-between gap-2 transition">
                        <h3 class="font-serif font-bold text-xs md:text-sm text-[#2c3345] flex items-center gap-2">
                            <span class="text-[#8c6d32]">✦</span>
                            <span>${s.idx + 1}. ${escaparHtml(tituloMostrar)}</span>
                            ${enEspanol ? '<span class="text-[10px] font-sans px-2 py-0.5 rounded-full bg-[#3b4255] text-[#f0dcaf]">ES · Gemini</span>' : ''}
                        </h3>
                        <div class="flex items-center gap-1.5 shrink-0" onclick="event.stopPropagation()">
                            <button onclick="traducirSeccionIndividual('${escaparComillas(n.id)}', ${s.idx}, this)" class="px-2.5 py-0.5 rounded-full bg-[#3b4255] hover:bg-[#252b3b] border border-[#d3bc8e] text-[10px] font-semibold text-[#f4efe6] transition flex items-center gap-1">
                                <i class="fa-solid fa-language text-[#f0dcaf]"></i> ${etiqBtnTrad}
                            </button>
                            <button onclick="copiarTextoAlPortapapeles(cacheNodos['${escaparComillas(n.id)}'].secciones[${s.idx}].mostrar_es ? cacheNodos['${escaparComillas(n.id)}'].secciones[${s.idx}].contenido_es : cacheNodos['${escaparComillas(n.id)}'].secciones[${s.idx}].contenido, this)" class="px-2.5 py-0.5 rounded-full bg-[#f5f0e6] hover:bg-white border border-[#b59d6e] text-[10px] font-semibold text-[#3b4255] transition flex items-center gap-1">
                                <i class="fa-regular fa-copy"></i> Copiar
                            </button>
                            <span class="text-[10px] px-2 py-0.5 rounded-full bg-[#cfc2a8] text-[#2c3345] font-bold pointer-events-none">▾</span>
                        </div>
                    </summary>
                    <div class="p-4 border-t border-[#c8b082] text-xs md:text-[13.5px] leading-relaxed whitespace-pre-line font-medium text-[#2c3345]">${escaparHtml(contenidoMostrar)}</div>
                </details>
            `;
            }).join('');
        }

        async function mostrarFichaNodo(id) {
            let n = cacheNodos[id];
            const inspector = document.getElementById('inspectorContent');
            inspector.scrollTop = 0;
            seccionActivaIdx = -1;
            erasAbiertasEntidad = new Set();

            if (!n || !n.cargadoCompleto) {
                document.getElementById('loadingOverlay').classList.remove('hidden');
                try {
                    const res = await fetch(`/api/nodo/${encodeURIComponent(id)}`);
                    const data = await res.json();
                    if (!data.error) {
                        guardarDatosNodoEnCache(data);
                        n = cacheNodos[data.id] || cacheNodos[id];
                    }
                } catch (e) {
                    console.error("Error cargando detalle del nodo:", e);
                } finally {
                    document.getElementById('loadingOverlay').classList.add('hidden');
                }
            }

            if (!n) return;

            // Al abrir la entidad, todas sus secciones empiezan cerradas de base
            if (n.secciones && Array.isArray(n.secciones)) {
                n.secciones.forEach(s => { s.abierto = false; });
            }

            document.getElementById("inspectorTypeBadge").textContent = "ENCICLOPEDIA DE ENTIDAD";
            document.getElementById("inspectorIdLabel").textContent = n.id;
            if (n.url) document.getElementById("inspectorWikiLink").href = n.url;

            const numAttrs = (n.atributos && n.atributos.length) ? n.atributos.length : 0;
            const htmlAtributos = numAttrs
                ? `<div id="contenedorAtributosFicha" class="space-y-1.5">
                    ${n.atributos.map(a => `
                        <div class="gi-stat-row flex items-start justify-between px-3 py-1.5 rounded-r-md text-xs gap-3">
                            <span class="text-[#b6afa0] font-medium flex items-center gap-1.5 shrink-0">
                                <span class="text-gi-gold text-[10px]">✦</span> ${escaparHtml(a.etiqueta)}
                            </span>
                            <span class="text-gi-cream font-semibold text-right break-words">
                                ${escaparHtml(a.valor_es || a.valor)}
                            </span>
                        </div>
                    `).join('')}
                   </div>`
                : '<span class="text-[#7c8599] text-xs italic">Sin atributos de ficha técnica</span>';

            const htmlPadres = (n.padres && n.padres.length)
                ? n.padres.map(p => {
                    const edgeId = idAristaUnica(p.origen, p.destino);
                    const nCitas = p.relaciones.length;
                    return `
                    <div class="flex items-center justify-between p-1.5 rounded-lg bg-[#181d2b]/90 border border-gi-gold/25 hover:border-gi-gold transition group gap-1.5">
                        <div class="flex items-center gap-2 min-w-0 flex-1 cursor-pointer" onclick="abrirSoloEntidadEnCamino('${escaparComillas(p.vecino_id)}', false)" title="Añadir solo ${escaparHtml(p.vecino_id)} a tu camino">
                            <img src="${urlAvatar(p.vecino_id)}" class="w-7 h-7 rounded-full object-cover border border-[#8ed1e6] shrink-0" alt="">
                            <span class="text-xs text-gi-cream font-medium truncate group-hover:text-gi-goldBright">${escaparHtml(p.vecino_id)}</span>
                        </div>
                        <button onclick="mostrarFichaArista('${escaparComillas(edgeId)}')" class="px-2 py-0.5 rounded-full bg-gi-gold/15 hover:bg-gi-gold/30 text-[10px] text-gi-goldBright border border-gi-gold/40 shrink-0 transition">
                            ${nCitas} ${nCitas === 1 ? 'cita' : 'citas'}
                        </button>
                    </div>`;
                }).join('')
                : '<span class="text-[#7c8599] text-xs italic">Sin menciones entrantes</span>';

            const htmlHijos = (n.hijos && n.hijos.length)
                ? n.hijos.map(h => {
                    const edgeId = idAristaUnica(h.origen, h.destino);
                    const nCitas = h.relaciones.length;
                    return `
                    <div class="flex items-center justify-between p-1.5 rounded-lg bg-[#181d2b]/90 border border-gi-gold/25 hover:border-gi-gold transition group gap-1.5">
                        <div class="flex items-center gap-2 min-w-0 flex-1 cursor-pointer" onclick="abrirSoloEntidadEnCamino('${escaparComillas(h.vecino_id)}', false)" title="Añadir solo ${escaparHtml(h.vecino_id)} a tu camino">
                            <img src="${urlAvatar(h.vecino_id)}" class="w-7 h-7 rounded-full object-cover border border-gi-gold shrink-0" alt="">
                            <span class="text-xs text-gi-cream font-medium truncate group-hover:text-gi-goldBright">${escaparHtml(h.vecino_id)}</span>
                        </div>
                        <button onclick="mostrarFichaArista('${escaparComillas(edgeId)}')" class="px-2 py-0.5 rounded-full bg-gi-gold/15 hover:bg-gi-gold/30 text-[10px] text-gi-goldBright border border-gi-gold/40 shrink-0 transition">
                            ${nCitas} ${nCitas === 1 ? 'cita' : 'citas'}
                        </button>
                    </div>`;
                }).join('')
                : '<span class="text-[#7c8599] text-xs italic">Sin menciones salientes</span>';

            const numSecs = (n.secciones && n.secciones.length) ? n.secciones.length : 1;
            const totalHitos = obtenerTodosLosHitosNodo(n.id).length;

            inspector.innerHTML = `
                <!-- Cabecera Compacta de la Entidad -->
                <div class="gi-card-inner p-4 rounded-xl space-y-3">
                    <div class="flex items-center gap-4">
                        <div class="relative shrink-0">
                            <img src="${urlAvatar(n.id)}" class="w-16 h-16 rounded-full object-cover border-2 border-gi-gold shadow-[0_0_16px_rgba(211,188,142,0.4)]" alt="${escaparHtml(n.id)}">
                            <span class="absolute -bottom-1 -right-1 w-5 h-5 rounded-full bg-[#1b202e] border border-gi-gold flex items-center justify-center text-[10px] text-gi-goldBright">✦</span>
                        </div>
                        <div class="min-w-0 flex-1">
                            <div class="flex items-start justify-between gap-2">
                                <h2 class="font-serif text-lg md:text-xl font-bold text-gi-cream leading-snug break-words">${escaparHtml(n.id)}</h2>
                                <button onclick="cerrarNodoDelGrafo('${escaparComillas(n.id)}')" class="px-2.5 py-0.5 rounded-full bg-[#351f26] hover:bg-[#4f2833] border border-rose-400/40 text-[10px] font-semibold text-rose-200 shrink-0 transition flex items-center gap-1" title="Cerrar esta entidad del grafo (también con clic derecho sobre el nodo)">
                                    <i class="fa-solid fa-xmark"></i> Cerrar del Grafo
                                </button>
                            </div>
                            <div class="flex flex-wrap items-center gap-2 mt-1.5">
                                <span class="text-[11px] px-2.5 py-0.5 rounded-full bg-gi-gold/20 text-gi-goldBright border border-gi-gold/40 font-semibold">${numSecs} secciones web</span>
                                <span class="text-[11px] px-2.5 py-0.5 rounded-full bg-[#243242] text-[#9fd6e3] border border-[#9fd6e3]/35 font-medium">${n.padres.length} entrantes</span>
                                <span class="text-[11px] px-2.5 py-0.5 rounded-full bg-[#2e2a22] text-gi-goldBright border border-gi-gold/40 font-medium">${n.hijos.length} salientes</span>
                            </div>
                        </div>
                    </div>

                    <div class="grid grid-cols-2 gap-2 pt-1">
                        <button onclick="abrirSoloEntidadEnCamino('${escaparComillas(n.id)}', true)" class="gi-btn-cream py-2 px-2.5 text-[11px] flex items-center justify-center gap-1.5" title="Dejar solo esta entidad en el lienzo para iniciar tu propio camino paso a paso">
                            <span>✦</span> Iniciar Camino
                        </button>
                        <button onclick="expandirEntidad('${escaparComillas(n.id)}', false)" class="gi-btn-dark py-2 px-2.5 text-[11px] flex items-center justify-center gap-1.5" title="Desplegar todas las entidades vecinas de golpe">
                            <i class="fa-solid fa-plus text-gi-gold text-[10px]"></i> Expandir Todas
                        </button>
                    </div>
                </div>

                <!-- BLOQUE 1: Todas las Secciones de la Entidad como Desplegables (cerrados de base) -->
                <div id="bloqueVistaCapitulos" class="space-y-3">

                    <!-- 1. DESPLEGABLE: Ficha Técnica y Atributos de la Entidad (cerrado de base) -->
                    <details class="gi-card-inner rounded-xl overflow-hidden group">
                        <summary class="px-4 py-3 cursor-pointer select-none flex items-center justify-between gap-2 bg-gradient-to-r from-[#242c40] to-[#181d2b] hover:from-[#2c364f] transition list-none">
                            <div class="flex items-center gap-2.5">
                                <span class="w-6 h-6 rounded-full bg-gi-gold/20 border border-gi-gold/50 flex items-center justify-center text-gi-goldBright text-xs">📋</span>
                                <div>
                                    <div class="font-serif font-bold text-xs md:text-sm text-gi-goldBright tracking-wide">
                                        Ficha Técnica y Atributos
                                    </div>
                                    <div class="text-[10px] text-[#9aa3b8]">
                                        ${numAttrs} ${numAttrs === 1 ? 'atributo registrado en la Wiki' : 'atributos registrados en la Wiki'}
                                    </div>
                                </div>
                            </div>
                            <span class="text-[10px] px-2.5 py-0.5 rounded-full bg-gi-gold/20 text-gi-goldBright border border-gi-gold/40 font-semibold">
                                Desplegar ▾
                            </span>
                        </summary>
                        <div class="p-3.5 border-t border-gi-gold/25 bg-[#151925]/90">
                            ${htmlAtributos}
                        </div>
                    </details>

                    <!-- 2. DESPLEGABLE: Historia / Información de la Web y Secciones de Lore (cerrado de base) -->
                    <details class="gi-card-inner rounded-xl overflow-hidden group">
                        <summary class="px-4 py-3 cursor-pointer select-none flex items-center justify-between gap-2 bg-gradient-to-r from-[#242c40] to-[#181d2b] hover:from-[#2c364f] transition list-none">
                            <div class="flex items-center gap-2.5">
                                <span class="w-6 h-6 rounded-full bg-gi-gold/20 border border-gi-gold/50 flex items-center justify-center text-gi-goldBright text-xs">📖</span>
                                <div>
                                    <div class="font-serif font-bold text-xs md:text-sm text-gi-goldBright tracking-wide">
                                        Historia y Capítulos de la Web (${numSecs})
                                    </div>
                                    <div class="text-[10px] text-[#9aa3b8]">
                                        Todas las secciones de la página de la Wiki desplegables
                                    </div>
                                </div>
                            </div>
                            <span class="text-[10px] px-2.5 py-0.5 rounded-full bg-gi-gold/20 text-gi-goldBright border border-gi-gold/40 font-semibold">
                                Desplegar ▾
                            </span>
                        </summary>
                        <div class="p-3.5 border-t border-gi-gold/25 space-y-3 bg-[#151925]/90">
                            <div class="flex flex-wrap items-center justify-between gap-2">
                                <div class="flex items-center gap-1.5 flex-wrap">
                                    <button onclick="alternarTodasLasSeccionesEntidad('${escaparComillas(n.id)}', true)" class="gi-btn-dark px-2.5 py-1 text-[10px] flex items-center gap-1">
                                        <span>▾</span> Abrir todas
                                    </button>
                                    <button onclick="alternarTodasLasSeccionesEntidad('${escaparComillas(n.id)}', false)" class="gi-btn-dark px-2.5 py-1 text-[10px] flex items-center gap-1">
                                        <span>▴</span> Cerrar todas
                                    </button>
                                </div>
                                <div class="flex items-center gap-1.5 flex-wrap">
                                    <button onclick="traducirTodoElArticulo('${escaparComillas(n.id)}', this)" class="gi-btn-cream px-3 py-1 text-[11px] flex items-center gap-1.5 shrink-0">
                                        <i class="fa-solid fa-language text-[#3b4255]"></i>
                                        <span>Traducir Todo con Gemini</span>
                                    </button>
                                    <button onclick="copiarTextoAlPortapapeles(obtenerTextoCompletoParaCopiar('${escaparComillas(n.id)}'), this)" class="gi-btn-dark px-3 py-1 text-[11px] flex items-center gap-1.5 shrink-0">
                                        <i class="fa-regular fa-copy text-gi-gold"></i> Copiar Todo
                                    </button>
                                </div>
                            </div>

                            <!-- Buscador interno dentro del artículo + Índice de Secciones -->
                            <div class="p-2.5 rounded-xl bg-[#181d2b] border border-gi-gold/25 space-y-2">
                                <div class="relative flex items-center">
                                    <i class="fa-solid fa-filter absolute left-3 text-gi-gold/70 text-[11px]"></i>
                                    <input type="text" id="filtroTextoArticulo" oninput="renderizarSeccionesEnciclopedia('${escaparComillas(n.id)}')" placeholder="Filtrar palabra o tema dentro de este artículo..." class="w-full bg-[#131722] border border-gi-gold/30 rounded-full pl-8 pr-3 py-1 text-xs text-gi-cream placeholder-[#7c8599] focus:outline-none focus:border-gi-gold">
                                </div>
                                <div id="contenedorIndiceSecciones" class="flex flex-wrap gap-1.5 max-h-28 overflow-y-auto pr-1"></div>
                            </div>

                            <!-- Capítulos íntegros en pergamino (cada uno desplegable y cerrado de base) -->
                            <div id="contenedorCapitulosLore" class="space-y-2.5"></div>
                        </div>
                    </details>

                    <!-- 3. DESPLEGABLE: Entidades Relacionadas / Vínculos del Irminsul (cerrado de base) -->
                    <details class="gi-card-inner rounded-xl overflow-hidden group">
                        <summary class="px-4 py-3 cursor-pointer select-none flex items-center justify-between gap-2 bg-gradient-to-r from-[#242c40] to-[#181d2b] hover:from-[#2c364f] transition list-none">
                            <div class="flex items-center gap-2.5">
                                <span class="w-6 h-6 rounded-full bg-gi-gold/20 border border-gi-gold/50 flex items-center justify-center text-gi-goldBright text-xs">✦</span>
                                <div>
                                    <div class="font-serif font-bold text-xs md:text-sm text-gi-goldBright tracking-wide">
                                        Entidades Relacionadas (Vínculos)
                                    </div>
                                    <div class="text-[10px] text-[#9aa3b8]">
                                        ${n.padres.length} menciones entrantes · ${n.hijos.length} menciones salientes
                                    </div>
                                </div>
                            </div>
                            <span class="text-[10px] px-2.5 py-0.5 rounded-full bg-gi-gold/20 text-gi-goldBright border border-gi-gold/40 font-semibold">
                                Desplegar ▾
                            </span>
                        </summary>
                        <div class="p-3.5 border-t border-gi-gold/25 bg-[#151925]/90 space-y-2.5">
                            <details class="rounded-xl border border-[#9fd6e3]/35 bg-[#181d2b]/90 overflow-hidden">
                                <summary class="px-3.5 py-2 cursor-pointer select-none list-none flex items-center justify-between text-[11px] font-bold uppercase tracking-wider text-[#9fd6e3] hover:bg-[#1f2638] transition">
                                    <span>✦ Menciones Entrantes (${n.padres.length})</span>
                                    <span class="text-[10px] px-2 py-0.5 rounded-full bg-[#9fd6e3]/15">▾</span>
                                </summary>
                                <div class="p-2.5 border-t border-[#9fd6e3]/20 space-y-1.5 max-h-60 overflow-y-auto">${htmlPadres}</div>
                            </details>

                            <details class="rounded-xl border border-gi-gold/35 bg-[#181d2b]/90 overflow-hidden">
                                <summary class="px-3.5 py-2 cursor-pointer select-none list-none flex items-center justify-between text-[11px] font-bold uppercase tracking-wider text-gi-goldBright hover:bg-[#1f2638] transition">
                                    <span>✦ Menciones Salientes (${n.hijos.length})</span>
                                    <span class="text-[10px] px-2 py-0.5 rounded-full bg-gi-gold/20">▾</span>
                                </summary>
                                <div class="p-2.5 border-t border-gi-gold/20 space-y-1.5 max-h-60 overflow-y-auto">${htmlHijos}</div>
                            </details>
                        </div>
                    </details>

                    <!-- 4. DESPLEGABLE: Línea Temporal de la Entidad (cerrado de base) -->
                    <details class="gi-card-inner rounded-xl overflow-hidden group">
                        <summary class="px-4 py-3 cursor-pointer select-none flex items-center justify-between gap-2 bg-gradient-to-r from-[#242c40] to-[#181d2b] hover:from-[#2c364f] transition list-none">
                            <div class="flex items-center gap-2.5">
                                <span class="w-6 h-6 rounded-full bg-gi-gold/20 border border-gi-gold/50 flex items-center justify-center text-gi-goldBright text-xs">⏳</span>
                                <div>
                                    <div class="font-serif font-bold text-xs md:text-sm text-gi-goldBright tracking-wide">
                                        Línea Temporal de la Entidad
                                    </div>
                                    <div class="text-[10px] text-[#9aa3b8]">
                                        ${totalHitos} ${totalHitos === 1 ? 'hito cronológico registrado' : 'hitos cronológicos ordenados por era'}
                                    </div>
                                </div>
                            </div>
                            <span class="text-[10px] px-2.5 py-0.5 rounded-full bg-gi-gold/20 text-gi-goldBright border border-gi-gold/40 font-semibold">
                                Desplegar ▾
                            </span>
                        </summary>
                        <div class="p-3.5 border-t border-gi-gold/25 space-y-3 bg-[#151925]/90">
                            <div class="flex items-center justify-between gap-2">
                                <span class="text-[11px] text-[#b6afa0] italic">Hitos y menciones de ${escaparHtml(n.id)} por épocas:</span>
                                <button onclick="traducirTodaCronologia('${escaparComillas(n.id)}', this)" class="gi-btn-cream px-3 py-1 text-[11px] flex items-center gap-1.5 shrink-0">
                                    <i class="fa-solid fa-language text-[#3b4255]"></i>
                                    <span>Traducir Cronología con Gemini</span>
                                </button>
                            </div>
                            <div id="contenedorCronologiaEpocas" class="space-y-3 pt-1 max-h-[520px] overflow-y-auto pr-1"></div>
                        </div>
                    </details>

                </div>
            `;

            alternarVistaEnciclopedia(n.id);
            inspector.scrollTop = 0;
        }

        function mostrarFichaArista(edgeId) {
            const arista = cacheAristas[edgeId];
            if (!arista || !arista.relaciones || !arista.relaciones.length) return;

            const inspector = document.getElementById('inspectorContent');
            inspector.scrollTop = 0;

            const total = arista.relaciones.length;
            const primeraRefUrl = arista.relaciones.find(r => r.ref_url)?.ref_url || '#';

            document.getElementById("inspectorTypeBadge").textContent = `VÍNCULO ASTRAL (${total})`;
            document.getElementById("inspectorIdLabel").textContent = `${arista.origen} ➔ ${arista.destino}`;
            document.getElementById("inspectorWikiLink").href = primeraRefUrl;

            const htmlTarjetasRelacion = arista.relaciones.map((r, idx) => {
                const enEs = Boolean(r.mostrar_es && r.parrafo_es);
                const parrafoMostrar = enEs ? r.parrafo_es : (r.parrafo || 'Sin párrafo guardado.');
                const refTextoMostrar = enEs ? (r.ref_texto_es || r.ref_texto) : r.ref_texto;
                const etiqBtn = r.parrafo_es ? (enEs ? '🌐 Ver EN' : '🌐 Ver ES') : '🌐 Traducir';
                const estaAbierto = Boolean(r.abierto);

                return `
                <details class="gi-card-inner rounded-xl overflow-hidden group" ${estaAbierto ? 'open' : ''} ontoggle="if(cacheAristas['${escaparComillas(edgeId)}']?.relaciones[${idx}]) cacheAristas['${escaparComillas(edgeId)}'].relaciones[${idx}].abierto = this.open">
                    <summary class="px-4 py-3 cursor-pointer select-none list-none flex flex-wrap items-center justify-between gap-2 bg-gradient-to-r from-[#242c40] to-[#181d2b] hover:from-[#2c364f] transition">
                        <span class="px-3 py-0.5 rounded-full text-[11px] font-bold bg-gi-gold/20 text-gi-goldBright border border-gi-gold/40">
                            ✦ Registro #${idx + 1} · ${escaparHtml(r.epoca || 'Overview')} ${enEs ? '(ES)' : ''}
                        </span>
                        <div class="flex items-center gap-2" onclick="event.stopPropagation()">
                            <button onclick="traducirCitaRelacion('${escaparComillas(edgeId)}', ${idx}, this)" class="px-2.5 py-0.5 rounded-full bg-[#3b4255] hover:bg-[#252b3b] border border-gi-gold text-[10px] font-semibold text-gi-cream transition flex items-center gap-1">
                                <i class="fa-solid fa-language text-gi-goldBright"></i> ${etiqBtn}
                            </button>
                            ${r.ref_url ? `<a href="${r.ref_url}" target="_blank" rel="noopener noreferrer" class="text-[11px] text-[#9fd6e3] hover:underline flex items-center gap-1 font-medium"><i class="fa-solid fa-book-bookmark text-[10px]"></i> ${escaparHtml(r.ref_titulo || 'Fuente Wiki')} ↗</a>` : ''}
                            <span class="text-[10px] px-2 py-0.5 rounded-full bg-gi-gold/20 text-gi-goldBright border border-gi-gold/40 font-semibold pointer-events-none">▾</span>
                        </div>
                    </summary>

                    <div class="p-4 border-t border-gi-gold/20 space-y-3 bg-[#151925]/90">
                        <div class="gi-parchment-box p-3.5 rounded-lg text-xs md:text-[13.5px] italic leading-relaxed font-medium">
                            "${escaparHtml(parrafoMostrar)}"
                        </div>

                        ${(refTextoMostrar && refTextoMostrar !== r.parrafo && refTextoMostrar !== 'Wiki Reference') ? `
                            <div class="p-3 rounded-lg bg-[#141824] border border-gi-gold/25 text-xs text-gi-cream/90 leading-relaxed whitespace-pre-line">
                                <div class="text-[10px] font-bold uppercase tracking-wider text-gi-gold mb-1">📜 Texto de la Fuente (${escaparHtml(r.ref_titulo || 'Referencia')})</div>
                                ${escaparHtml(refTextoMostrar)}
                            </div>
                        ` : ''}
                    </div>
                </details>
            `;
            }).join('');

            inspector.innerHTML = `
                <!-- Cabecera Origen -> Destino estilo Menú de Genshin -->
                <div class="gi-card-inner p-3.5 rounded-xl flex items-center justify-between gap-2">
                    <button onclick="expandirEntidad('${escaparComillas(arista.origen)}', false)" class="flex-1 p-2.5 rounded-lg bg-[#181d2b] hover:bg-[#232a3d] border border-gi-gold/30 text-left transition flex items-center gap-2.5 min-w-0">
                        <img src="${urlAvatar(arista.origen)}" class="w-10 h-10 rounded-full object-cover border border-gi-gold shrink-0" alt="">
                        <div class="min-w-0">
                            <div class="text-[10px] text-[#9aa3b8] uppercase">Origen</div>
                            <div class="font-serif font-bold text-sm text-gi-cream truncate">${escaparHtml(arista.origen)}</div>
                        </div>
                    </button>
                    <div class="flex flex-col items-center px-1.5 shrink-0">
                        <span class="text-[10px] text-gi-goldBright font-bold">${total} ${total === 1 ? 'cita' : 'citas'}</span>
                        <span class="text-gi-gold text-xs">✦➔</span>
                    </div>
                    <button onclick="expandirEntidad('${escaparComillas(arista.destino)}', false)" class="flex-1 p-2.5 rounded-lg bg-[#181d2b] hover:bg-[#232a3d] border border-gi-gold/30 text-left transition flex items-center gap-2.5 min-w-0">
                        <img src="${urlAvatar(arista.destino)}" class="w-10 h-10 rounded-full object-cover border border-gi-gold shrink-0" alt="">
                        <div class="min-w-0">
                            <div class="text-[10px] text-[#9aa3b8] uppercase">Destino</div>
                            <div class="font-serif font-bold text-sm text-gi-cream truncate">${escaparHtml(arista.destino)}</div>
                        </div>
                    </button>
                </div>

                <!-- Lista de todas las relaciones/citas de esta flecha -->
                <div class="space-y-3">
                    <div class="flex items-center justify-between gap-2">
                        <div class="flex items-center gap-2 text-xs font-serif font-bold uppercase tracking-wider text-gi-gold">
                            <span>✦</span>
                            <span>Pasajes del Archivo (${total})</span>
                        </div>
                        <button onclick="traducirTodasLasCitasArista('${escaparComillas(edgeId)}', this)" class="gi-btn-cream px-3 py-1 text-[11px] flex items-center gap-1.5 shrink-0">
                            <i class="fa-solid fa-language text-[#3b4255]"></i>
                            <span>Traducir Citas con Gemini</span>
                        </button>
                    </div>
                    ${htmlTarjetasRelacion}
                </div>
            `;
            inspector.scrollTop = 0;
        }

        function dibujarEstrellaCuatroPuntas(ctx, x, y, radioExterior, radioInterior, colorRelleno, colorBrillo = null) {
            ctx.save();
            ctx.beginPath();
            for (let i = 0; i < 8; i++) {
                const angulo = (i * Math.PI) / 4;
                const r = (i % 2 === 0) ? radioExterior : radioInterior;
                const px = x + Math.cos(angulo) * r;
                const py = y + Math.sin(angulo) * r;
                if (i === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
            }
            ctx.closePath();
            if (colorBrillo) {
                ctx.shadowColor = colorBrillo;
                ctx.shadowBlur = 10;
            }
            ctx.fillStyle = colorRelleno;
            ctx.fill();
            ctx.restore();
        }

        // Fondo astral del grafo: anillos de astrolabio centrados en la entidad activa y destellos en los hilos de constelación
        network.on("beforeDrawing", function (ctx) {
            const idsVisibles = nodes.getIds();
            if (!idsVisibles.length) return;
            const posiciones = network.getPositions(idsVisibles);

            // 1. Astrolabio estelar alrededor del nodo central activo
            const posCentral = nodoActualId ? posiciones[nodoActualId] : null;
            if (posCentral && !modoJerarquico) {
                ctx.save();
                const grad = ctx.createRadialGradient(posCentral.x, posCentral.y, 12, posCentral.x, posCentral.y, 260);
                grad.addColorStop(0, 'rgba(240, 220, 175, 0.20)');
                grad.addColorStop(0.35, 'rgba(159, 214, 227, 0.08)');
                grad.addColorStop(1, 'rgba(15, 20, 34, 0)');
                ctx.fillStyle = grad;
                ctx.beginPath();
                ctx.arc(posCentral.x, posCentral.y, 260, 0, Math.PI * 2);
                ctx.fill();

                // Anillo interior continuo de constelación
                ctx.strokeStyle = 'rgba(211, 188, 142, 0.22)';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.arc(posCentral.x, posCentral.y, 145, 0, Math.PI * 2);
                ctx.stroke();

                // Anillo exterior punteado de astrolabio
                ctx.setLineDash([5, 7]);
                ctx.strokeStyle = 'rgba(240, 220, 175, 0.18)';
                ctx.beginPath();
                ctx.arc(posCentral.x, posCentral.y, 225, 0, Math.PI * 2);
                ctx.stroke();
                ctx.setLineDash([]);

                // 4 pequeñas estrellas cardinales en el anillo del astrolabio
                const angulosCardinales = [0, Math.PI / 2, Math.PI, (Math.PI * 3) / 2];
                angulosCardinales.forEach(ang => {
                    const sx = posCentral.x + Math.cos(ang) * 145;
                    const sy = posCentral.y + Math.sin(ang) * 145;
                    dibujarEstrellaCuatroPuntas(ctx, sx, sy, 5.5, 1.8, 'rgba(240, 220, 175, 0.55)');
                });
                ctx.restore();
            }
        });

        // Engaste de estrella de Constelación (Stella Fortuna) alrededor de cada nodo
        network.on("afterDrawing", function (ctx) {
            const listaNodos = nodes.get();
            if (!listaNodos.length) return;
            const posiciones = network.getPositions(listaNodos.map(n => n.id));

            ctx.save();
            listaNodos.forEach(n => {
                if (n.hidden) return;
                const pos = posiciones[n.id];
                if (!pos) return;

                const esCentral = (n.rolConstelacion === 'central');
                const esExpandido = (n.rolConstelacion === 'expandido');
                const esPadre = (n.rolConstelacion === 'padre');

                const radioBase = n.size || (esCentral ? 46 : 27);
                const radioAnillo = radioBase + (esCentral ? 6.5 : 4.5);

                const colorLinea = esCentral
                    ? 'rgba(255, 243, 209, 0.85)'
                    : esPadre
                        ? 'rgba(159, 214, 227, 0.65)'
                        : 'rgba(229, 207, 158, 0.68)';

                const colorEstrella = esCentral
                    ? '#fff6dc'
                    : esPadre
                        ? '#c9f0fa'
                        : '#f5e4be';

                // Anillo fino exterior tipo engaste de Stella Fortuna
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, radioAnillo, 0, Math.PI * 2);
                ctx.strokeStyle = colorLinea;
                ctx.lineWidth = esCentral ? 1.5 : 1.1;
                ctx.stroke();

                // Puntas de estrella en los 4 puntos cardinales del nodo
                const puntas = [0, Math.PI / 2, Math.PI, (Math.PI * 3) / 2];
                const tamExt = esCentral ? 6.5 : (esExpandido ? 5 : 4);
                const tamInt = esCentral ? 2.2 : 1.4;
                puntas.forEach(ang => {
                    const px = pos.x + Math.cos(ang) * radioAnillo;
                    const py = pos.y + Math.sin(ang) * radioAnillo;
                    dibujarEstrellaCuatroPuntas(
                        ctx,
                        px,
                        py,
                        tamExt,
                        tamInt,
                        colorEstrella,
                        esCentral ? 'rgba(240, 220, 175, 0.9)' : null
                    );
                });
            });
            ctx.restore();
        });

        network.on("click", function (params) {
            if (params.nodes.length > 0) {
                mostrarFichaNodo(params.nodes[0]);
            } else if (params.edges.length > 0) {
                const edgeId = params.edges[0];
                mostrarFichaArista(edgeId);
                if (!mostrarEtiquetasSiempre) {
                    const actualizaciones = edges.get().map(e => ({
                        id: e.id,
                        label: e.id === edgeId ? etiquetaResumenArista(cacheAristas[e.id]) : ''
                    }));
                    edges.update(actualizaciones);
                }
            }
        });

        // Clic derecho sobre cualquier nodo del grafo para cerrarlo/quitarlo del lienzo
        network.on("oncontext", function (params) {
            if (params.event) params.event.preventDefault();
            const nodoBajoCursor = network.getNodeAt(params.pointer.DOM);
            if (nodoBajoCursor) {
                cerrarNodoDelGrafo(nodoBajoCursor);
            }
        });

        network.on("doubleClick", function (params) {
            if (params.nodes.length > 0) {
                expandirEntidad(params.nodes[0], false);
            }
        });

        function buscarEntidadInput() {
            const val = document.getElementById('searchInput').value.trim();
            if (val) expandirEntidad(val, true);
        }

        function reiniciarLienzo() {
            nodes.clear();
            edges.clear();
            expandidos.clear();
            nodoActualId = null;
            document.getElementById('searchInput').value = '';
            document.getElementById("inspectorTypeBadge").textContent = "ENCICLOPEDIA";
            document.getElementById("inspectorIdLabel").textContent = "Listo";
            document.getElementById('inspectorContent').innerHTML = `
                <div class="text-center py-14 text-gi-cream/60 space-y-2">
                    <div class="text-2xl text-gi-gold">✦</div>
                    <p class="text-sm">Busca una entidad arriba para iniciar tu camino en la constelación.</p>
                </div>
            `;
        }

        document.getElementById('searchInput').addEventListener('keydown', e => {
            if (e.key === 'Enter') buscarEntidadInput();
        });

        actualizarSugerencias("");
    </script>
</body>
</html>
    """


if __name__ == "__main__":
    puerto = int(os.getenv("PORT", "8000"))
    host = "0.0.0.0" if os.getenv("PORT") else "127.0.0.1"
    uvicorn.run(app, host=host, port=puerto)
