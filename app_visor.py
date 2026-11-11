import hashlib
import hmac
import html
import json
import os
import re
import requests
from fastapi import Body, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
from neo4j import GraphDatabase
import uvicorn
import crawler_entidades
import etimologia
import procedencia

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
        textos_juego_recuperados = []
        afirmaciones_recuperadas = []
        # 1. Búsqueda vectorial sobre los índices Neo4j ('chunk_embeddings' + 'textojuego_embeddings') o léxica (en Render)
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

                    try:
                        res_tj_vec = session.run(
                            """
                            CALL db.index.vector.queryNodes('textojuego_embeddings', 6, $vec)
                            YIELD node AS t, score
                            RETURN t.id AS id, t.tipo AS tipo, t.titulo AS titulo, t.titulo_obra AS obra,
                                   t.texto AS texto, t.version_juego AS version_juego, t.entidad_id AS entidad_id, score
                            ORDER BY score DESC
                            """,
                            vec=vectores_q[0],
                        ).data()
                        textos_juego_recuperados.extend(res_tj_vec)
                    except Exception:
                        pass
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

                res_tj_kw = session.run(
                    """
                    MATCH (t:TextoJuego)
                    WITH t, size([w IN $palabras WHERE toLower(t.texto) CONTAINS toLower(w) OR toLower(t.titulo) CONTAINS toLower(w)]) AS coincidencias
                    WHERE coincidencias > 0
                    RETURN t.id AS id, t.tipo AS tipo, t.titulo AS titulo, t.titulo_obra AS obra,
                           t.texto AS texto, t.version_juego AS version_juego, t.entidad_id AS entidad_id,
                           toFloat(coincidencias) / toFloat(size($palabras)) AS score
                    ORDER BY coincidencias DESC
                    LIMIT 5
                    """,
                    palabras=palabras_clave,
                ).data()
                textos_juego_recuperados.extend(res_tj_kw)
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
        for tj in textos_juego_recuperados:
            ce = obtener_nombre_canonico(tj.get("entidad_id") or "")
            if ce and ce.lower() not in vistas_ent and len(entidades_clave) < 7:
                vistas_ent.add(ce.lower())
                entidades_clave.append(ce)

        # 3. Recorrido de Grafo (1 salto RELATED_TO) desde las entidades clave
        relaciones_recuperadas = []
        if entidades_clave:
            # Recuperar textos canon del juego vinculados a estas entidades clave (sin duplicar los ya hallados)
            res_tj = session.run(
                """
                MATCH (t:TextoJuego)
                WHERE t.entidad_id IN $ents
                   OR any(e IN $ents WHERE (t)-[:SOBRE]->(:Entity {id: e}))
                   OR any(e IN $ents WHERE (t)-[:MENCIONA]->(:Entity {id: e}))
                RETURN t.id AS id, t.tipo AS tipo, t.titulo AS titulo, t.titulo_obra AS obra,
                       t.texto AS texto, t.version_juego AS version_juego, t.entidad_id AS entidad_id
                ORDER BY t.tipo, t.titulo
                LIMIT 6
                """,
                ents=entidades_clave
            ).data()
            ids_tj_ya = {t["id"] for t in textos_juego_recuperados}
            for tj in res_tj:
                if tj["id"] not in ids_tj_ya:
                    textos_juego_recuperados.append(tj)
                    ids_tj_ya.add(tj["id"])

            # Recuperar también afirmaciones externas / teorías vinculadas para contrastar canon vs teoría
            afirmaciones_recuperadas = session.run(
                """
                MATCH (a:Afirmacion)
                WHERE a.entidad_id IN $ents AND a.estado <> 'refutada'
                RETURN a.entidad_id AS entidad_id, a.texto AS texto, a.certeza AS certeza,
                       a.tipo_fuente AS tipo_fuente, a.fuente_titulo AS fuente_titulo, a.version_juego AS version_juego
                LIMIT 8
                """,
                ents=entidades_clave,
            ).data()

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

    bloques_contexto_juego = []
    for idx, tj in enumerate(textos_juego_recuperados[:8], start=1):
        cuerpo_tj = (tj.get("texto") or "").strip()[:1800]
        ver_tj = f" (v{tj['version_juego']})" if tj.get("version_juego") else ""
        bloques_contexto_juego.append(
            f"[Texto Canon del Juego TJ{idx} | Tipo: {tj.get('tipo', 'juego')} | Título: {tj.get('titulo')}{ver_tj} | Obra: {tj.get('obra')}]\n{cuerpo_tj}"
        )

    bloques_contexto_afirm = []
    for idx, af in enumerate(afirmaciones_recuperadas[:8], start=1):
        bloques_contexto_afirm.append(
            f"[Afirmación Externa A{idx} | Entidad: {af.get('entidad_id')} | Certeza: {af.get('certeza')} | Fuente: {af.get('fuente_titulo') or af.get('tipo_fuente')}]\n{af.get('texto')}"
        )

    contexto_total = (
        "=== TEXTOS CANON DEL JUEGO (LIBROS, HISTORIAS DE ARMAS Y ARTEFACTOS - AMBER) ===\n"
        + ("\n\n".join(bloques_contexto_juego) if bloques_contexto_juego else "Sin textos directos del juego.")
        + "\n\n=== FRAGMENTOS DE PÁGINAS DE LA WIKI (CHUNKS JERÁRQUICOS SMALL-TO-BIG) ===\n"
        + ("\n\n".join(bloques_contexto_chunks) if bloques_contexto_chunks else "Sin fragmentos directos.")
        + "\n\n=== RELACIONES Y CITAS DEL GRAFO DE CONOCIMIENTO (NEO4J) ===\n"
        + ("\n\n".join(bloques_contexto_rels) if bloques_contexto_rels else "Sin relaciones adicionales.")
        + ("\n\n=== AFIRMACIONES EXTERNAS, INFERENCIAS Y TEORÍAS ===\n" + "\n\n".join(bloques_contexto_afirm) if bloques_contexto_afirm else "")
    )

    instruccion_sistema = (
        "Eres el 'Oráculo del Irminsul', un asistente experto en el lore de Genshin Impact que ayuda a una autora "
        "a escribir su libro enciclopédico de lore en español.\n"
        "REGLAS ESTRICTAS:\n"
        "1. Responde SIEMPRE en español con estilo enciclopédico, claro, bien estructurado y riguroso.\n"
        "2. Basa tu respuesta con máxima prioridad en los Textos Canon del Juego [TJ1..TJn], seguidos de las Fuentes de la Wiki [F1..Fn] y Relaciones [R1..Rn].\n"
        "3. Si usas Afirmaciones Externas [A1..An], DEBES distinguir explícitamente si son 'inferido' o 'teoría' y nunca presentarlas como canon confirmado.\n"
        "4. Cita explícitamente en el texto de dónde sale cada dato usando corchetes, por ejemplo: "
        "[Libro: The Fox in the Dandelion Sea (TJ1)], [Capitano - The Cataclysm (F1)], [Relación Capitano ➔ Mavuika (R2)] o [Teoría (A1)].\n"
        "5. Utiliza la terminología oficial en español de Genshin Impact (Los Once de los Fatui, Reina Menor Kusanali, "
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


def clasificar_certeza_wiki(texto: str) -> str:
    """Detecta expresiones de incertidumbre o especulación en textos de la Wiki para separar canon firme de inferencias."""
    t = (texto or "").lower()
    marcadores_teoria = ("it is speculated", "theorized", "speculation", "rumor", "rumored")
    marcadores_inferencia = (
        "presumably", "possibly", "probably", "implied", "likely", "around ",
        "approximately", "circa", "sometime ", "unclear", "unknown", "believed to", "may have"
    )
    if any(k in t for k in marcadores_teoria):
        return "teoria"
    if any(k in t for k in marcadores_inferencia):
        return "inferido"
    return "canon"


def _resumir_titulo_hecho(texto: str, max_len: int = 64) -> str:
    """Extrae un titular corto (primera frase o corte limpio) para el nodo visual del eje cronológico."""
    t = re.sub(r"\s+", " ", (texto or "")).strip()
    punto = t.find(". ")
    if 18 <= punto <= max_len:
        return t[:punto]
    return t[:max_len].rstrip(" ,;:") + ("..." if len(t) > max_len else "")


def _fusionar_hitos_afirmaciones(eras_base: list[dict]) -> list[dict]:
    """Combina los hitos base cacheados de la Wiki con las afirmaciones externas cronológicas actuales de Neo4j."""
    hitos_ext = []
    try:
        hitos_ext = procedencia.listar_hitos_cronologicos(driver)
    except Exception:
        hitos_ext = []

    por_era: dict[str, list[dict]] = {}
    for a in hitos_ext:
        if a.get("estado") == "refutada":
            continue
        eid = a.get("era_id") or ""
        por_era.setdefault(eid, []).append({
            "id": f"ext_{a['id']}",
            "afirmacion_id": a["id"],
            "region": a.get("fecha_lore") or "Fuente externa",
            "nivel": 1,
            "titulo_corto": _resumir_titulo_hecho(a.get("texto", "")),
            "texto": a.get("texto", ""),
            "entidades": [a["entidad_id"]] if a.get("entidad_id") else [],
            "ref_titulo": a.get("fuente_titulo") or a.get("tipo_fuente") or "Fuente externa",
            "ref_url": a.get("fuente_url") or "",
            "certeza": a.get("certeza") or "teoria",
            "es_externo": True,
            "fecha_lore": a.get("fecha_lore") or "",
            "nota": a.get("nota") or "",
            "version_juego": a.get("version_juego") or "",
        })

    resultado = []
    for era in eras_base:
        copia = {**era, "hechos": list(era["hechos"]) + por_era.get(era["id"], [])}
        resultado.append(copia)
    return resultado


@app.get("/api/timeline_teyvat")
def obtener_linea_temporal_teyvat():
    """
    Construye la Línea del Tiempo Completa de Teyvat combinando la página oficial 'Timeline'
    y las 8 subpáginas 'Nación/History' de la Wiki junto con los hitos externos de procedencia.
    """
    if CACHE_TIMELINE_TEYVAT:
        return {"eras": _fusionar_hitos_afirmaciones(CACHE_TIMELINE_TEYVAT), "edicion_habilitada": bool(ADMIN_TOKEN)}

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
                    "titulo_corto": _resumir_titulo_hecho(texto_limpio),
                    "texto": texto_limpio,
                    "entidades": entidades_linea[:8],
                    "ref_titulo": ref_titulo[:100],
                    "ref_url": ref_url or f"{WIKI_BASE_URL}Timeline",
                    "certeza": clasificar_certeza_wiki(texto_limpio),
                    "es_externo": False,
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
    return {"eras": _fusionar_hitos_afirmaciones(CACHE_TIMELINE_TEYVAT), "edicion_habilitada": bool(ADMIN_TOKEN)}


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

        # Incorporar textos oficiales del cliente de juego (Libros, Armas, Artefactos) como secciones nativas de la entidad
        recs_juego = session.run(
            """
            MATCH (t:TextoJuego)
            WHERE t.entidad_id = $id OR (t)-[:SOBRE]->(:Entity {id: $id})
            RETURN t.id AS id, t.tipo AS tipo, t.titulo AS titulo, t.titulo_obra AS obra,
                   t.texto AS texto, t.version_juego AS version_juego
            ORDER BY t.tipo, t.titulo
            """,
            id=id_principal
        ).data()

        iconos_tipo = {"libro": "📚", "arma": "⚔️", "artefacto": "🛡️"}
        for tj in recs_juego:
            cuerpo = (tj.get("texto") or "").strip()
            if not cuerpo:
                continue
            ico = iconos_tipo.get(tj.get("tipo"), "📖")
            subtipo = (tj.get("tipo") or "juego").capitalize()
            ver = f" (v{tj['version_juego']})" if tj.get("version_juego") else ""
            secciones.append({
                "titulo": f"{ico} [Juego · {subtipo}] {tj['titulo']}{ver}",
                "contenido": cuerpo,
                "es_texto_juego": True,
                "tipo_juego": tj.get("tipo"),
                "abierto": False
            })

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


ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "").strip()


def _exigir_admin(token: str | None):
    """Las escrituras de procedencia solo se permiten con ADMIN_TOKEN configurado y correcto."""
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Edición deshabilitada: define ADMIN_TOKEN en el servidor.")
    if not token or not hmac.compare_digest(token, ADMIN_TOKEN):
        raise HTTPException(status_code=401, detail="Token de administrador incorrecto.")


@app.get("/api/procedencia/{entidad_id:path}")
def obtener_procedencia(entidad_id: str):
    canon_id = obtener_nombre_canonico(entidad_id.strip())
    return {
        "entidad_id": canon_id,
        "canon": {
            "tipo_fuente": "wiki",
            "fuente_titulo": "Genshin Impact Wiki (Fandom)",
            "fuente_url": f"{WIKI_BASE_URL}{canon_id.replace(' ', '_')}",
        },
        "afirmaciones": procedencia.listar_afirmaciones(driver, canon_id),
        "edicion_habilitada": bool(ADMIN_TOKEN),
    }


@app.post("/api/procedencia/{entidad_id:path}")
def crear_procedencia(entidad_id: str, datos: dict = Body(...), x_admin_token: str | None = Header(default=None)):
    _exigir_admin(x_admin_token)
    canon_id = obtener_nombre_canonico(entidad_id.strip())
    try:
        return procedencia.crear_afirmacion(driver, canon_id, datos)
    except procedencia.ErrorValidacion as e:
        raise HTTPException(status_code=422, detail=str(e))
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.patch("/api/afirmacion/{afirmacion_id}")
def editar_afirmacion(afirmacion_id: str, datos: dict = Body(...), x_admin_token: str | None = Header(default=None)):
    _exigir_admin(x_admin_token)
    try:
        return procedencia.actualizar_afirmacion(driver, afirmacion_id, datos)
    except procedencia.ErrorValidacion as e:
        raise HTTPException(status_code=422, detail=str(e))
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/api/afirmacion/{afirmacion_id}")
def eliminar_afirmacion(afirmacion_id: str, x_admin_token: str | None = Header(default=None)):
    _exigir_admin(x_admin_token)
    if not procedencia.borrar_afirmacion(driver, afirmacion_id):
        raise HTTPException(status_code=404, detail="Afirmación no encontrada")
    return {"ok": True}


try:
    procedencia.asegurar_indices(driver)
    etimologia.asegurar_indices(driver)
except Exception as _e:
    print(f"⚠️ No se pudieron crear los índices de procedencia/etimología: {_e}")


@app.get("/api/textos_juego/{entidad_id:path}")
def obtener_textos_juego(entidad_id: str):
    """Textos canon del juego (libros, armas, artefactos) importados de Project Amber que tratan de la entidad."""
    canon_id = obtener_nombre_canonico(entidad_id.strip())
    with driver.session() as session:
        recs = session.run(
            """
            MATCH (t:TextoJuego)
            WHERE t.entidad_id = $id OR $id IN t.menciona
            RETURN t.id AS id, t.tipo AS tipo, t.titulo AS titulo, t.titulo_obra AS obra,
                   t.descripcion AS descripcion, t.texto AS texto, t.version_juego AS version_juego,
                   t.fuente AS fuente, (t.entidad_id = $id) AS por_titulo
            ORDER BY por_titulo DESC, t.tipo, t.titulo
            LIMIT 30
            """,
            id=canon_id,
        )
        textos = [
            {**dict(r), "texto": (r["texto"] or "")[:6000], "relacion": "titulo" if r["por_titulo"] else "mencion"}
            for r in recs
        ]
    return {"entidad_id": canon_id, "textos": textos}


@app.get("/api/etimologia/{entidad_id:path}")
def obtener_etimologia(entidad_id: str):
    canon_id = obtener_nombre_canonico(entidad_id.strip())
    ficha = etimologia.obtener(driver, canon_id)
    # Si aún no tiene resumen de Wikipedia guardado en Neo4j, intentar buscarlo y persistirlo automáticamente
    if not ficha or not ficha.get("wikipedia_resumen"):
        wiki_data = etimologia.buscar_en_wikipedia(canon_id)
        if wiki_data:
            try:
                ficha = etimologia.guardar_wikipedia(driver, canon_id, wiki_data)
            except LookupError:
                pass
    return {
        "entidad_id": canon_id,
        "ficha": ficha,
        "wiki_url": f"{WIKI_BASE_URL}{canon_id.replace(' ', '_')}",
        "edicion_habilitada": bool(ADMIN_TOKEN),
    }


@app.post("/api/etimologia_wikipedia/{entidad_id:path}")
def buscar_y_guardar_wikipedia(entidad_id: str, datos: dict = Body(default={}), x_admin_token: str | None = Header(default=None)):
    """Busca un término concreto en Wikipedia real (ES/EN) y guarda su resumen en Neo4j para esa entidad."""
    _exigir_admin(x_admin_token)
    canon_id = obtener_nombre_canonico(entidad_id.strip())
    termino = (datos.get("termino") or canon_id).strip()
    wiki_data = etimologia.buscar_en_wikipedia(termino)
    if not wiki_data:
        raise HTTPException(status_code=404, detail=f"No se encontró artículo en Wikipedia para '{termino}'.")
    try:
        return etimologia.guardar_wikipedia(driver, canon_id, wiki_data)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.put("/api/etimologia/{entidad_id:path}")
def guardar_etimologia(entidad_id: str, datos: dict = Body(...), x_admin_token: str | None = Header(default=None)):
    _exigir_admin(x_admin_token)
    canon_id = obtener_nombre_canonico(entidad_id.strip())
    try:
        return etimologia.guardar(driver, canon_id, datos)
    except etimologia.ErrorValidacion as e:
        raise HTTPException(status_code=422, detail=str(e))
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/api/etimologia/{entidad_id:path}")
def eliminar_etimologia(entidad_id: str, x_admin_token: str | None = Header(default=None)):
    _exigir_admin(x_admin_token)
    canon_id = obtener_nombre_canonico(entidad_id.strip())
    if not etimologia.borrar(driver, canon_id):
        raise HTTPException(status_code=404, detail="No hay ficha de etimología para esta entidad")
    return {"ok": True}


@app.post("/api/etimologia_borrador/{entidad_id:path}")
def borrador_resumen_wiki(entidad_id: str, x_admin_token: str | None = Header(default=None)):
    """Genera (sin guardar) un resumen en español del artículo de la Wiki, usando solo ese texto."""
    _exigir_admin(x_admin_token)
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY no configurada en el servidor.")
    canon_id = obtener_nombre_canonico(entidad_id.strip())
    with driver.session() as session:
        rec = session.run("MATCH (e:Entity {id: $id}) RETURN e.texto_web AS t", id=canon_id).single()
        texto_bruto = asegurar_articulo_completo_wiki(session, canon_id, (rec["t"] if rec else "") or "")
    texto, _, _ = parsear_texto_y_atributos_wiki(texto_bruto)
    if not texto or not texto.strip():
        raise HTTPException(status_code=404, detail="No hay texto de la Wiki para resumir.")
    instruccion = (
        "Resume en español, en 3 a 5 frases, el artículo de la Wiki de Genshin Impact que te paso. "
        "Usa ÚNICAMENTE información presente en el texto: no añadas etimologías, teorías ni datos externos. "
        "Devuelve solo el resumen."
    )
    resumen, err = llamar_gemini_con_fallback(
        instruccion, [{"role": "user", "parts": [{"text": texto[:12000]}]}], temperature=0.2
    )
    if not resumen:
        raise HTTPException(status_code=502, detail=err or "No se pudo generar el resumen.")
    return {"resumen_wiki": resumen[:3000], "resumen_auto": True}


RUTA_ICONO_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icono.png")
RUTA_FAVICON_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "favicon.png")


@app.get("/icono.png")
def obtener_icono_web():
    ruta = RUTA_ICONO_PNG if os.path.exists(RUTA_ICONO_PNG) else RUTA_FAVICON_PNG
    if os.path.exists(ruta):
        with open(ruta, "rb") as f:
            return Response(content=f.read(), media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})
    return Response(status_code=404)


@app.get("/favicon.png")
@app.get("/favicon.ico")
def obtener_favicon_web():
    ruta = RUTA_FAVICON_PNG if os.path.exists(RUTA_FAVICON_PNG) else RUTA_ICONO_PNG
    if os.path.exists(ruta):
        with open(ruta, "rb") as f:
            return Response(content=f.read(), media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})
    return Response(status_code=404)


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
    <link rel="icon" type="image/png" href="/favicon.png">
    <link rel="shortcut icon" type="image/png" href="/favicon.png">
    <link rel="apple-touch-icon" href="/favicon.png">
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
                <div class="w-11 h-11 rounded-full bg-[#141824] border-2 border-gi-gold flex items-center justify-center shadow-[0_0_14px_rgba(211,188,142,0.35)] p-1 shrink-0">
                    <img src="/icono.png" alt="Irminsul" class="w-full h-full object-contain drop-shadow-[0_0_6px_rgba(240,220,175,0.45)]">
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

        <!-- VISTA ALTERNATIVA: LÍNEA DEL TIEMPO INTERACTIVA DE TEYVAT -->
        <div id="teyvatTimelineViewport" class="hidden flex-1 min-h-0 gi-constellation-canvas flex flex-col overflow-hidden relative">
            <!-- Barra Superior: Regla Cronológica Horizontal de las 12 Eras de Teyvat -->
            <div class="bg-[#161b28]/95 border-b border-gi-gold/40 px-4 py-2.5 shrink-0 z-20 space-y-2">
                <div class="flex flex-wrap items-center justify-between gap-2">
                    <div class="flex items-center gap-2.5">
                        <span class="text-gi-goldBright text-base">⏳</span>
                        <div>
                            <div class="font-serif font-bold text-xs md:text-sm text-gi-goldBright tracking-wide">Crónica Visual de Teyvat</div>
                            <div class="text-[10px] text-[#9aa3b8]">Haz clic en cualquier época o hito del eje para ver detalles y certeza</div>
                        </div>
                    </div>
                    <div class="flex flex-wrap items-center gap-1.5">
                        <button onclick="alternarFiltroCerteza('canon')" id="btnFiltroCanon" class="px-2.5 py-1 rounded-full text-[11px] font-bold border border-emerald-400/50 bg-emerald-500/20 text-emerald-200 transition">✓ Canon</button>
                        <button onclick="alternarFiltroCerteza('inferido')" id="btnFiltroInferido" class="px-2.5 py-1 rounded-full text-[11px] font-bold border border-amber-400/50 bg-amber-500/20 text-amber-200 transition">✓ Inferido</button>
                        <button onclick="alternarFiltroCerteza('teoria')" id="btnFiltroTeoria" class="px-2.5 py-1 rounded-full text-[11px] font-bold border border-fuchsia-400/50 bg-fuchsia-500/20 text-fuchsia-200 transition">✓ Teorías</button>
                        <button onclick="alternarModoPrincipal('grafo')" class="ml-1 px-3 py-1 rounded-full bg-[#232b3e] hover:bg-gi-gold/25 border border-gi-gold/40 text-[11px] text-gi-cream transition flex items-center gap-1">
                            <i class="fa-solid fa-arrow-left text-[9px] text-gi-gold"></i> Volver al Grafo
                        </button>
                    </div>
                </div>
                <!-- Eje horizontal scrollable de las 12 épocas -->
                <div id="listaErasTeyvatSidebar" class="flex items-center gap-2 overflow-x-auto pb-1 pt-0.5"></div>
            </div>

            <!-- Lienzo Central: Eje visual de hitos clicables + Panel de detalle al pulsar -->
            <div id="detalleEraTeyvatContent" class="flex-1 min-h-0 flex flex-col overflow-hidden z-10">
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
        let certezaTeyvatFiltro = new Set(["canon", "inferido", "teoria"]);
        let hechoTeyvatSeleccionadoId = null;
        let edicionTimelineHabilitada = false;
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

        async function cargarLineaTemporalTeyvat(forzarRecarga = false) {
            if (datosTimelineTeyvat.length > 0 && !forzarRecarga) {
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
                edicionTimelineHabilitada = Boolean(data.edicion_habilitada);
                if (eraTeyvatActivaIdx >= datosTimelineTeyvat.length) eraTeyvatActivaIdx = 0;
                if (!forzarRecarga) {
                    regionTeyvatFiltro = "TODAS";
                    hechoTeyvatSeleccionadoId = null;
                }
                renderizarSidebarErasTeyvat();
                renderizarDetalleEraTeyvat();
            } catch (e) {
                if (contDetalle) contDetalle.innerHTML = `<div class="p-4 rounded-xl bg-[#181d2b] border border-gi-gold/30 text-xs text-red-300">Error cargando la línea del tiempo: ${escaparHtml(e.message)}</div>`;
            }
        }

        function alternarFiltroCerteza(tipo) {
            if (certezaTeyvatFiltro.has(tipo)) {
                if (certezaTeyvatFiltro.size === 1) return; // evitar dejar todos vacíos por accidente
                certezaTeyvatFiltro.delete(tipo);
            } else {
                certezaTeyvatFiltro.add(tipo);
            }
            const mapaBtn = {
                canon: { id: 'btnFiltroCanon', txt: 'Canon', cls: 'border-emerald-400/50 bg-emerald-500/20 text-emerald-200' },
                inferido: { id: 'btnFiltroInferido', txt: 'Inferido', cls: 'border-amber-400/50 bg-amber-500/20 text-amber-200' },
                teoria: { id: 'btnFiltroTeoria', txt: 'Teorías', cls: 'border-fuchsia-400/50 bg-fuchsia-500/20 text-fuchsia-200' }
            };
            Object.entries(mapaBtn).forEach(([k, cfg]) => {
                const b = document.getElementById(cfg.id);
                if (!b) return;
                const on = certezaTeyvatFiltro.has(k);
                b.textContent = (on ? '✓ ' : '○ ') + cfg.txt;
                b.className = `px-2.5 py-1 rounded-full text-[11px] font-bold border transition ${on ? cfg.cls : 'border-gi-gold/20 bg-[#181d2b] text-[#7c8599] opacity-60'}`;
            });
            renderizarDetalleEraTeyvat();
        }

        function seleccionarEraTeyvat(idx) {
            eraTeyvatActivaIdx = idx;
            regionTeyvatFiltro = "TODAS";
            hechoTeyvatSeleccionadoId = null;
            renderizarSidebarErasTeyvat();
            renderizarDetalleEraTeyvat();
        }

        function seleccionarRegionEraTeyvat(region) {
            regionTeyvatFiltro = region;
            hechoTeyvatSeleccionadoId = null;
            renderizarDetalleEraTeyvat();
        }

        function seleccionarHechoTeyvat(hechoId) {
            hechoTeyvatSeleccionadoId = hechoId;
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
                const cCanon = era.hechos.filter(h => (h.certeza || 'canon') === 'canon').length;
                const cSup = era.hechos.length - cCanon;
                return `
                    <button onclick="seleccionarEraTeyvat(${idx})" class="shrink-0 min-w-[165px] max-w-[210px] text-left px-3 py-2 rounded-xl border transition relative ${activo ? 'bg-gi-gold text-[#161b26] border-gi-goldBright shadow-[0_0_16px_rgba(240,220,175,0.45)] font-bold' : 'bg-[#1b2234]/90 text-gi-cream border-gi-gold/25 hover:border-gi-gold/70'}">
                        <div class="flex items-center justify-between gap-1 text-[10px] ${activo ? 'text-[#2c3345]' : 'text-gi-gold'}">
                            <span class="font-bold">ÉPOCA ${era.orden}</span>
                            <span>${cCanon}✓${cSup ? ' · ' + cSup + '~' : ''}</span>
                        </div>
                        <div class="font-serif text-[11.5px] leading-tight truncate mt-0.5">${escaparHtml(era.titulo_es.replace(/^[IVXLCDM]+\.\s*/, ''))}</div>
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
                <button onclick="seleccionarRegionEraTeyvat('TODAS')" class="px-2.5 py-0.5 rounded-full text-[11px] font-semibold transition ${regionTeyvatFiltro === 'TODAS' ? 'bg-gi-gold text-[#161b26] font-bold shadow' : 'bg-[#181d2b] text-gi-cream/85 border border-gi-gold/30 hover:border-gi-gold'}">
                    ✦ Todas (${era.hechos.length})
                </button>
            `;
            (era.regiones || []).forEach(reg => {
                const cnt = era.hechos.filter(h => h.region === reg).length;
                const act = (regionTeyvatFiltro === reg);
                pillsRegiones += `
                    <button onclick="seleccionarRegionEraTeyvat('${escaparComillas(reg)}')" class="px-2.5 py-0.5 rounded-full text-[11px] font-medium transition ${act ? 'bg-gi-gold text-[#161b26] font-bold shadow' : 'bg-[#181d2b] text-gi-cream/85 border border-gi-gold/30 hover:border-gi-gold'}">
                        ${escaparHtml(reg)} (${cnt})
                    </button>
                `;
            });

            // Hechos filtrados por región, texto y nivel de certeza (canon / inferido / teoria)
            const hechosFiltrados = era.hechos
                .map((h, idxOriginal) => ({ ...h, idxOriginal, certeza: h.certeza || 'canon' }))
                .filter(h => regionTeyvatFiltro === 'TODAS' || h.region === regionTeyvatFiltro)
                .filter(h => certezaTeyvatFiltro.has(h.certeza))
                .filter(h => {
                    if (!filtroTxt) return true;
                    const t = ((h.mostrar_es && h.texto_es) ? h.texto_es : h.texto).toLowerCase();
                    const ents = (h.entidades || []).join(' ').toLowerCase();
                    return t.includes(filtroTxt) || ents.includes(filtroTxt) || h.region.toLowerCase().includes(filtroTxt);
                });

            if (hechosFiltrados.length && !hechosFiltrados.some(h => h.id === hechoTeyvatSeleccionadoId)) {
                hechoTeyvatSeleccionadoId = hechosFiltrados[0].id;
            }
            const hechoActivo = hechosFiltrados.find(h => h.id === hechoTeyvatSeleccionadoId) || null;

            const ESTILOS_NODO = {
                canon: { borde: 'border-emerald-400/70', punto: 'bg-emerald-400 border-emerald-200', badge: 'bg-emerald-500/20 text-emerald-300 border-emerald-400/40', txt: 'Canon', linea: 'border-solid' },
                inferido: { borde: 'border-amber-400/70', punto: 'bg-amber-400 border-amber-200', badge: 'bg-amber-500/20 text-amber-300 border-amber-400/40', txt: 'Inferido', linea: 'border-dashed' },
                teoria: { borde: 'border-fuchsia-400/70', punto: 'bg-fuchsia-400 border-fuchsia-200', badge: 'bg-fuchsia-500/20 text-fuchsia-300 border-fuchsia-400/40', txt: 'Teoría', linea: 'border-dotted' }
            };

            // Nodos visuales interactivos sobre el eje horizontal de la época
            const htmlNodosEje = hechosFiltrados.map((h, pos) => {
                const st = ESTILOS_NODO[h.certeza] || ESTILOS_NODO.canon;
                const activo = (hechoActivo && h.id === hechoActivo.id);
                const tituloCorto = h.mostrar_es && h.texto_es
                    ? (h.texto_es.slice(0, 62) + (h.texto_es.length > 62 ? '...' : ''))
                    : (h.titulo_corto || h.texto.slice(0, 62));
                const arriba = (pos % 2 === 0);
                const entPrincipal = (h.entidades && h.entidades[0]) || '';

                return `
                <div class="shrink-0 w-56 flex flex-col items-center relative group cursor-pointer" onclick="seleccionarHechoTeyvat('${escaparComillas(h.id)}')">
                    <div class="w-full p-2.5 rounded-xl bg-[#181d2b]/95 ${st.linea} border-2 ${activo ? 'border-gi-goldBright ring-2 ring-gi-gold/40 bg-[#232b3e] scale-[1.02]' : st.borde + ' hover:border-gi-gold'} transition shadow-lg space-y-1.5 ${arriba ? 'mb-3' : 'mt-3 order-3'}">
                        <div class="flex items-center justify-between gap-1 text-[9.5px]">
                            <span class="px-1.5 py-0.5 rounded-full border font-bold ${st.badge}">${st.txt}</span>
                            <span class="text-[#9fd6e3] truncate font-medium">${escaparHtml(h.region)}</span>
                        </div>
                        <div class="text-[11px] font-semibold text-gi-cream leading-snug line-clamp-3">${escaparHtml(tituloCorto)}</div>
                        ${entPrincipal ? `
                        <div class="flex items-center gap-1.5 pt-1 border-t border-gi-gold/15">
                            <img src="${urlAvatar(entPrincipal)}" class="w-4 h-4 rounded-full object-cover border border-gi-gold/60 shrink-0" alt="">
                            <span class="text-[10px] text-gi-goldBright truncate">${escaparHtml(entPrincipal)}${h.entidades.length > 1 ? ' +' + (h.entidades.length - 1) : ''}</span>
                        </div>` : ''}
                    </div>
                    <div class="w-0.5 h-3 bg-gi-gold/60 ${arriba ? '' : 'order-2'}"></div>
                    <div class="w-4 h-4 rounded-full border-2 ${activo ? 'bg-gi-goldBright border-white scale-125 shadow-[0_0_12px_rgba(240,220,175,0.9)]' : st.punto} transition z-10 ${arriba ? '' : 'order-1'}"></div>
                </div>`;
            }).join('');

            // Panel inferior de detalle al hacer clic en un nodo del eje
            let htmlDetalleSeleccionado = '<div class="p-6 text-center text-xs text-[#9aa3b8] italic">Haz clic sobre cualquier nodo en la línea temporal superior para inspeccionar su texto completo, certeza, fuentes y entidades vinculadas.</div>';
            if (hechoActivo) {
                const st = ESTILOS_NODO[hechoActivo.certeza] || ESTILOS_NODO.canon;
                const enEs = Boolean(hechoActivo.mostrar_es && hechoActivo.texto_es);
                const txtMostrar = enEs ? hechoActivo.texto_es : hechoActivo.texto;
                const etiqBtn = hechoActivo.texto_es ? (enEs ? '🌐 Ver EN' : '🌐 Ver ES') : '🌐 Traducir';
                const pillsEnts = (hechoActivo.entidades && hechoActivo.entidades.length)
                    ? `<div class="flex flex-wrap items-center gap-1.5 pt-2">
                        ${hechoActivo.entidades.map(ent => `
                            <button onclick="mostrarFichaNodo('${escaparComillas(ent)}')" class="px-2.5 py-1 rounded-full bg-[#181d2b] hover:bg-[#252e45] border border-gi-gold/40 text-[11px] font-semibold text-gi-cream flex items-center gap-1.5 transition">
                                <img src="${urlAvatar(ent)}" class="w-4 h-4 rounded-full object-cover border border-gi-gold" alt="">
                                <span>${escaparHtml(ent)}</span>
                            </button>
                        `).join('')}
                       </div>`
                    : '';

                htmlDetalleSeleccionado = `
                    <div class="gi-parchment-box rounded-xl overflow-hidden shadow-lg">
                        <div class="px-4 py-2.5 bg-[#e2d8c3] border-b border-[#c8b082] flex flex-wrap items-center justify-between gap-2">
                            <div class="flex flex-wrap items-center gap-2">
                                <span class="px-2.5 py-0.5 rounded-full bg-[#1b202e] text-gi-goldBright text-[10px] font-bold">${escaparHtml(st.txt.toUpperCase())}</span>
                                <span class="px-2.5 py-0.5 rounded-full bg-[#3b4255] text-[#f0dcaf] text-[10px] font-bold">📍 ${escaparHtml(hechoActivo.region)}</span>
                                ${hechoActivo.version_juego ? `<span class="text-[10px] font-bold text-[#5a4622]">v${escaparHtml(hechoActivo.version_juego)}</span>` : ''}
                                ${hechoActivo.ref_url
                                    ? `<a href="${escaparHtml(urlSegura(hechoActivo.ref_url))}" target="_blank" rel="noopener noreferrer" class="text-[11px] font-semibold text-[#5a4622] hover:underline flex items-center gap-1"><i class="fa-solid fa-book-bookmark text-[10px]"></i> ${escaparHtml(hechoActivo.ref_titulo)} ↗</a>`
                                    : `<span class="text-[11px] font-semibold text-[#5a4622]">${escaparHtml(hechoActivo.ref_titulo || 'Fuente')}</span>`}
                            </div>
                            <div class="flex items-center gap-1.5">
                                <button onclick="traducirHechoTeyvat(${eraTeyvatActivaIdx}, ${hechoActivo.idxOriginal}, this)" class="px-2.5 py-0.5 rounded-full bg-[#3b4255] hover:bg-[#252b3b] border border-[#d3bc8e] text-[10px] font-semibold text-[#f4efe6] transition">${etiqBtn}</button>
                                <button onclick="copiarTextoAlPortapapeles(datosTimelineTeyvat[${eraTeyvatActivaIdx}].hechos[${hechoActivo.idxOriginal}].mostrar_es ? datosTimelineTeyvat[${eraTeyvatActivaIdx}].hechos[${hechoActivo.idxOriginal}].texto_es : datosTimelineTeyvat[${eraTeyvatActivaIdx}].hechos[${hechoActivo.idxOriginal}].texto, this)" class="px-2.5 py-0.5 rounded-full bg-[#f5f0e6] hover:bg-white border border-[#b59d6e] text-[10px] font-semibold text-[#3b4255] transition"><i class="fa-regular fa-copy"></i> Copiar</button>
                                ${hechoActivo.es_externo && edicionTimelineHabilitada ? `<button onclick="borrarHitoCronologico('${escaparComillas(hechoActivo.afirmacion_id)}')" class="px-2.5 py-0.5 rounded-full bg-[#6d222e] text-white text-[10px] font-semibold">Borrar hito</button>` : ''}
                            </div>
                        </div>
                        <div class="p-4 text-xs md:text-sm leading-relaxed font-medium text-[#2c3345] space-y-2">
                            <div>${escaparHtml(txtMostrar)}</div>
                            ${hechoActivo.nota ? `<div class="text-xs italic text-[#5a4622] border-l-2 border-[#8c6d32] pl-2.5">${escaparHtml(hechoActivo.nota)}</div>` : ''}
                            ${pillsEnts}
                        </div>
                    </div>`;
            }

            const formHito = edicionTimelineHabilitada ? `
                <details class="rounded-xl border border-gi-gold/35 bg-[#181d2b]/95 overflow-hidden">
                    <summary class="px-3.5 py-1.5 cursor-pointer select-none list-none text-[11px] font-bold text-gi-goldBright hover:bg-[#20273b]">+ Situar nuevo hito o teoría en esta época</summary>
                    <div class="p-3 border-t border-gi-gold/20 grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
                        <input id="tlEntidad" placeholder="Entidad vinculada exacta (ej. Paimon, Zhongli)" class="bg-[#131722] border border-gi-gold/30 rounded p-1.5 text-gi-cream">
                        <input id="tlFecha" placeholder="Momento / Región (ej. Hace ~2000 años · Liyue)" class="bg-[#131722] border border-gi-gold/30 rounded p-1.5 text-gi-cream">
                        <select id="tlCerteza" class="bg-[#131722] border border-gi-gold/30 rounded p-1.5 text-gi-cream">
                            <option value="canon">Canon (confirmado)</option>
                            <option value="inferido">Inferido (aproximado/deducido)</option>
                            <option value="teoria" selected>Teoría / Hipótesis</option>
                        </select>
                        <input id="tlFuenteUrl" placeholder="URL fuente (opcional https://...)" class="bg-[#131722] border border-gi-gold/30 rounded p-1.5 text-gi-cream">
                        <textarea id="tlTexto" rows="2" placeholder="Descripción del suceso o teoría en esta época..." class="md:col-span-2 bg-[#131722] border border-gi-gold/30 rounded p-1.5 text-gi-cream"></textarea>
                        <button onclick="guardarHitoCronologico('${escaparComillas(era.id)}')" class="gi-btn-cream px-3 py-1 text-[11px] w-fit">Guardar en Línea Temporal</button>
                    </div>
                </details>` : '';

            cont.innerHTML = `
                <!-- Cabecera compacta + Filtros de la Época -->
                <div class="p-3.5 bg-[#151a27]/95 border-b border-gi-gold/30 space-y-2.5 shrink-0">
                    <div class="flex flex-wrap items-center justify-between gap-2">
                        <div>
                            <div class="text-[10px] font-serif uppercase tracking-widest text-gi-gold">ÉPOCA ${era.orden} DE ${datosTimelineTeyvat.length} · ${hechosFiltrados.length} HITOS VISIBLES</div>
                            <h2 class="font-serif text-base md:text-lg font-bold text-gi-goldBright">${escaparHtml(era.titulo_es)}</h2>
                        </div>
                        <div class="flex items-center gap-2 shrink-0">
                            <div class="relative flex items-center">
                                <i class="fa-solid fa-magnifying-glass absolute left-2.5 text-gi-gold/70 text-[10px]"></i>
                                <input type="text" id="filtroTextoEraTeyvat" value="${escaparHtml(filtroTxt)}" oninput="renderizarDetalleEraTeyvat()" placeholder="Filtrar en esta época..." class="w-44 md:w-56 bg-[#111520] border border-gi-gold/30 rounded-full pl-7 pr-3 py-1 text-[11px] text-gi-cream placeholder-[#7c8599] focus:outline-none focus:border-gi-goldBright">
                            </div>
                            <button onclick="traducirTodaEraTeyvat(${eraTeyvatActivaIdx}, this)" class="gi-btn-cream px-3 py-1 text-[11px] flex items-center gap-1"><i class="fa-solid fa-language text-[#3b4255]"></i><span>Traducir Época</span></button>
                            <a href="${era.url_wiki}" target="_blank" rel="noopener noreferrer" class="gi-btn-dark px-2.5 py-1 text-[11px]">Wiki ↗</a>
                        </div>
                    </div>
                    <div class="flex flex-wrap items-center gap-1.5">${pillsRegiones}</div>
                    ${formHito}
                </div>

                <!-- Eje Horizontal Visual Scrollable con Nodos Clicables -->
                <div class="relative bg-[#10141f]/90 border-b border-gi-gold/30 shrink-0">
                    <div class="overflow-x-auto px-6 py-5 flex items-center gap-5 relative">
                        <div class="absolute left-0 right-0 top-1/2 h-0.5 bg-gradient-to-r from-gi-gold/20 via-gi-gold/70 to-gi-gold/20 pointer-events-none"></div>
                        ${htmlNodosEje || '<div class="py-8 w-full text-center text-xs text-[#9aa3b8] italic">No hay hitos que coincidan con los filtros activos.</div>'}
                    </div>
                </div>

                <!-- Panel Inferior de Detalle del Hito Seleccionado -->
                <div class="flex-1 min-h-0 overflow-y-auto p-4 space-y-3 bg-[#141926]/90">
                    ${htmlDetalleSeleccionado}
                </div>
            `;
        }

        async function guardarHitoCronologico(eraId) {
            const v = id => (document.getElementById(id)?.value || '').trim();
            const entidad = v('tlEntidad');
            if (!entidad) { alert('Indica el nombre exacto de la entidad en el grafo.'); return; }
            const cuerpo = {
                texto: v('tlTexto'),
                certeza: v('tlCerteza'),
                tipo_fuente: 'nota_propia',
                fuente_titulo: 'Crónica de Teyvat',
                fuente_url: v('tlFuenteUrl'),
                era_id: eraId,
                fecha_lore: v('tlFecha')
            };
            const r = await llamadaProcedencia(`/api/procedencia/${encodeURIComponent(entidad)}`, 'POST', cuerpo);
            if (r) {
                await cargarLineaTemporalTeyvat(true);
            }
        }

        async function borrarHitoCronologico(afirmacionId) {
            if (!confirm('¿Borrar este hito de la línea temporal?')) return;
            const r = await llamadaProcedencia(`/api/afirmacion/${encodeURIComponent(afirmacionId)}`, 'DELETE');
            if (r) {
                hechoTeyvatSeleccionadoId = null;
                await cargarLineaTemporalTeyvat(true);
            }
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

                    <!-- 5. DESPLEGABLE: Procedencia y Certeza (fuentes externas: vídeos, Reddit, notas) -->
                    <details class="gi-card-inner rounded-xl overflow-hidden group" ontoggle="if(this.open) cargarProcedencia('${escaparComillas(n.id)}')">
                        <summary class="px-4 py-3 cursor-pointer select-none flex items-center justify-between gap-2 bg-gradient-to-r from-[#242c40] to-[#181d2b] hover:from-[#2c364f] transition list-none">
                            <div class="flex items-center gap-2.5">
                                <span class="w-6 h-6 rounded-full bg-gi-gold/20 border border-gi-gold/50 flex items-center justify-center text-gi-goldBright text-xs">🔎</span>
                                <div>
                                    <div class="font-serif font-bold text-xs md:text-sm text-gi-goldBright tracking-wide">Procedencia y Certeza</div>
                                    <div class="text-[10px] text-[#9aa3b8]">Canon de la Wiki frente a teorías y fuentes externas</div>
                                </div>
                            </div>
                            <span class="text-[10px] px-2.5 py-0.5 rounded-full bg-gi-gold/20 text-gi-goldBright border border-gi-gold/40 font-semibold">Desplegar ▾</span>
                        </summary>
                        <div id="contenedorProcedencia" class="p-3.5 border-t border-gi-gold/25 space-y-3 bg-[#151925]/90">
                            <div class="text-xs text-[#9aa3b8] italic">Cargando...</div>
                        </div>
                    </details>

                    <!-- 6. DESPLEGABLE: Etimología y Mundo Real -->
                    <details class="gi-card-inner rounded-xl overflow-hidden group" ontoggle="if(this.open) cargarEtimologia('${escaparComillas(n.id)}')">
                        <summary class="px-4 py-3 cursor-pointer select-none flex items-center justify-between gap-2 bg-gradient-to-r from-[#242c40] to-[#181d2b] hover:from-[#2c364f] transition list-none">
                            <div class="flex items-center gap-2.5">
                                <span class="w-6 h-6 rounded-full bg-gi-gold/20 border border-gi-gold/50 flex items-center justify-center text-gi-goldBright text-xs">🌍</span>
                                <div>
                                    <div class="font-serif font-bold text-xs md:text-sm text-gi-goldBright tracking-wide">Etimología y Mundo Real</div>
                                    <div class="text-[10px] text-[#9aa3b8]">Origen del nombre, mitología y resumen de la Wiki</div>
                                </div>
                            </div>
                            <span class="text-[10px] px-2.5 py-0.5 rounded-full bg-gi-gold/20 text-gi-goldBright border border-gi-gold/40 font-semibold">Desplegar ▾</span>
                        </summary>
                        <div id="contenedorEtimologia" class="p-3.5 border-t border-gi-gold/25 space-y-3 bg-[#151925]/90">
                            <div class="text-xs text-[#9aa3b8] italic">Cargando...</div>
                        </div>
                    </details>

                </div>
            `;

            alternarVistaEnciclopedia(n.id);
            inspector.scrollTop = 0;
        }

        const cacheEtimologia = {};
        const CERTEZA_UI = {
            canon: { etiqueta: 'Canon', clase: 'bg-emerald-500/15 text-emerald-300 border-emerald-400/40' },
            inferido: { etiqueta: 'Inferido', clase: 'bg-amber-500/15 text-amber-300 border-amber-400/40' },
            teoria: { etiqueta: 'Teoría', clase: 'bg-fuchsia-500/15 text-fuchsia-300 border-fuchsia-400/40' }
        };
        const ESTADO_UI = { vigente: 'Vigente', en_revision: 'En revisión', refutada: 'Refutada' };
        const TIPO_FUENTE_UI = { wiki: 'Wiki', juego: 'Juego', youtube: 'YouTube', reddit: 'Reddit', nota_propia: 'Nota propia', otro: 'Otra' };

        function urlSegura(u) {
            return (u && /^https?:\/\//i.test(u)) ? u : '';
        }

        function tokenAdmin(forzarPedir) {
            let t = localStorage.getItem('adminToken') || '';
            if (!t || forzarPedir) {
                t = (prompt('Token de administrador para editar la procedencia:') || '').trim();
                if (t) localStorage.setItem('adminToken', t);
            }
            return t;
        }

        async function llamadaProcedencia(url, metodo, cuerpo) {
            const token = tokenAdmin(false);
            if (!token) return null;
            const res = await fetch(url, {
                method: metodo,
                headers: { 'Content-Type': 'application/json', 'X-Admin-Token': token },
                body: cuerpo ? JSON.stringify(cuerpo) : undefined
            });
            if (res.status === 401) { localStorage.removeItem('adminToken'); }
            if (!res.ok) {
                let msg = res.statusText;
                try { msg = (await res.json()).detail || msg; } catch (e) {}
                alert('Error: ' + msg);
                return null;
            }
            return await res.json();
        }

        async function cargarProcedencia(nodoId) {
            const cont = document.getElementById('contenedorProcedencia');
            if (!cont) return;
            try {
                const res = await fetch(`/api/procedencia/${encodeURIComponent(nodoId)}`);
                const data = await res.json();
                renderizarProcedencia(nodoId, data);
            } catch (e) {
                cont.innerHTML = '<div class="text-xs text-rose-300">No se pudo cargar la procedencia.</div>';
            }
        }

        function renderizarProcedencia(nodoId, data) {
            const cont = document.getElementById('contenedorProcedencia');
            if (!cont) return;
            const idJs = escaparComillas(nodoId);
            const canonUrl = urlSegura(data.canon && data.canon.fuente_url);
            const afirmaciones = data.afirmaciones || [];

            const tarjetas = afirmaciones.map(a => {
                const cz = CERTEZA_UI[a.certeza] || CERTEZA_UI.teoria;
                const url = urlSegura(a.fuente_url);
                const refutada = a.estado === 'refutada';
                const idA = escaparComillas(a.id);
                return `
                <div class="p-3 rounded-lg bg-[#181d2b] border border-gi-gold/25 space-y-1.5 ${refutada ? 'opacity-60' : ''}">
                    <div class="flex flex-wrap items-center gap-1.5 text-[10px]">
                        <span class="px-2 py-0.5 rounded-full border font-bold ${cz.clase}">${cz.etiqueta}</span>
                        <span class="px-2 py-0.5 rounded-full bg-[#243242] text-[#9fd6e3] border border-[#9fd6e3]/30">${escaparHtml(TIPO_FUENTE_UI[a.tipo_fuente] || a.tipo_fuente)}</span>
                        <span class="px-2 py-0.5 rounded-full bg-[#2e2a22] text-gi-goldBright border border-gi-gold/30">${escaparHtml(ESTADO_UI[a.estado] || a.estado)}</span>
                        ${a.version_juego ? `<span class="text-[#9aa3b8]">v${escaparHtml(a.version_juego)}</span>` : ''}
                        <span class="text-[#7c8599] ml-auto">${escaparHtml((a.fecha_captura || '').slice(0, 10))}</span>
                    </div>
                    <div class="text-xs text-gi-cream leading-relaxed ${refutada ? 'line-through' : ''}">${escaparHtml(a.texto)}</div>
                    ${a.nota ? `<div class="text-[11px] text-[#b6afa0] italic">${escaparHtml(a.nota)}</div>` : ''}
                    <div class="flex flex-wrap items-center gap-2 text-[11px]">
                        ${url ? `<a href="${escaparHtml(url)}" target="_blank" rel="noopener noreferrer" class="text-[#9fd6e3] hover:underline">${escaparHtml(a.fuente_titulo || 'Fuente')}${a.fuente_ref ? ' · ' + escaparHtml(a.fuente_ref) : ''} ↗</a>` : `<span class="text-[#9aa3b8]">${escaparHtml(a.fuente_titulo || 'Sin enlace')}</span>`}
                        ${data.edicion_habilitada ? `
                        <span class="ml-auto flex gap-1.5">
                            <select onchange="cambiarCampoAfirmacion('${idJs}', '${idA}', 'certeza', this.value)" class="bg-[#131722] border border-gi-gold/30 rounded px-1 py-0.5 text-[10px] text-gi-cream">
                                ${Object.keys(CERTEZA_UI).map(k => `<option value="${k}" ${k === a.certeza ? 'selected' : ''}>${CERTEZA_UI[k].etiqueta}</option>`).join('')}
                            </select>
                            <select onchange="cambiarCampoAfirmacion('${idJs}', '${idA}', 'estado', this.value)" class="bg-[#131722] border border-gi-gold/30 rounded px-1 py-0.5 text-[10px] text-gi-cream">
                                ${Object.keys(ESTADO_UI).map(k => `<option value="${k}" ${k === a.estado ? 'selected' : ''}>${ESTADO_UI[k]}</option>`).join('')}
                            </select>
                            <button onclick="borrarAfirmacion('${idJs}', '${idA}')" class="px-2 py-0.5 rounded bg-[#351f26] border border-rose-400/40 text-rose-200 text-[10px]">Borrar</button>
                        </span>` : ''}
                    </div>
                </div>`;
            }).join('');

            const formulario = data.edicion_habilitada ? `
                <details class="rounded-xl border border-gi-gold/35 bg-[#181d2b]/90 overflow-hidden">
                    <summary class="px-3.5 py-2 cursor-pointer select-none list-none text-[11px] font-bold uppercase tracking-wider text-gi-goldBright hover:bg-[#1f2638]">+ Añadir afirmación</summary>
                    <div class="p-3 border-t border-gi-gold/20 space-y-2 text-xs">
                        <textarea id="procTexto" rows="3" placeholder="Afirmación o dato..." class="w-full bg-[#131722] border border-gi-gold/30 rounded p-2 text-gi-cream"></textarea>
                        <div class="grid grid-cols-2 gap-2">
                            <select id="procCerteza" class="bg-[#131722] border border-gi-gold/30 rounded p-1 text-gi-cream">
                                ${Object.keys(CERTEZA_UI).map(k => `<option value="${k}" ${k === 'teoria' ? 'selected' : ''}>${CERTEZA_UI[k].etiqueta}</option>`).join('')}
                            </select>
                            <select id="procTipo" class="bg-[#131722] border border-gi-gold/30 rounded p-1 text-gi-cream">
                                ${Object.keys(TIPO_FUENTE_UI).map(k => `<option value="${k}">${TIPO_FUENTE_UI[k]}</option>`).join('')}
                            </select>
                            <input id="procTitulo" placeholder="Título de la fuente" class="bg-[#131722] border border-gi-gold/30 rounded p-1 text-gi-cream">
                            <input id="procUrl" placeholder="URL (https://...)" class="bg-[#131722] border border-gi-gold/30 rounded p-1 text-gi-cream">
                            <input id="procRef" placeholder="Minuto / hilo (ej. 12:34)" class="bg-[#131722] border border-gi-gold/30 rounded p-1 text-gi-cream">
                            <input id="procVersion" placeholder="Versión del juego (ej. 5.0)" class="bg-[#131722] border border-gi-gold/30 rounded p-1 text-gi-cream">
                        </div>
                        <input id="procNota" placeholder="Nota (opcional)" class="w-full bg-[#131722] border border-gi-gold/30 rounded p-1 text-gi-cream">
                        <button onclick="guardarAfirmacion('${idJs}')" class="gi-btn-cream px-3 py-1.5 text-[11px]">Guardar</button>
                    </div>
                </details>` : '<div class="text-[10px] text-[#7c8599] italic">Edición deshabilitada (define ADMIN_TOKEN en el servidor para añadir afirmaciones).</div>';

            cont.innerHTML = `
                <div class="p-3 rounded-lg bg-[#181d2b] border border-emerald-400/30 space-y-1">
                    <div class="flex items-center gap-1.5 text-[10px]">
                        <span class="px-2 py-0.5 rounded-full border font-bold ${CERTEZA_UI.canon.clase}">Canon</span>
                        <span class="text-[#9aa3b8]">Texto de la Wiki, mostrado en las demás secciones</span>
                    </div>
                    ${canonUrl ? `<a href="${escaparHtml(canonUrl)}" target="_blank" rel="noopener noreferrer" class="text-[11px] text-[#9fd6e3] hover:underline">${escaparHtml(data.canon.fuente_titulo)} ↗</a>` : ''}
                </div>
                <div class="text-[11px] font-bold uppercase tracking-wider text-gi-gold">Fuentes externas (${afirmaciones.length})</div>
                ${tarjetas || '<div class="text-xs text-[#7c8599] italic">Sin afirmaciones externas registradas.</div>'}
                ${formulario}
                <div id="contenedorTextosJuego" class="space-y-2 pt-1"></div>
            `;
            cargarTextosJuego(nodoId);
        }

        async function cargarTextosJuego(nodoId) {
            const cont = document.getElementById('contenedorTextosJuego');
            if (!cont) return;
            try {
                const d = await (await fetch(`/api/textos_juego/${encodeURIComponent(nodoId)}`)).json();
                const textos = d.textos || [];
                if (!textos.length) { cont.innerHTML = ''; return; }
                const TIPO = { libro: '📚 Libro', arma: '⚔️ Arma', artefacto: '🛡️ Artefacto' };
                cont.innerHTML = `
                    <div class="text-[11px] font-bold uppercase tracking-wider text-gi-gold">Textos del juego · canon (${textos.length})</div>
                    ${textos.map(t => `
                        <details class="rounded-lg bg-[#181d2b] border border-emerald-400/25">
                            <summary class="px-3 py-2 cursor-pointer select-none list-none text-xs text-gi-cream flex flex-wrap items-center gap-1.5">
                                <span class="px-2 py-0.5 rounded-full border font-bold text-[10px] ${CERTEZA_UI.canon.clase}">Canon</span>
                                <span class="text-[10px] text-[#9fd6e3]">${TIPO[t.tipo] || escaparHtml(t.tipo)}</span>
                                <span class="font-semibold">${escaparHtml(t.titulo)}</span>
                                ${t.relacion === 'mencion' ? '<span class="text-[10px] text-[#9aa3b8]">· la menciona (detección automática)</span>' : ''}
                                ${t.version_juego ? `<span class="text-[10px] text-[#9aa3b8] ml-auto">v${escaparHtml(t.version_juego)}</span>` : ''}
                            </summary>
                            <div class="px-3 py-2.5 border-t border-emerald-400/20 text-xs text-gi-cream/90 leading-relaxed whitespace-pre-line max-h-72 overflow-y-auto">${escaparHtml(t.texto)}</div>
                            <div class="px-3 pb-2 text-[10px] text-[#7c8599]">Fuente: Project Amber (gi.yatta.moe)${t.obra && t.obra !== t.titulo ? ' · ' + escaparHtml(t.obra) : ''} · texto original en inglés</div>
                        </details>`).join('')}`;
            } catch (e) {
                cont.innerHTML = '';
            }
        }

        async function guardarAfirmacion(nodoId) {
            const v = id => (document.getElementById(id)?.value || '').trim();
            const cuerpo = {
                texto: v('procTexto'), certeza: v('procCerteza'), tipo_fuente: v('procTipo'),
                fuente_titulo: v('procTitulo'), fuente_url: v('procUrl'), fuente_ref: v('procRef'),
                version_juego: v('procVersion'), nota: v('procNota')
            };
            const r = await llamadaProcedencia(`/api/procedencia/${encodeURIComponent(nodoId)}`, 'POST', cuerpo);
            if (r) cargarProcedencia(nodoId);
        }

        async function cambiarCampoAfirmacion(nodoId, idA, campo, valor) {
            const r = await llamadaProcedencia(`/api/afirmacion/${encodeURIComponent(idA)}`, 'PATCH', { [campo]: valor });
            if (r) cargarProcedencia(nodoId);
        }

        async function borrarAfirmacion(nodoId, idA) {
            if (!confirm('¿Borrar esta afirmación?')) return;
            const r = await llamadaProcedencia(`/api/afirmacion/${encodeURIComponent(idA)}`, 'DELETE');
            if (r) cargarProcedencia(nodoId);
        }

        // ---------- Etimología y Mundo Real ----------
        async function cargarEtimologia(nodoId) {
            const cont = document.getElementById('contenedorEtimologia');
            if (!cont) return;
            try {
                const res = await fetch(`/api/etimologia/${encodeURIComponent(nodoId)}`);
                renderizarEtimologia(nodoId, await res.json(), false);
            } catch (e) {
                cont.innerHTML = '<div class="text-xs text-rose-300">No se pudo cargar la etimología.</div>';
            }
        }

        function filaEtim(etiqueta, valor) {
            if (!valor) return '';
            return `<div class="p-2.5 rounded-lg bg-[#181d2b] border border-gi-gold/25">
                <div class="text-[10px] font-bold uppercase tracking-wider text-gi-gold mb-0.5">${etiqueta}</div>
                <div class="text-xs text-gi-cream leading-relaxed whitespace-pre-line">${escaparHtml(valor)}</div>
            </div>`;
        }

        function renderizarEtimologia(nodoId, data, editando) {
            const cont = document.getElementById('contenedorEtimologia');
            if (!cont) return;
            cacheEtimologia[nodoId] = data;
            const f = data.ficha;
            const idJs = escaparComillas(nodoId);
            const botones = data.edicion_habilitada ? `
                <div class="flex gap-2">
                    <button onclick="renderizarEtimologia('${idJs}', cacheEtimologia['${idJs}'], ${!editando})" class="gi-btn-dark px-3 py-1 text-[11px]">${editando ? 'Cancelar' : (f ? 'Editar' : 'Crear ficha')}</button>
                    ${f && !editando ? `<button onclick="borrarEtimologia('${idJs}')" class="px-3 py-1 rounded bg-[#351f26] border border-rose-400/40 text-rose-200 text-[11px]">Borrar</button>` : ''}
                </div>` : '';

            if (editando) {
                const v = k => escaparHtml((f && f[k]) || '');
                const refs = ((f && f.referencias) || []).map(r => (r.titulo || '') + ' | ' + (r.url || '')).join('\n');
                const estilo = 'w-full bg-[#131722] border border-gi-gold/30 rounded p-1.5 text-xs text-gi-cream';
                cont.innerHTML = `
                    ${botones}
                    <div class="p-2.5 rounded-lg bg-[#181d2b] border border-[#9fd6e3]/30 space-y-1.5 text-xs">
                        <div class="text-[10px] font-bold uppercase tracking-wider text-[#9fd6e3]">🌐 Importar resumen desde Wikipedia (se guarda en la BBDD)</div>
                        <div class="flex gap-2">
                            <input id="etWikiTerm" placeholder="Término en Wikipedia (ej. Icor, Paimon, Ars Goetia)" value="${v('wikipedia_titulo') || escaparHtml(nodoId)}" class="flex-1 bg-[#131722] border border-gi-gold/30 rounded p-1.5 text-xs text-gi-cream">
                            <button onclick="buscarWikipediaParaFicha('${idJs}')" class="gi-btn-dark px-3 py-1 text-[11px] shrink-0">Buscar y guardar</button>
                        </div>
                        <textarea id="etWikiResumen" rows="3" placeholder="Resumen de Wikipedia guardado en Neo4j..." class="${estilo}">${v('wikipedia_resumen')}</textarea>
                        <input type="hidden" id="etWikiTitulo" value="${v('wikipedia_titulo')}">
                        <input type="hidden" id="etWikiUrl" value="${v('wikipedia_url')}">
                        <input type="hidden" id="etWikiIdioma" value="${v('wikipedia_idioma')}">
                    </div>
                    <div class="space-y-2 text-xs">
                        <textarea id="etOrigen" rows="3" placeholder="Origen del nombre (ej. Paimon viene de...)" class="${estilo}">${v('origen_nombre')}</textarea>
                        <div class="grid grid-cols-2 gap-2">
                            <input id="etIdioma" placeholder="Idioma de origen" value="${v('idioma_origen')}" class="${estilo}">
                            <input id="etCultura" placeholder="Cultura / mitología" value="${v('cultura')}" class="${estilo}">
                        </div>
                        <textarea id="etRef" rows="2" placeholder="Referencia del mundo real (mito, obra, figura...)" class="${estilo}">${v('referencia_real')}</textarea>
                        <textarea id="etResumen" rows="4" placeholder="Resumen de la Wiki de Genshin" class="${estilo}">${v('resumen_wiki')}</textarea>
                        <label class="flex items-center gap-1.5 text-[11px] text-[#9aa3b8]"><input type="checkbox" id="etResumenAuto" ${(f && f.resumen_auto) ? 'checked' : ''}> Resumen generado automáticamente (sin revisar)</label>
                        <button onclick="generarBorradorResumen('${idJs}')" class="gi-btn-dark px-3 py-1 text-[11px]">✨ Generar borrador del resumen con Gemini</button>
                        <textarea id="etRefs" rows="3" placeholder="Referencias, una por línea: Título | https://url" class="${estilo}">${escaparHtml(refs)}</textarea>
                        <button onclick="guardarEtimologia('${idJs}')" class="gi-btn-cream px-3 py-1.5 text-[11px]">Guardar</button>
                    </div>`;
                return;
            }

            if (!f) {
                cont.innerHTML = `${botones}
                    <div class="text-xs text-[#7c8599] italic">Sin ficha de etimología ni artículo coincidente en Wikipedia para esta entidad.</div>
                    <a href="${escaparHtml(urlSegura(data.wiki_url))}" target="_blank" rel="noopener noreferrer" class="text-[11px] text-[#9fd6e3] hover:underline">Ver en la Wiki ↗</a>`;
                return;
            }
            const refsHtml = (f.referencias || []).map(r => {
                const u = urlSegura(r.url);
                return u
                    ? `<li><a href="${escaparHtml(u)}" target="_blank" rel="noopener noreferrer" class="text-[#9fd6e3] hover:underline">${escaparHtml(r.titulo || u)} ↗</a></li>`
                    : `<li class="text-[#9aa3b8]">${escaparHtml(r.titulo)}</li>`;
            }).join('');
            const meta = [f.idioma_origen, f.cultura].filter(Boolean).join(' · ');
            const wikiUrlReal = urlSegura(f.wikipedia_url);
            const bloqueWikipedia = f.wikipedia_resumen ? `
                <div class="p-3 rounded-lg bg-[#162030] border border-[#9fd6e3]/40 space-y-1.5">
                    <div class="flex items-center justify-between gap-2">
                        <span class="text-[10px] font-bold uppercase tracking-wider text-[#9fd6e3]">🌐 Wikipedia (${escaparHtml((f.wikipedia_idioma || 'es').toUpperCase())}) · ${escaparHtml(f.wikipedia_titulo || '')}</span>
                        ${wikiUrlReal ? `<a href="${escaparHtml(wikiUrlReal)}" target="_blank" rel="noopener noreferrer" class="text-[10px] text-[#9fd6e3] hover:underline">Leer en Wikipedia ↗</a>` : ''}
                    </div>
                    <div class="text-xs text-gi-cream/95 leading-relaxed whitespace-pre-line">${escaparHtml(f.wikipedia_resumen)}</div>
                </div>` : '';
            cont.innerHTML = `
                ${botones}
                ${bloqueWikipedia}
                ${filaEtim('Origen del nombre' + (meta ? ' (' + escaparHtml(meta) + ')' : ''), f.origen_nombre)}
                ${filaEtim('Referencia del mundo real', f.referencia_real)}
                ${filaEtim('Resumen de la Wiki de Genshin' + (f.resumen_auto ? ' · automático, sin revisar' : ''), f.resumen_wiki)}
                ${refsHtml ? `<div class="p-2.5 rounded-lg bg-[#181d2b] border border-gi-gold/25"><div class="text-[10px] font-bold uppercase tracking-wider text-gi-gold mb-1">Referencias</div><ul class="text-xs space-y-0.5 list-disc ml-4">${refsHtml}</ul></div>` : ''}
                <a href="${escaparHtml(urlSegura(data.wiki_url))}" target="_blank" rel="noopener noreferrer" class="text-[11px] text-[#9fd6e3] hover:underline">Ver en la Wiki ↗</a>`;
        }

        async function buscarWikipediaParaFicha(nodoId) {
            const termino = (document.getElementById('etWikiTerm')?.value || nodoId).trim();
            const r = await llamadaProcedencia(`/api/etimologia_wikipedia/${encodeURIComponent(nodoId)}`, 'POST', { termino });
            if (!r) return;
            if (document.getElementById('etWikiResumen')) document.getElementById('etWikiResumen').value = r.wikipedia_resumen || '';
            if (document.getElementById('etWikiTitulo')) document.getElementById('etWikiTitulo').value = r.wikipedia_titulo || '';
            if (document.getElementById('etWikiUrl')) document.getElementById('etWikiUrl').value = r.wikipedia_url || '';
            if (document.getElementById('etWikiIdioma')) document.getElementById('etWikiIdioma').value = r.wikipedia_idioma || '';
            if (document.getElementById('etWikiTerm') && r.wikipedia_titulo) document.getElementById('etWikiTerm').value = r.wikipedia_titulo;
            cacheEtimologia[nodoId] = { ...(cacheEtimologia[nodoId] || {}), ficha: r };
        }

        async function guardarEtimologia(nodoId) {
            const v = id => (document.getElementById(id)?.value || '').trim();
            const referencias = v('etRefs').split(String.fromCharCode(10)).map(l => l.trim()).filter(Boolean).map(l => {
                const i = l.lastIndexOf('|');
                return i >= 0 ? { titulo: l.slice(0, i).trim(), url: l.slice(i + 1).trim() } : { titulo: l, url: '' };
            });
            const cuerpo = {
                origen_nombre: v('etOrigen'), idioma_origen: v('etIdioma'), cultura: v('etCultura'),
                referencia_real: v('etRef'), resumen_wiki: v('etResumen'),
                wikipedia_titulo: v('etWikiTitulo'), wikipedia_resumen: v('etWikiResumen'),
                wikipedia_url: v('etWikiUrl'), wikipedia_idioma: v('etWikiIdioma'),
                resumen_auto: Boolean(document.getElementById('etResumenAuto')?.checked), referencias
            };
            const r = await llamadaProcedencia(`/api/etimologia/${encodeURIComponent(nodoId)}`, 'PUT', cuerpo);
            if (r) cargarEtimologia(nodoId);
        }

        async function borrarEtimologia(nodoId) {
            if (!confirm('¿Borrar la ficha de etimología de esta entidad?')) return;
            const r = await llamadaProcedencia(`/api/etimologia/${encodeURIComponent(nodoId)}`, 'DELETE');
            if (r) cargarEtimologia(nodoId);
        }

        async function generarBorradorResumen(nodoId) {
            const r = await llamadaProcedencia(`/api/etimologia_borrador/${encodeURIComponent(nodoId)}`, 'POST');
            if (!r) return;
            document.getElementById('etResumen').value = r.resumen_wiki;
            document.getElementById('etResumenAuto').checked = true;
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
