from collections import deque
import hashlib
import os
import re
import sys
import time
import warnings
import requests
from neo4j import GraphDatabase
from fastembed import TextEmbedding

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

# Asegurar salida UTF-8 en consolas de Windows para evitar errores con emojis
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Cargar variables desde archivo .env local si existe (para no exponer claves en el código público)
_RUTA_ENV = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_RUTA_ENV):
    with open(_RUTA_ENV, "r", encoding="utf-8") as _f_env:
        for _linea in _f_env:
            _l = _linea.strip()
            if _l and not _l.startswith("#") and "=" in _l:
                _k, _v = _l.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# Configuración para la Wiki en Inglés (https://genshin-impact.fandom.com/wiki/Lore)
WIKI_API_URL = "https://genshin-impact.fandom.com/api.php"
WIKI_BASE_URL = "https://genshin-impact.fandom.com/wiki/"
HEADERS = {"User-Agent": "Genshin-LoreCrawler-EN/1.0"}

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://90f31cb2.databases.neo4j.io").strip()
NEO4J_USER = os.getenv("NEO4J_USER", "90f31cb2").strip()
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "").strip()

# Configuración de Embeddings locales 100% gratuitos y multilingües (FastEmbed ONNX, 384 dims, sin límites de cuota)
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIMENSIONS = 384
_MODELO_EMBEDDINGS: TextEmbedding | None = None


def obtener_modelo_embeddings() -> TextEmbedding:
    global _MODELO_EMBEDDINGS
    if _MODELO_EMBEDDINGS is None:
        _MODELO_EMBEDDINGS = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
    return _MODELO_EMBEDDINGS

# 1. HUBS: Índices de descubrimiento (se leen sus enlaces y párrafos, pero NO se crean como nodo :Entity)
HUBS_INDICE = {
    "lore", "timeline", "history", "books", "book", "archive"
}

# 2. SECCIONES EXCLUSIVAMENTE DE MECÁNICAS NUMÉRICAS / META A IGNORAR
# Conservamos todas las secciones útiles para lore y enciclopedia (Profile, Story, History, Trivia, Etymology, Notes, Character Stories, etc.)
SECCIONES_IGNORADAS = {
    "combat info", "stats", "ascensions and stats", "ascensions", "talents",
    "constellations", "attributes", "stat scaling", "scaling", "abilities",
    "mechanics", "strategy", "rewards", "possible rewards", "drops",
    "how to obtain", "availability", "shop", "recipe", "crafting",
    "character ascension", "talent leveling", "gameplay notes", "gameplay info",
    "talent upgrade", "character event wishes", "chronicled wishes",
    "change history", "other languages", "navigation", "gallery", "animations",
    "references", "voice-overs", "mail", "outfit", "outfits", "namecard",
    "videos", "paimon's paintings", "genius invokation tcg"
}

# 3. PLANTILLAS DE COMBATE/INTERFAZ A ELIMINAR ANTES DE LEER
PLANTILLAS_IGNORADAS_REGEX = (
    r"\{\{\s*(?:Character Ascensions and Stats|Talent|Constellation|Enemy Stats|"
    r"Weapon Stats|Change History|Other Languages|Navigation|Navbox|"
    r"Voice-Overs|Dialogue|Shop|Recipe|Drop|Character Tabs)[^}]*\}\}"
)

# 4. NO-ENTIDADES (Conceptos de mecánicas del juego que nunca deben ser nodos)
NO_ENTIDADES = {
    "genshin impact", "hoyolab", "hoyoverse", "mihoyo", "characters", "character",
    "npc", "npcs", "quests", "quest", "archon quest", "archon quests",
    "world quest", "world quests", "story quest", "story quests", "events", "event",
    "weapons", "weapon", "artifacts", "artifact", "elements", "element",
    "achievements", "achievement", "domains", "domain", "enemies", "enemy",
    "bosses", "boss", "wishes", "wish", "primogem", "primogems", "mora",
    "adventure rank", "original resin", "hero's wit", "atk", "def", "hp",
    "elemental mastery", "energy recharge", "crit rate", "crit dmg",
    "normal attack", "elemental skill", "elemental burst", "traveler"
}

CACHE_CANONICO: dict[str, str] = {}
http_session = requests.Session()
http_session.headers.update(HEADERS)


def normalizar_entidad(nombre_bruto: str) -> str:
    """Limpia anclas (#) y subpáginas (/Lore, /Storyline, /Voice-Overs) para quedarse con el ID real."""
    if not nombre_bruto:
        return ""
    base = nombre_bruto.split("#")[0].strip().lstrip(":").strip().replace("_", " ")
    if "/" in base:
        raiz, sub = base.split("/", 1)
        if sub.lower().startswith(("lore", "storyline", "history", "profile", "voice-overs", "gallery", "outfits", "combat", "companion")):
            base = raiz.strip()
        else:
            return ""
    if not base:
        return ""
    return base[0].upper() + base[1:]


def es_entidad_valida(nombre: str) -> bool:
    if not nombre or len(nombre) < 2:
        return False
    lower = nombre.strip().lstrip(":").strip().lower()
    prefijos = ("file:", "category:", "special:", "template:", "user:", "help:", "talk:", "module:", "media:", "project:", "fandom:")
    if nombre.strip().startswith(":") or lower.startswith(prefijos) or lower in NO_ENTIDADES or lower in HUBS_INDICE:
        return False
    if re.match(r"^[a-z]{2,3}(?:-[a-z]+)?:", lower):
        return False
    return True


def resolver_redirecciones_wiki(nombres: list[str]) -> dict[str, str]:
    """
    Resuelve en lotes de 50 contra la API de MediaWiki las redirecciones (redirects=1)
    para que alias como 'Buer', 'Lesser Lord Kusanali' y 'Nahida' apunten todos a 'Nahida'.
    """
    pendientes = []
    vistos = set()
    for n in nombres:
        limpio = normalizar_entidad(n)
        if not es_entidad_valida(limpio):
            continue
        k = limpio.lower()
        if k not in CACHE_CANONICO and k not in vistos:
            vistos.add(k)
            pendientes.append(limpio)

    for i in range(0, len(pendientes), 50):
        lote = pendientes[i : i + 50]
        try:
            res = http_session.get(
                WIKI_API_URL,
                params={
                    "action": "query",
                    "format": "json",
                    "titles": "|".join(lote),
                    "redirects": 1,
                },
                timeout=10,
            )
            if res.status_code != 200:
                for orig in lote:
                    CACHE_CANONICO[orig.lower()] = orig
                continue

            qdata = res.json().get("query", {})
            alias_map: dict[str, str] = {}
            for item in qdata.get("normalized", []):
                alias_map[item["from"].lower()] = item["to"]
            for item in qdata.get("redirects", []):
                origen_l = item["from"].lower()
                destino_t = item["to"]
                alias_map[origen_l] = destino_t
                for k_orig, v_dest in list(alias_map.items()):
                    if v_dest.lower() == origen_l:
                        alias_map[k_orig] = destino_t

            titulos_exactos: dict[str, str] = {}
            for page in qdata.get("pages", {}).values():
                t = page.get("title", "")
                if t:
                    titulos_exactos[t.lower()] = t

            for orig in lote:
                k = orig.lower()
                resuelto = alias_map.get(k, titulos_exactos.get(k, orig))
                canonico = normalizar_entidad(resuelto)
                CACHE_CANONICO[k] = canonico if es_entidad_valida(canonico) else ""
        except Exception as e:
            print(f"⚠️ Error resolviendo redirecciones: {e}")
            for orig in lote:
                CACHE_CANONICO[orig.lower()] = orig

    resultado = {}
    for n in nombres:
        limpio = normalizar_entidad(n)
        if not limpio:
            continue
        canon = CACHE_CANONICO.get(limpio.lower(), limpio)
        if es_entidad_valida(canon):
            resultado[n] = canon
    return resultado


def obtener_wikitexto(titulo_pagina):
    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "rvprop": "content",
        "titles": titulo_pagina,
        "redirects": 1,
    }
    try:
        res = http_session.get(WIKI_API_URL, params=params, timeout=10)
        if res.status_code == 200:
            pages = res.json().get("query", {}).get("pages", {})
            for pid, info in pages.items():
                if pid != "-1":
                    return info.get("revisions", [{}])[0].get("*", "")
    except Exception as e:
        print(f"❌ Error descargando '{titulo_pagina}': {e}")
    return ""


def obtener_wikitexto_lore(entidad):
    """
    Descarga la página principal y fusiona TANTO '{entidad}/Storyline' COMO '{entidad}/Lore'
    cuando existen, para conservar biografía completa + Character Stories 1-5 + citas oficiales.
    """
    texto_base = obtener_wikitexto(entidad)
    if not texto_base or entidad.lower() in HUBS_INDICE:
        return texto_base

    bloques = [texto_base]
    if "{{Character Infobox" in texto_base or "{{Playable Character" in texto_base or "{{NPC Infobox" in texto_base:
        for sufijo in ("Storyline", "Lore"):
            sub = obtener_wikitexto(f"{entidad}/{sufijo}")
            # Evitar duplicar si la subpágina redirigió a la misma página principal
            if sub and len(sub) > 150 and sub[:200] != texto_base[:200]:
                print(f"   📖 Subpágina '{entidad}/{sufijo}' fusionada.")
                bloques.append(f"\n\n== {sufijo} ==\n" + sub)

    return "\n".join(bloques)


def expandir_plantillas_narrativas(wikitext: str) -> str:
    """
    Convierte plantillas de contenido narrativo ({{Quote|...}}, {{Description|...}},
    {{Character Story|title1=...|text1=...}}) en texto y subsecciones legibles antes de limpiar plantillas.
    """
    if not wikitext:
        return ""

    # Eliminar comentarios HTML <!-- ... -->
    texto = re.sub(r"<!--.*?-->", "", wikitext, flags=re.DOTALL)

    # Extraer historias de {{Character Story | title1 = ... | text1 = ...}}
    def repl_cs(m):
        bloque = m.group(0)
        partes_cs = []
        for idx in range(1, 25):
            mt = re.search(rf"\|\s*title{idx}\s*=\s*(.*?)(?=\n\s*\||\n\}}\}})", bloque, flags=re.DOTALL)
            mx = re.search(rf"\|\s*(?:text|story){idx}\s*=\s*(.*?)(?=\n\s*\||\n\}}\}})", bloque, flags=re.DOTALL)
            if mx and mx.group(1).strip():
                titulo_cs = mt.group(1).strip() if (mt and mt.group(1).strip()) else f"Character Story {idx}"
                cuerpo_cs = mx.group(1).strip()
                partes_cs.append(f"\n=== {titulo_cs} ===\n{cuerpo_cs}\n")
        return "\n".join(partes_cs)

    texto = re.sub(r"\{\{Character Story\b.*?\n\}\}", repl_cs, texto, flags=re.DOTALL | re.IGNORECASE)

    # Convertir {{Quote|texto|autor}} en cita legible
    def repl_quote(m):
        cita = m.group(1).strip().replace("<br>", "\n").replace("<br/>", "\n")
        autor = m.group(2).strip() if m.group(2) else ""
        return f'\n«{cita}» — {autor}\n' if autor else f'\n«{cita}»\n'

    texto = re.sub(r"\{\{Quote\|([^|}]+)(?:\|([^|}]+))?[^}]*\}\}", repl_quote, texto, flags=re.IGNORECASE)
    texto = re.sub(r"\{\{Description\|([^|}]+)(?:\|([^|}]+))?[^}]*\}\}", repl_quote, texto, flags=re.IGNORECASE)

    return texto


def eliminar_tablas_y_gameplay(wikitext):
    texto = expandir_plantillas_narrativas(wikitext)
    texto = re.sub(r"\{\|.*?\|\}", "", texto, flags=re.DOTALL)
    texto = re.sub(PLANTILLAS_IGNORADAS_REGEX, "", texto, flags=re.IGNORECASE | re.DOTALL)
    return texto


def limpiar_texto_plano(wikitext):
    texto = re.sub(r"<ref.*?>.*?</ref>", "", wikitext, flags=re.DOTALL)
    texto = re.sub(r"<ref[^>]*/>", "", texto)
    texto = re.sub(r"</?p\s*/?>", "\n", texto, flags=re.IGNORECASE)
    texto = re.sub(r"<br\s*/?>", "\n", texto, flags=re.IGNORECASE)
    prev = None
    while prev != texto:
        prev = texto
        texto = re.sub(r"\{\{[^{}]*\}\}", "", texto, flags=re.DOTALL)
    texto = re.sub(r"\[\[[^\]|]+\|([^\]]+)\]\]", r"\1", texto)
    texto = re.sub(r"\[\[([^\]]+)\]\]", r"\1", texto)
    texto = re.sub(r"''+", "", texto)
    texto = re.sub(r"<[^>]+>", "", texto)
    texto = text_entities = texto.replace("&mdash;", "—").replace("&ndash;", "–").replace("&nbsp;", " ")
    texto = re.sub(r"^[\*\#\:\;]+\s*", "• ", texto.strip(), flags=re.MULTILINE)
    return texto.strip()


def extraer_bloque_infobox(wikitext_bruto: str) -> str:
    """Extrae las líneas |clave = valor de la Infobox inicial para conservar todos los metadatos técnicos reales (sin idiomas)."""
    cabecera = re.split(r"(?m)^==+|\{\{\s*Other\s+Languages", wikitext_bruto[:4500], maxsplit=1, flags=re.IGNORECASE)[0]
    prefijos_excluidos = (
        "image", "caption", "voice", "obtain", "release", "en", "zh", "zhs", "zht",
        "chs", "cht", "ja", "jp", "ko", "kr", "es", "fr", "ru", "th", "vi", "vn",
        "de", "id", "pt", "tr", "it", "tl", "rm", "lit",
    )
    lineas_info = []
    for m in re.finditer(r"^\s*\|\s*([a-zA-Z0-9_]+)\s*=\s*(.+)$", cabecera, flags=re.MULTILINE):
        k, v = m.group(1).strip(), m.group(2).strip()
        k_low = re.sub(r"\d+$", "", k.lower()).strip("_")
        if (
            v
            and "note" not in k_low
            and not k_low.startswith(prefijos_excluidos)
            and not k_low.endswith(("_rm", "_tl", "_lit", "_note", "_ref"))
        ):
            val_limpio = limpiar_texto_plano(v)
            if val_limpio:
                lineas_info.append(f"|{k} = {val_limpio}")
    if lineas_info:
        return "{{Infobox\n" + "\n".join(lineas_info) + "\n}}\n\n"
    return ""


def extraer_entidades_brutas_de_fragmento(texto):
    matches = re.findall(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", texto)
    entidades = set()
    for m in matches:
        ent = normalizar_entidad(m)
        if es_entidad_valida(ent):
            entidades.add(ent)
    return sorted(list(entidades))


def formatear_cita_ref(ref_raw):
    """Convierte plantillas de referencia de la wiki en inglés ({{Ref/Quest|...}}, {{Ref/CS|...}}) en texto legible."""
    m_cs = re.search(r"\{\{Ref/CS\|([^|}]+)\|(\d+)", ref_raw, flags=re.IGNORECASE)
    if m_cs:
        pj, num = m_cs.group(1).strip(), m_cs.group(2).strip()
        return f"{pj}'s Character Story {num}", f"{WIKI_BASE_URL}{pj.replace(' ', '_')}/Lore#Character_Story_{num}"

    m_q = re.search(r"\{\{Ref/Quest\|([^|}]+)", ref_raw, flags=re.IGNORECASE)
    if m_q:
        quest = m_q.group(1).strip()
        return f"Quest: {quest}", f"{WIKI_BASE_URL}{quest.replace(' ', '_')}"

    m_book = re.search(r"\{\{Ref/Book\|([^|}]+)", ref_raw, flags=re.IGNORECASE)
    if m_book:
        book = m_book.group(1).strip()
        return f"Book: {book}", f"{WIKI_BASE_URL}{book.replace(' ', '_')}"

    m_vo = re.search(r"\{\{Ref/VO\|([^|}]+)\|([^|}]+)", ref_raw, flags=re.IGNORECASE)
    if m_vo:
        pj, vo = m_vo.group(1).strip(), m_vo.group(2).strip()
        return f"Voice-Over: {pj} ({vo})", f"{WIKI_BASE_URL}{pj.replace(' ', '_')}/Voice-Overs"

    m_item = re.search(r"\{\{Ref/(?:Item|Artifact|Weapon|GCG)\|([^|}]+)", ref_raw, flags=re.IGNORECASE)
    if m_item:
        item = m_item.group(1).strip()
        return f"Item: {item}", f"{WIKI_BASE_URL}{item.replace(' ', '_')}"

    m_link = re.search(r"\[\[([^\]|#]+)(?:#([^\]|]+))?", ref_raw)
    url = ""
    if m_link:
        pag = m_link.group(1).strip()
        sec = f"#{m_link.group(2).strip()}" if m_link.group(2) else ""
        url = f"{WIKI_BASE_URL}{(pag + sec).replace(' ', '_')}"

    limpio = limpiar_texto_plano(ref_raw)
    return (limpio or "Wiki Reference"), url


def procesar_pagina(titulo_pagina, wikitext_bruto):
    es_hub_actual = titulo_pagina.lower() in HUBS_INDICE
    enlaces_semilla_hub = extraer_entidades_brutas_de_fragmento(wikitext_bruto) if es_hub_actual else []

    cabecera_infobox = extraer_bloque_infobox(wikitext_bruto)
    wikitext = eliminar_tablas_y_gameplay(wikitext_bruto)

    mapa_refs = {
        m.group(1).strip(): m.group(2).strip()
        for m in re.finditer(r'<ref\s+name=["\']?([^"\'>\s]+)["\']?\s*>(.*?)</ref>', wikitext, flags=re.DOTALL)
    }

    partes = re.split(r"(==+)\s*(.*?)\s*\1", wikitext)
    secciones = [("Overview", partes[0])] if partes else []
    i = 1
    while i + 2 < len(partes):
        sec_nombre = partes[i + 1].strip()
        if sec_nombre.lower() not in SECCIONES_IGNORADAS:
            secciones.append((sec_nombre, partes[i + 2]))
        i += 3

    # Construir el artículo completo con TODAS las secciones de la página para la enciclopedia
    secciones_enciclopedia = []
    lineas_procesables = []
    todas_entidades_brutas = set(enlaces_semilla_hub)

    for epoca_sec, contenido_sec in secciones:
        # Limpiar plantillas multilínea sobre la sección entera para no dejar '{{' colgados
        contenido_sec_limpio = limpiar_texto_plano(contenido_sec)
        parrafos_seccion = []
        for parrafo in contenido_sec_limpio.split("\n"):
            p_strip = parrafo.strip()
            if not p_strip or len(p_strip) < 3 or p_strip.startswith(("{|", "|", "!", "{{", "}}")):
                continue
            if re.search(r"\b(CRIT Rate|CRIT DMG|DMG Bonus|Normal Attack|Elemental Burst|Lv\. \d+)\b", p_strip, flags=re.IGNORECASE):
                continue
            parrafos_seccion.append(p_strip)

        if parrafos_seccion:
            cuerpo_sec = "\n\n".join(parrafos_seccion)
            if epoca_sec == "Overview":
                secciones_enciclopedia.append(cuerpo_sec)
            else:
                secciones_enciclopedia.append(f"== {epoca_sec} ==\n{cuerpo_sec}")

        # Extraer relaciones línea por línea sobre el wikitext con enlaces [[...]] y <ref>
        for linea in contenido_sec.split("\n"):
            linea = linea.strip()
            if not linea or len(linea) < 20 or linea.startswith(("{|", "|", "!", "{{", "}}")):
                continue
            if re.search(r"\b(CRIT Rate|CRIT DMG|DMG Bonus|Normal Attack|Elemental Burst|Lv\. \d+)\b", linea, flags=re.IGNORECASE):
                continue
            entidades_brutas_linea = extraer_entidades_brutas_de_fragmento(linea)
            parrafo_limpio = limpiar_texto_plano(linea)
            if parrafo_limpio and entidades_brutas_linea:
                todas_entidades_brutas.update(entidades_brutas_linea)
                lineas_procesables.append((epoca_sec, linea, parrafo_limpio, entidades_brutas_linea))

    texto_web_entidad = cabecera_infobox + "\n\n".join(secciones_enciclopedia)

    # Resolver en lote todas las redirecciones de esta página
    mapa_canon = resolver_redirecciones_wiki(sorted(list(todas_entidades_brutas)))

    vecinos_descubiertos = set()
    for e_bruta in todas_entidades_brutas:
        if e_bruta in mapa_canon:
            vecinos_descubiertos.add(mapa_canon[e_bruta])

    relaciones = []
    rel_ids_vistos = set()

    for epoca_sec, linea, parrafo_limpio, entidades_brutas_linea in lineas_procesables:
        entidades_canonicas = sorted(list({
            mapa_canon[eb] for eb in entidades_brutas_linea if eb in mapa_canon
        }))
        if not entidades_canonicas:
            continue

        refs_raw = re.findall(r"<ref(?:\s+[^>]*)?>(.*?)</ref>", linea, flags=re.DOTALL)
        for nombre_reuso in re.findall(r'<ref\s+name=["\']?([^"\'>\s/]+)["\']?\s*/>', linea):
            if nombre_reuso in mapa_refs:
                refs_raw.append(mapa_refs[nombre_reuso])

        if refs_raw:
            ref_titulo, ref_url = formatear_cita_ref(refs_raw[0])
            ref_texto = limpiar_texto_plano(refs_raw[0]) or ref_titulo
        else:
            ref_titulo = f"{titulo_pagina} ({epoca_sec})"
            ref_url = f"{WIKI_BASE_URL}{titulo_pagina.replace(' ', '_')}"
            ref_texto = parrafo_limpio

        hash_parrafo = hashlib.md5(parrafo_limpio.encode("utf-8")).hexdigest()[:8]

        if not es_hub_actual:
            for destino in entidades_canonicas:
                if destino.lower() != titulo_pagina.lower():
                    rel_id = f"{titulo_pagina}->{destino}#{epoca_sec}#{hash_parrafo}"
                    if rel_id not in rel_ids_vistos:
                        rel_ids_vistos.add(rel_id)
                        relaciones.append({
                            "rel_id": rel_id,
                            "origen": titulo_pagina,
                            "destino": destino,
                            "epoca": epoca_sec,
                            "parrafo": parrafo_limpio,
                            "ref_titulo": ref_titulo[:120],
                            "ref_url": ref_url or f"{WIKI_BASE_URL}{titulo_pagina.replace(' ', '_')}",
                            "ref_texto": ref_texto
                        })
        else:
            for idx_a in range(len(entidades_canonicas)):
                for idx_b in range(idx_a + 1, len(entidades_canonicas)):
                    orig, dest = entidades_canonicas[idx_a], entidades_canonicas[idx_b]
                    if orig.lower() != dest.lower():
                        rel_id = f"{titulo_pagina}#{orig}->{dest}#{epoca_sec}#{hash_parrafo}"
                        if rel_id not in rel_ids_vistos:
                            rel_ids_vistos.add(rel_id)
                            relaciones.append({
                                "rel_id": rel_id,
                                "origen": orig,
                                "destino": dest,
                                "epoca": epoca_sec,
                                "parrafo": parrafo_limpio,
                                "ref_titulo": ref_titulo[:120],
                                "ref_url": ref_url or f"{WIKI_BASE_URL}{titulo_pagina.replace(' ', '_')}",
                                "ref_texto": ref_texto
                            })

    return texto_web_entidad, relaciones, sorted(list(vecinos_descubiertos))


def generar_chunks_jerarquicos_small_to_big(titulo_pagina: str, texto_web: str) -> list[dict]:
    """
    Estrategia de Chunking Jerárquico Small-to-Big para GraphRAG:
    - Nivel Padre (Big): Cada sección o capítulo de la página (ej. 'The Cataclysm', 'Character Story 1')
      conserva su contexto amplio en `texto_padre`.
    - Nivel Hijo (Small): Fragmentos semánticos de ~500-800 caracteres con cabecera contextual
      `[Entidad: X | Sección: Y]` sobre los que se calcula el embedding vectorial de alta precisión.
    """
    if not texto_web or not texto_web.strip():
        return []

    url_base = f"{WIKI_BASE_URL}{titulo_pagina.replace(' ', '_')}"
    texto = texto_web.strip()

    # Extraer bloque Infobox inicial si existe para crear un chunk de Ficha Técnica
    chunks_resultado = []
    orden_global = 0

    m_info = re.match(r"^\{\{Infobox\n(.*?)\n\}\}\s*", texto, flags=re.DOTALL)
    if m_info:
        bloque_info = m_info.group(1).strip()
        texto = texto[m_info.end():].strip()
        if bloque_info:
            texto_embed_info = f"[Entidad: {titulo_pagina} | Sección: Ficha Técnica / Infobox]\n{bloque_info}"
            chunk_id = f"{titulo_pagina}#chunk#{orden_global}"
            chunks_resultado.append({
                "id": chunk_id,
                "entidad_id": titulo_pagina,
                "seccion": "Infobox / Ficha",
                "orden": orden_global,
                "texto": bloque_info,
                "texto_padre": bloque_info,
                "texto_embedding": texto_embed_info[:2000],
                "url_wiki": url_base,
            })
            orden_global += 1

    # Dividir por encabezados == Sección ==
    partes = re.split(r"^==+\s*(.+?)\s*==+\s*$", texto, flags=re.MULTILINE)
    secciones = []
    if partes[0].strip():
        secciones.append(("Overview", partes[0].strip()))
    idx = 1
    while idx + 1 < len(partes):
        nom_sec = partes[idx].strip()
        cuerpo_sec = partes[idx + 1].strip()
        if cuerpo_sec:
            secciones.append((nom_sec, cuerpo_sec))
        idx += 2

    MAX_CHARS_HIJO = 650
    MAX_CHARS_PADRE = 2400

    for nom_sec, cuerpo_sec in secciones:
        parrafos = [p.strip() for p in re.split(r"\n{2,}", cuerpo_sec) if p.strip()]
        if not parrafos:
            continue

        # Texto Padre (Big): la sección completa (o ventana amplia de hasta 2400 chars)
        texto_padre_sec = "\n\n".join(parrafos)[:MAX_CHARS_PADRE]
        ancla = nom_sec.replace(" ", "_")
        url_sec = url_base if nom_sec == "Overview" else f"{url_base}#{ancla}"

        # Agrupar párrafos en Chunks Hijos (Small) de ~650 chars
        buffer_actual = []
        long_actual = 0

        for p in parrafos:
            if long_actual + len(p) > MAX_CHARS_HIJO and buffer_actual:
                texto_hijo = "\n\n".join(buffer_actual)
                texto_embed = f"[Entidad: {titulo_pagina} | Sección: {nom_sec}]\n{texto_hijo}"
                chunks_resultado.append({
                    "id": f"{titulo_pagina}#chunk#{orden_global}",
                    "entidad_id": titulo_pagina,
                    "seccion": nom_sec,
                    "orden": orden_global,
                    "texto": texto_hijo,
                    "texto_padre": texto_padre_sec,
                    "texto_embedding": texto_embed[:2000],
                    "url_wiki": url_sec,
                })
                orden_global += 1
                # Solapamiento suave: conservar el último párrafo si era corto
                if len(buffer_actual[-1]) < 220:
                    buffer_actual = [buffer_actual[-1], p]
                    long_actual = len(buffer_actual[0]) + len(p)
                else:
                    buffer_actual = [p]
                    long_actual = len(p)
            else:
                buffer_actual.append(p)
                long_actual += len(p)

        if buffer_actual:
            texto_hijo = "\n\n".join(buffer_actual)
            texto_embed = f"[Entidad: {titulo_pagina} | Sección: {nom_sec}]\n{texto_hijo}"
            chunks_resultado.append({
                "id": f"{titulo_pagina}#chunk#{orden_global}",
                "entidad_id": titulo_pagina,
                "seccion": nom_sec,
                "orden": orden_global,
                "texto": texto_hijo,
                "texto_padre": texto_padre_sec,
                "texto_embedding": texto_embed[:2000],
                "url_wiki": url_sec,
            })
            orden_global += 1

    return chunks_resultado


def calcular_embeddings_lote(textos: list[str]) -> list[list[float]]:
    """
    Calcula embeddings multilingües de 384 dimensiones de forma 100% local y gratuita
    usando FastEmbed (ONNX) con 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'.
    """
    if not textos:
        return []
    try:
        modelo = obtener_modelo_embeddings()
        return [vec.tolist() for vec in modelo.embed(textos, batch_size=64)]
    except Exception as e:
        print(f"   ⚠️ Excepción calculando embeddings locales: {e}")
        return []


def guardar_en_neo4j(driver, titulo_pagina, texto_web, relaciones, generar_embeddings=True):
    with driver.session() as session:
        if titulo_pagina.lower() not in HUBS_INDICE:
            session.run(
                """
                MERGE (e:Entity {id: $id})
                SET e.url_wiki = $url,
                    e.texto_web = $texto_web
                """,
                id=titulo_pagina,
                url=f"{WIKI_BASE_URL}{titulo_pagina.replace(' ', '_')}",
                texto_web=texto_web
            )

            if generar_embeddings and texto_web:
                chunks = generar_chunks_jerarquicos_small_to_big(titulo_pagina, texto_web)
                if chunks:
                    vectores = calcular_embeddings_lote([c["texto_embedding"] for c in chunks])
                    if len(vectores) == len(chunks):
                        for c, vec in zip(chunks, vectores):
                            c["embedding"] = vec
                        session.run(
                            """
                            MATCH (e:Entity {id: $entidad_id})
                            OPTIONAL MATCH (e)-[old_r:HAS_CHUNK]->(old_c:Chunk)
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
                            entidad_id=titulo_pagina,
                            chunks=chunks,
                        )
                        print(f"   🧠 Embeddings Small-to-Big guardados: {len(chunks)} chunks vectorizados.")

        if relaciones:
            session.run(
                """
                UNWIND $rels AS r
                MERGE (o:Entity {id: r.origen})
                ON CREATE SET o.url_wiki = $base_url + replace(r.origen, ' ', '_')
                MERGE (d:Entity {id: r.destino})
                ON CREATE SET d.url_wiki = $base_url + replace(r.destino, ' ', '_')
                MERGE (o)-[rel:RELATED_TO {id: r.rel_id}]->(d)
                SET rel.epoca = r.epoca,
                    rel.parrafo = r.parrafo,
                    rel.ref_titulo = r.ref_titulo,
                    rel.ref_url = r.ref_url,
                    rel.ref_texto = r.ref_texto
                """,
                rels=relaciones,
                base_url=WIKI_BASE_URL
            )


def asegurar_indices_neo4j(driver):
    """Crea las restricciones de unicidad y el índice vectorial de 384 dimensiones en Neo4j."""
    with driver.session() as s:
        s.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
        s.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
        s.run(
            "CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS "
            "FOR (c:Chunk) ON (c.embedding) "
            "OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}"
        )


def indexar_entidades_existentes_en_neo4j(driver=None, reconstruir_todo=False):
    """
    Genera los Chunks Jerárquicos Small-to-Big y sus Embeddings locales gratuitos para las entidades
    que están en Neo4j.
    """
    cerrar = False
    if driver is None:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        cerrar = True

    if reconstruir_todo:
        with driver.session() as s:
            s.run("MATCH (c:Chunk) DETACH DELETE c")
            s.run("DROP INDEX chunk_embeddings IF EXISTS")

    asegurar_indices_neo4j(driver)
    total_chunks = 0
    entidades_indexadas = 0

    try:
        with driver.session() as session:
            registros = session.run(
                """
                MATCH (e:Entity)
                WHERE e.texto_web IS NOT NULL AND trim(e.texto_web) <> ''
                  AND NOT (e)-[:HAS_CHUNK]->(:Chunk)
                RETURN e.id AS id, e.texto_web AS texto_web
                """
            ).data()

            for reg in registros:
                ent_id = reg["id"]
                texto = reg["texto_web"]
                chunks = generar_chunks_jerarquicos_small_to_big(ent_id, texto)
                if not chunks:
                    continue
                vectores = calcular_embeddings_lote([c["texto_embedding"] for c in chunks])
                if len(vectores) == len(chunks):
                    for c, vec in zip(chunks, vectores):
                        c["embedding"] = vec
                    session.run(
                        """
                        MATCH (e:Entity {id: $entidad_id})
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
                        entidad_id=ent_id,
                        chunks=chunks,
                    )
                    entidades_indexadas += 1
                    total_chunks += len(chunks)
    finally:
        if cerrar:
            driver.close()

    return {"entidades_indexadas": entidades_indexadas, "total_chunks": total_chunks}


def limpiar_base_de_datos(driver):
    """Elimina todos los nodos y relaciones existentes en lotes seguros antes de iniciar un nuevo crawl."""
    print("🧹 Limpiando base de datos en Neo4j (borrando todos los nodos y relaciones anteriores en lotes)...")
    with driver.session() as session:
        while True:
            res = session.run(
                """
                MATCH (n)
                WITH n LIMIT 4000
                DETACH DELETE n
                RETURN count(*) AS borrados
                """
            ).single()
            if not res or res["borrados"] == 0:
                break
    print("✅ Base de datos vacía y lista para el nuevo crawl.")


def ejecutar_crawler(semilla="Lore", max_profundidad=999, max_paginas_seguridad=8000, borrar_antes=True):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    if borrar_antes:
        limpiar_base_de_datos(driver)

    asegurar_indices_neo4j(driver)

    if semilla.lower() not in HUBS_INDICE:
        res_semilla = resolver_redirecciones_wiki([semilla])
        semilla = res_semilla.get(semilla, semilla)

    cola = deque([(semilla, 0)])
    visitados = set()
    paginas_procesadas = 0

    # Si se lanza con borrar_antes=False (modo reanudar), marcar como ya visitadas las que ya tienen texto_web
    # y cargar en la cola los nodos stub pendientes que aún no tienen texto_web
    if not borrar_antes:
        with driver.session() as session:
            ya_hechas = session.run(
                "MATCH (e:Entity) WHERE e.texto_web IS NOT NULL AND trim(e.texto_web) <> '' RETURN toLower(e.id) AS id"
            ).data()
            for reg in ya_hechas:
                visitados.add(reg["id"])

            pendientes_db = session.run(
                "MATCH (e:Entity) WHERE e.texto_web IS NULL OR trim(e.texto_web) = '' RETURN e.id AS id"
            ).data()
            for reg in pendientes_db:
                if reg["id"] and reg["id"].lower() not in visitados:
                    cola.append((reg["id"], 1))
        print(f"🔄 Modo reanudar: {len(visitados)} entidades completas y {len(cola)} entidades en cola listas para continuar.")

    print(f"🚀 Iniciando Crawler desde '{semilla}' (profundidad={max_profundidad}, tope seguridad={max_paginas_seguridad} páginas)...")

    try:
        while cola and paginas_procesadas < max_paginas_seguridad:
            actual, prof = cola.popleft()
            if actual.lower() in visitados or prof > max_profundidad:
                continue
            visitados.add(actual.lower())
            paginas_procesadas += 1

            es_hub_actual = actual.lower() in HUBS_INDICE
            icono = "🗂️ [HUB]" if es_hub_actual else "👤 [ENTITY]"
            print(f"[{paginas_procesadas}/{max_paginas_seguridad}] {icono} (Depth {prof}) '{actual}'...")

            wikitext = obtener_wikitexto_lore(actual)
            if not wikitext:
                continue

            texto_web, relaciones, vecinos = procesar_pagina(actual, wikitext)
            guardar_en_neo4j(driver, actual, texto_web, relaciones)
            print(f"   💾 Texto enciclopédico ({len(texto_web)} chars) | +{len(relaciones)} relaciones | +{len(vecinos)} entidades.")

            sig_prof = prof if es_hub_actual else prof + 1
            if sig_prof <= max_profundidad:
                for v in vecinos:
                    if v.lower() not in visitados:
                        cola.append((v, sig_prof))

            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n⏸️ Crawler detenido manualmente por el usuario. Todo lo procesado hasta aquí ya está guardado en Neo4j.")
    finally:
        driver.close()
        print(f"\n🎉 Finalizado. Páginas procesadas: {paginas_procesadas}.")


if __name__ == "__main__":
    reanudar = "--reanudar" in sys.argv
    ejecutar_crawler(semilla="Lore", max_profundidad=999, max_paginas_seguridad=8000, borrar_antes=not reanudar)
