"""Reconstruye las relaciones de los textos del juego a partir de sus propiedades.

No necesita re-crawl ni re-importar: solo lee `entidad_id` y `menciona` de los nodos :TextoJuego.
Crea (:TextoJuego)-[:SOBRE]->(:Entity), (:TextoJuego)-[:MENCIONA]->(:Entity) y encadena los
volúmenes de un mismo libro con [:SIGUIENTE]. Es idempotente y se puede ejecutar cuando se quiera.

    python enlaces_juego.py
"""


def reenlazar(driver) -> dict:
    with driver.session() as s:
        s.run("MATCH (:TextoJuego)-[r:SOBRE|MENCIONA|SIGUIENTE]->() DELETE r")
        sobre = s.run(
            """
            MATCH (t:TextoJuego) WHERE t.entidad_id <> ''
            MATCH (e:Entity {id: t.entidad_id})
            MERGE (t)-[:SOBRE]->(e)
            RETURN count(*) AS n
            """
        ).single()["n"]
        menciona = s.run(
            """
            MATCH (t:TextoJuego) WHERE size(t.menciona) > 0
            UNWIND t.menciona AS nombre
            MATCH (e:Entity {id: nombre})
            MERGE (t)-[:MENCIONA]->(e)
            RETURN count(*) AS n
            """
        ).single()["n"]
        siguiente = s.run(
            """
            MATCH (t:TextoJuego {tipo: 'libro'})
            WITH t.titulo_obra AS obra, t ORDER BY toInteger(split(t.id, ':')[1])
            WITH obra, collect(t) AS ts WHERE size(ts) > 1
            UNWIND range(0, size(ts) - 2) AS i
            WITH ts[i] AS a, ts[i + 1] AS b
            MERGE (a)-[:SIGUIENTE]->(b)
            RETURN count(*) AS n
            """
        ).single()["n"]
    return {"sobre": sobre, "menciona": menciona, "siguiente": siguiente}


def indexar_textos_juego_existentes(driver) -> int:
    """Genera embeddings faltantes para los nodos :TextoJuego ya existentes en Neo4j."""
    import crawler_entidades

    with driver.session() as s:
        s.run(
            "CREATE VECTOR INDEX textojuego_embeddings IF NOT EXISTS "
            "FOR (t:TextoJuego) ON (t.embedding) "
            "OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}"
        )
        pendientes = s.run(
            "MATCH (t:TextoJuego) WHERE t.embedding IS NULL AND t.texto IS NOT NULL RETURN t.id AS id, t.titulo AS titulo, t.titulo_obra AS obra, t.texto AS texto"
        ).data()

    if not pendientes:
        return 0

    print(f"Generando embeddings vectoriales para {len(pendientes)} textos del juego...")
    total = 0
    for i in range(0, len(pendientes), 64):
        lote = pendientes[i:i + 64]
        textos_vec = [f"{r.get('obra', '')} - {r.get('titulo', '')}: {(r.get('texto') or '')[:1500]}" for r in lote]
        vectores = crawler_entidades.calcular_embeddings_lote(textos_vec)
        if len(vectores) != len(lote):
            continue
        pares = [{"id": r["id"], "vec": vec} for r, vec in zip(lote, vectores)]
        with driver.session() as s:
            s.run(
                """
                UNWIND $pares AS p
                MATCH (t:TextoJuego {id: p.id})
                SET t.embedding = p.vec
                """,
                pares=pares,
            )
        total += len(pares)
    return total


if __name__ == "__main__":
    import sys
    import crawler_entidades

    d = crawler_entidades.crear_driver_neo4j()
    try:
        print("Relaciones:", reenlazar(d))
        if "--indexar" in sys.argv:
            print("Textos vectorizados:", indexar_textos_juego_existentes(d))
    finally:
        d.close()
