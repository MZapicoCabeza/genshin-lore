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


if __name__ == "__main__":
    import crawler_entidades

    d = crawler_entidades.crear_driver_neo4j()
    try:
        print(reenlazar(d))
    finally:
        d.close()
