import os
import gradio as gr
from fastapi.responses import HTMLResponse
from app_visor import app as visor_app, interfaz_web

# Registrar ruta /irminsul para servir la interfaz completa dentro del contenedor ZeroGPU/Gradio
visor_app.add_api_route("/irminsul", interfaz_web, methods=["GET"], response_class=HTMLResponse)

# Inyectar todos los endpoints (/irminsul y /api/*) de nuestro FastAPI dentro de la app interna de Gradio
_orig_create_app = gr.routes.App.create_app


def _custom_create_app(*args, **kwargs):
    fastapi_app = _orig_create_app(*args, **kwargs)
    rutas_existentes = {getattr(r, "path", "") for r in fastapi_app.router.routes}
    for route in visor_app.router.routes:
        ruta_path = getattr(route, "path", "")
        if ruta_path and ruta_path not in ("/", "/docs", "/redoc", "/openapi.json") and ruta_path not in rutas_existentes:
            fastapi_app.router.routes.insert(0, route)
    return fastapi_app


gr.routes.App.create_app = _custom_create_app

CSS_FULLSCREEN = """
footer { display: none !important; }
.gradio-container { padding: 0 !important; margin: 0 !important; max-width: 100vw !important; width: 100vw !important; height: 100vh !important; overflow: hidden !important; }
"""

with gr.Blocks(css=CSS_FULLSCREEN, title="Archivo del Irminsul · Genshin Impact Lore") as demo:
    gr.HTML(
        '<iframe src="/irminsul" style="position:fixed; top:0; left:0; width:100vw; height:100vh; border:none; margin:0; padding:0; z-index:999999; background:#161b26;" allow="clipboard-write"></iframe>'
    )

if __name__ == "__main__":
    puerto = int(os.getenv("GRADIO_SERVER_PORT", os.getenv("PORT", "7860")))
    ruta_fav = os.path.join(os.path.dirname(os.path.abspath(__file__)), "favicon.png")
    demo.launch(server_name="0.0.0.0", server_port=puerto, favicon_path=ruta_fav if os.path.exists(ruta_fav) else None)
