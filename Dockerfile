FROM python:3.11-slim

# Crear usuario sin privilegios (requisito de Hugging Face Spaces: UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/home/user/.cache/fastembed

WORKDIR $HOME/app

# Instalar dependencias Python
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-descargar el modelo multilingüe de embeddings (FastEmbed 384d) durante el build
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

# Copiar los archivos de la aplicación (sin el archivo .env local)
COPY --chown=user crawler_entidades.py app_visor.py icono.png favicon.png ./

EXPOSE 7860

CMD ["uvicorn", "app_visor:app", "--host", "0.0.0.0", "--port", "7860"]
