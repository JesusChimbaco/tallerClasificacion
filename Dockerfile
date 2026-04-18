# Imagen base oficial de Python (fijada a bookworm para mayor estabilidad)
FROM python:3.11-slim-bookworm

# Evita que Python genere archivos .pyc y que bufferee stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalar dependencias del sistema que necesita OpenCV
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update -o Acquire::Retries=3 \
    && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependencias primero (aprovecha caché de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY main.py .

# Crear carpeta de modelos con permisos correctos
RUN mkdir -p /app/models

# Exponer el puerto de la API
EXPOSE 8000

# Comando para iniciar el servidor
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
