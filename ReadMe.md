#  Agente de Telegram para Clasificación de Imágenes

Agente de inteligencia artificial integrado con Telegram, orquestado mediante **n8n**, que permite entrenar y usar modelos de clasificación de imágenes por histograma de color RGB usando Regresión Logística.



##  Arquitectura
<img width="551" height="733" alt="imagen" src="https://github.com/user-attachments/assets/30aa135f-b0fa-4a1e-a980-63d591823d3b" />


``

### Flujo A — Entrenamiento
1. El usuario envía un archivo `.zip` con imágenes organizadas en subcarpetas (una por clase) y un nombre para el clasificador.
2. El AI Agent en n8n detecta la intención de entrenamiento y llama a la tool `POST /train`.
3. FastAPI descomprime el `.zip`, extrae histogramas RGB de cada imagen con OpenCV y entrena un modelo de Regresión Logística con scikit-learn.
4. El modelo se serializa con joblib en `/models/<nombre>.pkl`.
5. El agente responde al usuario con el **accuracy** obtenido.

### Flujo B — Clasificación
1. El usuario envía una imagen y el nombre del clasificador a usar.
2. El AI Agent llama a la tool `POST /classify`.
3. FastAPI carga el modelo `.pkl`, extrae el histograma de la imagen y devuelve la predicción.
4. El agente responde con la **clase predicha** y la **confianza**.

##  Endpoints de la API

| Método | Ruta | Parámetros | Respuesta |
|---|---|---|---|
| `POST` | `/train` | `file` (.zip), `classifier_name` (string) | `{"classifier_name": "...", "accuracy": 0.95, "classes": [...]}` |
| `POST` | `/classify` | `file` (imagen), `classifier_name` (string) | `{"prediction": "clase", "confidence": 0.87}` |
| `GET` | `/health` | — | `{"status": "ok"}` |

---

##  Formato del `.zip` de entrenamiento

```´´
dataset.zip
└── dataset/
    ├── clase_1/
    │   ├── imagen1.jpg
    │   └── imagen2.png
    ├── clase_2/
    │   └── ...
    └── clase_N/
        └── ...
```

- Mínimo recomendado: **20 imágenes por clase**
- Formatos soportados: `.jpg`, `.jpeg`, `.png`, `.bmp`


##  Dependencias principales

```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6
scikit-learn>=1.3.0
opencv-python-headless>=4.8.0
numpy>=1.24.0
joblib>=1.3.0
```
