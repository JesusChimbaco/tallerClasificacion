"""
API generalizada de clasificación de imágenes con FastAPI.
Permite entrenar múltiples modelos desde archivos .zip y predecir con detección automática.
"""

import os
import io
import zipfile
import shutil
import tempfile
import json
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
import joblib

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

MODELS_DIR = Path("models")
REGISTRY_FILE = MODELS_DIR / "registry.json"
MODELS_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="Image Classifier API",
    description="Entrena y predice modelos de clasificación de imágenes por histograma de color.",
    version="1.0.0",
)

def extract_histogram(image_bytes: bytes) -> np.ndarray:
    """
    Devuelve un vector de 96 dimensiones: 3 histogramas 1D separados
    (canales B, G, R) con 32 bins cada uno, normalizados y concatenados.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("No se pudo decodificar la imagen.")
    image = cv2.resize(image, (128, 128))
    histograms = []
    for i in range(3):  # canales B, G, R
        hist = cv2.calcHist([image], [i], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        histograms.append(hist)
    return np.concatenate(histograms).astype(np.float32)


def extract_histogram_from_path(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        return extract_histogram(f.read())



def load_registry() -> dict:
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE, "r") as f:
            return json.load(f)
    return {}


def save_registry(registry: dict):
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

def train_model_from_directory(data_dir: Path, model_name: str) -> dict:
    classes = [d.name for d in data_dir.iterdir() if d.is_dir()]
    if len(classes) < 2:
        raise ValueError(
            f"Se necesitan al menos 2 clases; se encontraron: {classes}"
        )

    X, y = [], []
    for cls in classes:
        class_dir = data_dir / cls
        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            try:
                feat = extract_histogram_from_path(str(img_file))
                X.append(feat)
                y.append(cls)
            except Exception as e:
                print(f"[WARN] Omitiendo {img_file}: {e}")

    if len(X) < 10:
        raise ValueError(
            f"Muy pocas imágenes ({len(X)}). Se requieren al menos 10 por entrenamiento."
        )

    X = np.array(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)

    model_path = MODELS_DIR / f"{model_name}.pkl"
    scaler_path = MODELS_DIR / f"{model_name}_scaler.pkl"
    joblib.dump(clf, model_path)
    joblib.dump(scaler, scaler_path)

    registry = load_registry()
    registry[model_name] = {
        "classes": classes,
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "num_images": len(X),
        "train_accuracy": round(train_acc, 4),
        "test_accuracy": round(test_acc, 4),
    }
    save_registry(registry)

    return {
        "model_name": model_name,
        "classes": classes,
        "num_images": len(X),
        "train_accuracy": round(train_acc, 4),
        "test_accuracy": round(test_acc, 4),
    }

def predict_image(image_bytes: bytes, model_name: Optional[str] = None) -> dict:
    registry = load_registry()
    if not registry:
        raise ValueError("No hay modelos entrenados. Sube un dataset primero.")

    feat = extract_histogram(image_bytes).reshape(1, -1)

    if model_name:
        if model_name not in registry:
            raise ValueError(f"Modelo '{model_name}' no encontrado. Disponibles: {list(registry.keys())}")
        candidates = {model_name: registry[model_name]}
    else:
        candidates = registry

    best = None
    best_conf = -1.0
    best_model_name = None

    for name, info in candidates.items():
        clf = joblib.load(info["model_path"])
        scaler = joblib.load(info["scaler_path"])

        feat_scaled = scaler.transform(feat)
        probas = clf.predict_proba(feat_scaled)[0]
        max_conf = float(np.max(probas))
        predicted_class = clf.classes_[np.argmax(probas)]

        if max_conf > best_conf:
            best_conf = max_conf
            best = {"class": predicted_class, "probabilities": dict(zip(clf.classes_, probas.tolist()))}
            best_model_name = name

    return {
        "model_used": best_model_name,
        "predicted_class": best["class"],
        "confidence": round(best_conf, 4),
        "probabilities": {k: round(v, 4) for k, v in best["probabilities"].items()},
    }

@app.get("/", summary="Estado de la API")
def root():
    registry = load_registry()
    return {
        "status": "ok",
        "registered_models": list(registry.keys()),
        "total_models": len(registry),
    }


@app.post("/train", summary="Entrenar un modelo desde un ZIP")
async def train(
    file: UploadFile = File(..., description="Archivo .zip con subcarpetas por clase"),
    model_name: str = Form(..., description="Nombre identificador del modelo (ej: 'aves', 'frutas')"),
):
    """
    Recibe un `.zip` con la estructura:

    ```
    dataset.zip/
        clase_A/
            img1.jpg
        clase_B/
            img2.jpg
    ```

    Entrena un modelo de Regresión Logística y lo guarda.
    """
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="El archivo debe ser un .zip")

    content = await file.read()

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "dataset.zip")
        with open(zip_path, "wb") as f:
            f.write(content)

        extract_dir = Path(tmpdir) / "data"
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        # Buscar la carpeta raíz del dataset
        # (por si el zip contiene una carpeta contenedora)
        subdirs = [d for d in extract_dir.iterdir() if d.is_dir()]
        if len(subdirs) == 1 and not any(extract_dir.glob("*.jpg")):
            data_root = subdirs[0]
        else:
            data_root = extract_dir

        try:
            metrics = train_model_from_directory(data_root, model_name)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

    return JSONResponse(content={"status": "entrenado", **metrics})


@app.post("/predict", summary="Clasificar una imagen")
async def predict(
    file: UploadFile = File(..., description="Imagen a clasificar (jpg, png, etc.)"),
    model_name: Optional[str] = Form(
        None,
        description="Nombre del modelo a usar. Si se omite, se elige automáticamente.",
    ),
):
    """
    Clasifica una imagen.

    - Si se envía `model_name`, usa ese modelo específico.
    - Si se omite, prueba **todos** los modelos y elige el de mayor confianza.
    """
    content = await file.read()
    try:
        result = predict_image(content, model_name)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return JSONResponse(content=result)


@app.get("/models", summary="Listar modelos entrenados")
def list_models():
    """Devuelve el registro completo de modelos disponibles."""
    return load_registry()


@app.delete("/models/{model_name}", summary="Eliminar un modelo")
def delete_model(model_name: str):
    registry = load_registry()
    if model_name not in registry:
        raise HTTPException(status_code=404, detail=f"Modelo '{model_name}' no existe.")

    info = registry.pop(model_name)
    for key in ("model_path", "scaler_path"):
        p = Path(info[key])
        if p.exists():
            p.unlink()

    save_registry(registry)
    return {"status": "eliminado", "model_name": model_name}