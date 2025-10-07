import os
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torchvision.transforms as T

# ✅ Import model
from .model import load_model, DEVICE

# ✅ Path setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

MODEL_PATH = os.path.join(WEIGHTS_DIR, "best_ms_resnet18_fast.pth")
model = load_model(MODEL_PATH)

# ✅ Image Preprocessing
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

app = FastAPI(title="Multiple Sclerosis Detector")

# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Serve frontend
@app.get("/")
async def read_index():
    index_path = os.path.join(ASSETS_DIR, "index.html")
    if not os.path.isfile(index_path):
        return JSONResponse(content={"error": "index.html not found"}, status_code=404)
    return FileResponse(index_path)

# ✅ Serve static files
static_dir = os.path.join(ASSETS_DIR, "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ✅ Prediction route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img = Image.open(file.file).convert("RGB")
        inp = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(inp)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()

        class_names = ["Control", "MS"]
        result = {
            "prediction": class_names[pred_class],
            "confidence": float(probs[0][pred_class].item())
        }
        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
