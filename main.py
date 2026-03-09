from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import pennylane as qml
from pennylane import numpy as np
from torchvision import models, transforms
from PIL import Image
import io
import uvicorn

app = FastAPI()

# This allows your HTML file to "talk" to this Python server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- QUANTUM ENGINE ---
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

class QuantumModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_net = models.resnet18(weights='IMAGENET1K_V1')
        self.pre_net.fc = torch.nn.Linear(512, n_qubits)
        self.q_weights = torch.nn.Parameter(0.01 * torch.randn(3, n_qubits, 3))

    def forward(self, x):
        features = torch.sigmoid(self.pre_net(x)) * np.pi
        q_out = torch.stack([quantum_circuit(f, self.q_weights) for f in features])
        return (q_out + 1) / 2

model = QuantumModel()
model.eval()

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # Read the image sent from the HTML
    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        pred = model(img_t).item()
    
    status = "Malignant" if pred > 0.5 else "Benign"
    return {"status": status, "confidence": round(pred * 100 if pred > 0.5 else (1-pred)*100, 2)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)