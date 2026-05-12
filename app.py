import torch
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
import torch.nn as nn
import gradio as gr
import json

# ---------------------------
# CONFIG (Updated paths for HF Spaces)
# ---------------------------
MODEL_PATH = "best_model.pth"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = 160
NUM_CLASSES = 50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# LOAD CLASS NAMES
# ---------------------------
print("Loading class names...")
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = json.load(f)
print(f"Loaded {len(class_names)} classes")

# ---------------------------
# TRANSFORM
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# ---------------------------
# MODEL DEFINITION
# ---------------------------
class SpatialAttention(nn.Module):
    def __init__(self, embed_dim=512, reduction=16):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(embed_dim // reduction, embed_dim, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.attn(x)
        return x * w, w

class FaceClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = InceptionResnetV1(pretrained="vggface2")
        self.attention = SpatialAttention()
        self.head = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        emb = self.backbone(x)
        emb, _ = self.attention(emb)
        return self.head(emb)

# ---------------------------
# LOAD MODEL
# ---------------------------
print("Loading model...")
model = FaceClassifier(NUM_CLASSES).to(DEVICE)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
else:
    model.load_state_dict(checkpoint)

model.eval()
print("Model loaded successfully!")

# ---------------------------
# PREDICTION FUNCTION
# ---------------------------
def predict_face(image):
    """Predicts the person in the input image."""
    if image is None:
        return {}
    
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    # Preprocess
    img_tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
    
    # Get top 5 predictions
    top5_probs, top5_indices = torch.topk(probs, min(5, NUM_CLASSES))
    
    # Create results dictionary
    results = {}
    for prob, idx in zip(top5_probs[0], top5_indices[0]):
        results[class_names[idx.item()]] = float(prob.item())
    
    return results

# ---------------------------
# GRADIO INTERFACE
# ---------------------------
demo = gr.Interface(
    fn=predict_face,
    inputs=gr.Image(type="pil", label="Upload Face Image"),
    outputs=gr.Label(num_top_classes=5, label="Predictions"),
    title="Face Recognition System",
    description="Upload a face image to identify the person. The model will show the top 5 predictions with confidence scores.",
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

# ---------------------------
# LAUNCH
# ---------------------------
if __name__ == "__main__":
    demo.launch()