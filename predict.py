import torch
import json
from PIL import Image
from torchvision import transforms

from model import FaceClassifier

IMG_SIZE = 160
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONF_THRESHOLD = 0.25 

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# load class names
with open(r"D:\OneDrive\FastAPI projects\Face_recognition_FastAPI\class_names (1).json", "r") as f:
    class_names = json.load(f)

# load model
model = FaceClassifier(num_classes=len(class_names), freeze_backbone=False)
state = torch.load(
    r"D:\OneDrive\FastAPI projects\Face_recognition_FastAPI\face_recognition_robust_full_model.pth",
    map_location=DEVICE
)
model.load_state_dict(state["model_state_dict"])
model.to(DEVICE)
model.eval()


def predict_image(image: Image.Image):
    img = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(img)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

    confidence = float(conf.item())
    predicted_class = class_names[pred.item()]

    # apply threshold
    if confidence < CONF_THRESHOLD:
        predicted_class = "Stranger"

    return {
        "class": predicted_class,
        "confidence": confidence
    }
