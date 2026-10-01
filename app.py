import ast
import torch
import torch.nn as nn
import streamlit as st
from torchvision import transforms
from PIL import Image

CLASSES = [
    'Corn Cercospora Gray Leaf Spot', 'Corn Common Rust', 'Corn Northern Leaf Blight', 'Corn Healthy',
    'Rice Bacterial Leaf Blight', 'Rice Brown Spot', 'Rice Leaf Blast', 'Rice Leaf Scald',
    'Rice Sheath Blight', 'Rice Healthy'
]


class ScratchCropCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ScratchCropCNN, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.block2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.block3 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.block4 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


@st.cache_resource
def load_model():
    model = ScratchCropCNN(num_classes=10)
    model.load_state_dict(torch.load("crop_disease_scratch_cnn.pth", map_location=torch.device('cpu')))
    model.eval()
    return model


@st.cache_resource
def load_normalization_stats():
    """
    Loads the dataset-specific mean/std saved during training
    (normalization_stats.txt, produced in Module 3d of the training script).
    This replaces the earlier ImageNet-stat approach, which wasn't actually
    justified once transfer learning was dropped in favor of a scratch model.
    """
    with open("normalization_stats.txt", "r") as f:
        lines = f.readlines()
    mean = ast.literal_eval(lines[0].split("=", 1)[1].strip())
    std = ast.literal_eval(lines[1].split("=", 1)[1].strip())
    return mean, std


model = load_model()
DATASET_MEAN, DATASET_STD = load_normalization_stats()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD)
])

st.title("Crop Disease Detection Classifier")
st.caption("Supports corn (maize) and rice leaf images only — 10 classes total.")

uploaded_file = st.file_uploader("Upload a Maize/Corn or Rice Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        conf, pred_idx = torch.max(probabilities, dim=0)

    confidence_pct = conf.item() * 100

    if confidence_pct >= 60.0:
        st.success(f"**Prediction:** {CLASSES[pred_idx.item()]}")
        st.info(f"**Confidence Score:** {confidence_pct:.2f}%")
    else:
        st.warning("⚠️ **Uncertainty Fallback:** Confidence is below 60%. Unclear sample detected. Please re-upload a clearer leaf image.")
