import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from torchvision import models, transforms
from PIL import Image

# -------------------------------
# MODEL LOADING (ONCE)
# -------------------------------
NUM_CLASSES = 38
MODEL_PATH = "sai1.pth"

device = torch.device("cpu")

model = models.resnet34(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


# -------------------------------
# IMAGE TRANSFORM (INFERENCE ONLY)
# -------------------------------
# IMAGE TRANSFORM (MATCH TRAINING)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])



def transform_image(image: Image.Image):
    """
    Takes PIL Image, returns tensor
    """
    image = image.convert("RGB")
    return transform(image).unsqueeze(0)


# -------------------------------
# PREDICTION
# -------------------------------
def get_prediction(image_tensor):
    image_tensor = image_tensor.to(device)

    with torch.no_grad():          # 🔥 VERY IMPORTANT
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)

    return predicted.item()

with torch.no_grad():
    outputs = model(image_tensor)
    probs = torch.softmax(outputs, dim=1)
    top_prob, top_class = probs.max(1)

print("Top prob:", top_prob.item())
print("Class index:", top_class.item())


