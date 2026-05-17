from torchvision import transforms
from PIL import Image
import torch

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

def preprocess_image(image):

    image = image.convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)

    return image

def get_top_predictions(outputs, classes):

    probs = torch.softmax(outputs, dim=1)
    top_probs, top_indices = torch.topk(probs, 3)

    results = []

    for prob, idx in zip(top_probs[0], top_indices[0]):
        results.append({
            "class": classes[idx],
            "confidence": float(prob) * 100
        })

    return results