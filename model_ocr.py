import torch
import torch.nn as nn

def load_ocr_model(model_path, device):
    from torchvision import models
    classes = 30  # số lớp ký tự
    model = models.mobilenet_v3_small(weights=None)
    model.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device).eval()

def predict_character(model, image_tensor, class_names):
    with torch.no_grad():
        output = model(image_tensor)
        _, pred = output.max(1)
        return class_names[pred.item()]
