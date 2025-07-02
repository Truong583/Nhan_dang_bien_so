import torch
import torch.nn as nn
from torchvision import models, transforms
from dataset import LicensePlateCharacterDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
           "A", "B", "C", "D", "E", "F", "G", "H", "K", "L",
           "M", "N", "P", "S", "T", "U", "V", "X", "Y", "Z"]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

val_dir = r"D:\Bien_So\char_data\val"
dataset = LicensePlateCharacterDataset(val_dir, classes, transform=transform)

model = models.mobilenet_v3_small(weights=None)
model.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
model.load_state_dict(torch.load(r"D:\Bien_So\model_ocr.pth", map_location=device))
model = model.to(device)
model.eval()

correct = 0
total = len(dataset)

with torch.no_grad():
    for idx in range(total):
        image, label = dataset[idx]
        input_tensor = image.unsqueeze(0).to(device)
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()

        if pred == label:
            correct += 1
        else:
            print(f"Sai: Ảnh {idx} - Dự đoán: {classes[pred]} - Thực tế: {classes[label]}")

print(f"\nTổng số ảnh: {total}")
print(f"Số đúng: {correct}")
print(f"Độ chính xác: {100 * correct / total:.2f}%")
