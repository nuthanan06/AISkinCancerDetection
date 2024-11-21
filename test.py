import torch 
from pathlib import Path
from torchvision import datasets, models
from torchvision import datasets, transforms
from PIL import Image


torch.manual_seed(42)
MODEL_PATH = Path("./models")
MODEL_PATH.mkdir(parents=True, exist_ok = True)
MODEL_NAME = "CancerModel.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
loaded_model_2 = models.resnet101()
loaded_model_2.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

data_transform = transforms.Compose([
      transforms.Resize(size=(224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])

image_path = "actinic keratoses.jpeg"
image = Image.open(image_path)
input_tensor = data_transform(image)
input_tensor = input_tensor.unsqueeze(0)

class_dict = {0: "akiec", 1: "bcc", 2: "bkl", 3: "df", 4: "mel", 5: "nv", 6: "vasc"}

with torch.no_grad():
    loaded_model_2.eval()
    y_pred = loaded_model_2(input_tensor)
    classprediction = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
    print(class_dict[classprediction.item()])

