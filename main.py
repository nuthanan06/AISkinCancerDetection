from torchvision import datasets, models
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image
import os


class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.labels = labels
        self.images = images
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.images[idx]
        return (image, label)

def my_collate(batch):
    """Define collate_fn myself because the default_collate_fn throws errors like crazy"""
    # item: a tuple of (img, label)
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    data = torch.stack(data)
    target = torch.LongTensor(target)
    print("ran")
    return [data, target]

def train_step(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer):
  train_loss = 0
  train_acc = 0
  count = 0
  model.train()

  for batch, (X, y) in enumerate(data_loader):
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    train_loss += loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
    train_acc += (y_pred_class==y).sum().item()/len(y_pred)
    print(f"{count} | Training Loss: {loss} | Train Acc: {(y_pred_class==y).sum().item()/len(y_pred)}") 
    count += 1

  train_loss /= len(data_loader)
  train_acc /= len(data_loader)
  return train_loss, train_acc

def test_step(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module):
  test_loss, test_acc = 0,0
  model.eval()
  with torch.inference_mode():
      for X, y in data_loader:
        test_pred = model(X)
        test_loss += loss_fn(test_pred, y)
        test_pred_labels = test_pred.argmax(dim=1)
        test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

      test_loss /= len(data_loader)
      test_acc /= len(data_loader)

  return test_loss, test_acc

def train(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader, test_dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module = nn.CrossEntropyLoss(), epochs: int = 5):
  results = {"train_loss": [],
             "train_acc": [],
             "test_loss": [], 
             "test_acc": []}
  
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Learning rate scheduler

  for epoch in range(epochs):
    print('training')
    train_loss, train_acc = train_step(model=model, data_loader=train_dataloader, loss_fn=loss_fn, optimizer = optimizer)
    test_loss, test_acc = test_step(model=model, data_loader = test_dataloader, loss_fn=loss_fn)

    print(f"Epoch: {epoch} | Train loss; {train_loss:.4f} | Train acc: {train_acc:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")
    results["train_loss"].append(train_loss.item())
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)

    scheduler.step()  # Update learning rate


  return results


def main():
  class_dict = {0: "akiec", 1: "bcc", 2: "bkl", 3: "df", 4: "mel", 5: "nv", 6: "vasc"}
  norm_mean = (0.49139968, 0.48215827, 0.44653124)
  norm_std = (0.24703233, 0.24348505, 0.26158768)
  data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(), 
    transforms.Normalize(norm_mean, norm_std)
  ])

  tensor_transform = transforms.Compose([
      transforms.Resize(size=(224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(norm_mean, norm_std)
  ])

  # Load the original dataset
  train_dir = "./data/train"
  test_dir = "./data/test"
  original_train_data = datasets.ImageFolder(root=train_dir, transform=tensor_transform)

  # Calculate the number of images per class in the original dataset
  class_counts = torch.bincount(torch.tensor(original_train_data.targets))

  # Calculate the maximum number of images per class
  max_class_count = max(class_counts)

  # Create a ConcatDataset to store the augmented data
  augmented_datasets = []
  class_datasets = []
  for class_idx, class_count in enumerate(class_counts):
      image_folder_path = "./data/train/" + class_dict[class_idx] 
      image_filenames = [filename for filename in os.listdir(image_folder_path) if filename.endswith(".jpeg") or filename.endswith(".jpg")]
      class_images = []  # To store images
      class_labels = []  
      for image in image_filenames: 
        random_image_path = os.path.join(image_folder_path, image)
        random_image = Image.open(random_image_path)
        random_image = tensor_transform(random_image)
        class_images.append(random_image)
        class_labels.append(class_idx)  # Add corresponding class index as the label

    # Create a Dataset from the images and labels
      class_images_tensor = torch.stack(class_images)
      class_labels_tensor = torch.tensor(class_labels)

      # Combine images and labels into a Dataset
      class_dataset = torch.utils.data.TensorDataset(class_images_tensor, class_labels_tensor)
      class_datasets.append(class_dataset)

      if class_count < max_class_count:
        num_augmented_samples = max_class_count - class_count

        while num_augmented_samples > 0: 
          # Augment the class dataset by randomly selecting samples and applying transformations
          augmented_samples = torch.randperm(len(class_dataset))[:num_augmented_samples]
          augmented_dataset = torch.utils.data.Subset(class_dataset, augmented_samples)

          augmneted_images_list = []
          augmented_labels_list = []
          for image in augmented_dataset: 
            augmented_images = data_transform(image[0]) 
            augmneted_images_list.append(augmented_images)
            augmented_labels_list.append(image[1])

          augmented_images_tensor = torch.stack(augmneted_images_list)
          augmented_labels_tensor = torch.tensor(augmented_labels_list)

          # Combine images and labels into a new TensorDataset
          augmented_class_dataset = torch.utils.data.TensorDataset(
              augmented_images_tensor,
              augmented_labels_tensor
          )

          augmented_datasets.append(augmented_class_dataset)
          num_augmented_samples -= len(augmented_samples)

  # Concatenate the original dataset with augmented datasets

  images = []
  labels = []

  for i in class_datasets: 
      for j in i:  
        images.append(j[0])
        labels.append(j[1])

  for augmented_dataset in augmented_datasets:
      for i in augmented_dataset:
          images.append(i[0])
          labels.append(i[1])

  balanced_data = CustomDataset(images, labels)

  # Create dataloader for the balanced training data
  BATCH_SIZE = 32
  train_dataloader = DataLoader(dataset=balanced_data, collate_fn = my_collate, batch_size=BATCH_SIZE, num_workers=1, shuffle=True)

  model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
  model.fc = nn.Linear(2048, 7)

  test_data = datasets.ImageFolder(root=test_dir, transform=tensor_transform)
  test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, num_workers=1, shuffle=False)


  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01, weight_decay=0.001)
  model_results = train(model=model, train_dataloader=train_dataloader, test_dataloader=test_dataloader, optimizer=optimizer, loss_fn = loss_fn, epochs=50)
  print(model_results)

  MODEL_PATH = Path("./models")
  MODEL_PATH.mkdir(parents=True, exist_ok = True)
  MODEL_NAME = "CancerModel.pth"
  MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
  torch.save(obj = model.state_dict(), f = MODEL_SAVE_PATH)

if __name__ == '__main__':
     main()
