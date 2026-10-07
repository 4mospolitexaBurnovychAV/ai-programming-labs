import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import copy
import numpy as np

# Пристрій для обчислень (CPU або GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Підготовка трансформацій для датасету з Лабораторної роботи №1
# Приводимо зображення до розміру 224x224 та переводимо у тензор
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. Механізм DataLoader з розбиттям, збереженим у Лабораторній роботі №1
class ImagePathDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = LAB1_DIR / Path(str(self.image_paths[index]).replace("\\", "/"))
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, int(self.labels[index])


LAB1_DIR = Path(__file__).resolve().parent.parent
with np.load(LAB1_DIR / "dataset_splits.npz") as split_data:
    classes = split_data["classes"].tolist()
    train_dataset = ImagePathDataset(split_data["X_train"], split_data["y_train"], transform)
    val_dataset = ImagePathDataset(split_data["X_val"], split_data["y_val"], transform)
    test_dataset = ImagePathDataset(split_data["X_test"], split_data["y_test"], transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

num_classes = len(classes)

# 3. Базова CNN щонайменше з двома згортковими блоками
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        # Перший блок
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Другий блок
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Повнозв'язні шари (224 -> 112 -> 56)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.fc(x)
        return x

model = SimpleCNN(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Явний цикл train/validation
num_epochs = 10
train_losses, val_losses = [], []
train_accs, val_accs = [], []

best_model_wts = copy.deepcopy(model.state_dict())
best_val_acc = 0.0

for epoch in range(num_epochs):
    # --- TRAIN ---
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad() # Обнулення градієнтів
        logits = model(x)     # Forward
        loss = criterion(logits, y) # Loss
        loss.backward()       # Backward
        optimizer.step()      # Optimizer step
        
        running_loss += loss.item() * x.size(0)
        _, preds = torch.max(logits, 1)
        correct += torch.sum(preds == y.data).item()
        total += y.size(0)
        
    train_losses.append(running_loss / total)
    train_accs.append(correct / total)

    # --- VALIDATION ---
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad(): # Відключення обчислення градієнтів
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            val_loss += loss.item() * x.size(0)
            _, preds = torch.max(logits, 1)
            correct += torch.sum(preds == y.data).item()
            total += y.size(0)
            
    epoch_val_loss = val_loss / total
    epoch_val_acc = correct / total
    val_losses.append(epoch_val_loss)
    val_accs.append(epoch_val_acc)
    
    # 5. Збереження ваг найкращої моделі за валідаційною метрикою
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), 'best_model.pth')

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_losses[-1]:.4f} | Val Acc: {val_accs[-1]:.4f}")

# Завантажуємо найкращу модель для тестування
model.load_state_dict(best_model_wts)

# 6. Графіки loss та accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.title('Loss per Epoch')

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.legend()
plt.title('Accuracy per Epoch')
plt.show()

# 7. Оцінка на test-наборі
model.eval()
all_preds, all_labels, all_images = [], [], []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        _, preds = torch.max(logits, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        all_images.extend(x.cpu().numpy()) # зберігаємо для візуалізації

print("Accuracy:", accuracy_score(all_labels, all_preds))
print("Precision (macro):", precision_score(all_labels, all_preds, average='macro', zero_division=0))
print("Recall (macro):", recall_score(all_labels, all_preds, average='macro', zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))

# 8. Показ 5 правильних і 5 помилкових прогнозів
correct_idx = np.where(np.array(all_preds) == np.array(all_labels))[0]
incorrect_idx = np.where(np.array(all_preds) != np.array(all_labels))[0]

def imshow(img, title):
    # Денормалізація для візуалізації
    img = img.transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')

# Візуалізація правильних прогнозів
plt.figure(figsize=(15, 3))
for i, idx in enumerate(correct_idx[:5]):
    plt.subplot(1, 5, i+1)
    title = f"True: {all_labels[idx]} | Pred: {all_preds[idx]}"
    imshow(all_images[idx], title)
plt.suptitle('Correct Predictions')
plt.show()

# Візуалізація помилкових прогнозів
plt.figure(figsize=(15, 3))
for i, idx in enumerate(incorrect_idx[:5]):
    plt.subplot(1, 5, i+1)
    title = f"True: {all_labels[idx]} | Pred: {all_preds[idx]}"
    imshow(all_images[idx], title)
plt.suptitle('Incorrect Predictions')
plt.show()