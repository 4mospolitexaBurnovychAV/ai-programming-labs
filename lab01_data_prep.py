import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
import albumentations as A

# 1. Фіксація Seed для відтворюваності (вимога п. 1.5)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_DIR = "data/raw_dataset"
IMG_SIZE = (224, 224)

def analyze_dataset(data_dir):
    """2. Програма аналізу датасету"""
    print("=== АНАЛІЗ ДАТАСЕТУ ===")
    if not os.path.exists(data_dir):
        print(f"[Помилка] Папка {data_dir} не знайдена!")
        return [], [], []

    classes = sorted(os.listdir(data_dir))
    classes = [c for c in classes if os.path.isdir(os.path.join(data_dir, c))]
    
    file_paths = []
    labels = []
    class_counts = {}
    image_sizes = []
    corrupted_files = 0

    for idx, class_name in enumerate(classes):
        class_folder = os.path.join(data_dir, class_name)
        images = os.listdir(class_folder)
        class_counts[class_name] = 0
        
        for img_name in images:
            img_path = os.path.join(class_folder, img_name)
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
            
            # Перевірка на нечитабельні файли
            img = cv2.imread(img_path)
            if img is None:
                corrupted_files += 1
                print(f"[Попередження] Пошкоджений файл: {img_path}")
                continue
                
            image_sizes.append(img.shape[:2])
            file_paths.append(img_path)
            labels.append(idx)
            class_counts[class_name] += 1

    print(f"Знайдено класів: {len(classes)} -> {classes}")
    print(f"Загальна кількість валідних зображень: {len(file_paths)}")
    print(f"Кількість пошкоджених файлів: {corrupted_files}")
    print(f"Розподіл по класах: {class_counts}")
    
    if image_sizes:
        avg_h = sum(s[0] for s in image_sizes) / len(image_sizes)
        avg_w = sum(s[1] for s in image_sizes) / len(image_sizes)
        print(f"Типові (середні) розміри зображень: Висота ~ {avg_h:.1f}, Ширина ~ {avg_w:.1f}")

    return file_paths, labels, classes

def demonstrate_augmentations(sample_img_path):
    """4. Реалізація не менше трьох змістовно допустимих аугментацій"""
    img = cv2.imread(sample_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)

    transform_flip = A.HorizontalFlip(p=1.0)
    transform_rotate = A.Rotate(limit=25, p=1.0)
    transform_bright = A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0)

    aug_flip = transform_flip(image=img)['image']
    aug_rotate = transform_rotate(image=img)['image']
    aug_bright = transform_bright(image=img)['image']

    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    axes[0].imshow(img); axes[0].set_title("Оригінал")
    axes[1].imshow(aug_flip); axes[1].set_title("Віддзеркалення")
    axes[2].imshow(aug_rotate); axes[2].set_title("Поворот")
    axes[3].imshow(aug_bright); axes[3].set_title("Яскравість/Контраст")
    
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.savefig("augmentations_example.png")
    print("Звіт з аугментаціями збережено у файл 'augmentations_example.png'.")
    plt.close()

def visualize_class_distribution_and_samples(file_paths, labels, classes):
    """6. Візуалізація розподілу класів та прикладів"""
    class_counts = Counter(labels)
    plt.figure(figsize=(6, 4))
    plt.bar(classes, [class_counts[i] for i in range(len(classes))], color=['skyblue', 'salmon', 'lightgreen'])
    plt.xlabel("Класи"); plt.ylabel("Кількість зображень"); plt.title("Розподіл класів у датасеті")
    plt.savefig("class_distribution.png")
    plt.close()

    fig, axes = plt.subplots(len(classes), 3, figsize=(9, 3 * len(classes)))
    for i, class_name in enumerate(classes):
        class_indices = [idx for idx, label in enumerate(labels) if label == i]
        selected_samples = random.sample(class_indices, min(3, len(class_indices)))
        
        for j, sample_idx in enumerate(selected_samples):
            img_path = file_paths[sample_idx]
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            ax = axes[i, j] if len(classes) > 1 else axes[j]
            ax.imshow(img); ax.set_title(f"{class_name} #{j+1}"); ax.axis('off')
            
    plt.tight_layout()
    plt.savefig("class_samples.png")
    print("Зразки класів збережено у файл 'class_samples.png'.")
    plt.close()

def main():
    file_paths, labels, classes = analyze_dataset(DATA_DIR)
    if not file_paths:
        print("Перевір, чи заповнена папка data/raw_dataset картинками!")
        return

    visualize_class_distribution_and_samples(file_paths, labels, classes)
    demonstrate_augmentations(file_paths[0])

    # 5. Стратифіковане розбиття Train / Validation / Test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        file_paths, labels, test_size=0.15, stratify=labels, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.176, stratify=y_train_val, random_state=SEED
    )

    print("\n=== РОЗБИТТЯ ДАТАСЕТУ ===")
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # 7. Збереження для Лаби №2
    np.savez("dataset_splits.npz", 
             X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val,
             X_test=X_test, y_test=y_test,
             classes=classes)
    print("\nУспішно! Файл 'dataset_splits.npz' та графіки згенеровано.")

if __name__ == "__main__":
    main()