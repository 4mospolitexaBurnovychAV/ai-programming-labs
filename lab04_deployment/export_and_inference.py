import torch
import torch.nn as nn
from torchvision import models
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import time

# 1. Завантаження моделі та адаптація (як у Лаб №3)
device = torch.device("cpu") # Експорт зазвичай виконується на CPU
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

num_classes = 3
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.eval()

# 2. Підготовка фіктивного входу (dummy input) для експорту
dummy_input = torch.randn(1, 3, 224, 224, device=device)

# Шлях для збереження ONNX моделі
models_dir = Path(__file__).resolve().parent.parent / "models"
models_dir.mkdir(exist_ok=True)
onnx_model_path = models_dir / "resnet18_custom.onnx"

print("Експорт моделі у формат ONNX...")
torch.onnx.export(
    model,
    dummy_input,
    str(onnx_model_path),
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
print(f"Модель успішно збережено за шляхом: {onnx_model_path}")

# 3. Перевірка валідності ONNX моделі
onnx_model = onnx.load(str(onnx_model_path))
onnx.checker.check_model(onnx_model)
print("ONNX модель пройшла перевірку валідності!")

# Виведення інформації про входи та виходи
session = ort.InferenceSession(str(onnx_model_path), providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print(f"Input name: {input_name}, shape: {session.get_inputs()[0].shape}, type: {session.get_inputs()[0].type}")
print(f"Output name: {output_name}, shape: {session.get_outputs()[0].shape}")

# 4. Порівняння результатів PyTorch та ONNX Runtime
np_input = dummy_input.numpy()

# PyTorch inference
with torch.no_grad():
    pytorch_output = model(dummy_input).numpy()

# ONNX Runtime inference
onnx_output = session.run([output_name], {input_name: np_input})[0]

# Обчислення максимального відхилення
max_diff = np.max(np.abs(pytorch_output - onnx_output))
print(f"Максимальне числове відхилення між PyTorch та ONNX Runtime: {max_diff:.6f}")

# 5. Бенчмарк ONNX Runtime після warm-up
print("Вимірювання продуктивності ONNX Runtime (Benchmark)...")
for _ in range(10): # Warm-up
    _ = session.run([output_name], {input_name: np_input})

num_iterations = 100
start_time = time.perf_counter()
for _ in range(num_iterations):
    _ = session.run([output_name], {input_name: np_input})
end_time = time.perf_counter()

total_time = end_time - start_time
avg_latency_ms = (total_time / num_iterations) * 1000
fps = num_iterations / total_time

print(f"ONNX Runtime - Середній час inference (Latency): {avg_latency_ms:.2f} мс")
print(f"ONNX Runtime - Пропускна здатність (Throughput / FPS): {fps:.2f} кадрів/сек")