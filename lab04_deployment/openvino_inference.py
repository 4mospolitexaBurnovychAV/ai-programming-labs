import time
import numpy as np
from pathlib import Path
from openvino import Core

# 1. Ініціалізація OpenVINO Runtime Core
core = Core()

# Шлях до експортованої ONNX моделі
models_dir = Path(__file__).resolve().parent.parent / "models"
onnx_model_path = models_dir / "resnet18_custom.onnx"

if not onnx_model_path.exists():
    raise FileNotFoundError(f"Не знайдено модель за шляхом: {onnx_model_path}")

print(f"Читання та компіляція моделі через OpenVINO із: {onnx_model_path}")

# 2. Читання моделі та компіляція під пристрій (CPU за замовчуванням)
model = core.read_model(model=str(onnx_model_path))
compiled_model = core.compile_model(model=model, device_name="CPU")

# Отримуємо об'єкти входу та виходу
input_key = compiled_model.input(0)
output_key = compiled_model.output(0)

# 3. Підготовка тестових даних (dummy input у форматі NCHW)
batch_size = 1
input_shape = (batch_size, 3, 224, 224)
dummy_input = np.random.rand(*input_shape).astype(np.float32)

print("Вимірювання продуктивності OpenVINO (Benchmark)...")

# 4. Прогрів (Warm-up)
for _ in range(10):
    _ = compiled_model([dummy_input])

# 5. Бенчмарк на 100 ітераціях
num_iterations = 100
start_time = time.perf_counter()

for _ in range(num_iterations):
    _ = compiled_model([dummy_input])

end_time = time.perf_counter()

total_time = end_time - start_time
avg_latency_ms = (total_time / num_iterations) * 1000
fps = num_iterations / total_time

print(f"OpenVINO - Середній час inference (Latency): {avg_latency_ms:.2f} мс")
print(f"OpenVINO - Пропускна здатність (Throughput / FPS): {fps:.2f} кадрів/сек")