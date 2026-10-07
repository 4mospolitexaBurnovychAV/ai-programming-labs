import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import onnx
from torchvision import models, transforms
from pathlib import Path
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader

# 1. Шляхи до моделі та тестового зображення
base_dir = Path(__file__).resolve().parent.parent
models_dir = base_dir / "models"
onnx_fp32_path = models_dir / "resnet18_custom.onnx"
onnx_int8_path = models_dir / "resnet18_int8.onnx"

# Знайдемо наше тестове зображення з 3-ї лаби
image_path = base_dir / "lab03_computer_vision" / "ziba.jpg"
if not image_path.exists():
    # Fallback якщо шлях інший
    image_path = list(base_dir.rglob("*.jpg"))[0]

print(f"Використовується зображення: {image_path}")

# 2. Калібрувальний дата-рідер для INT8 квантування (Post-Training Quantization)
class ResNetCalibrationDataReader(CalibrationDataReader):
    def __init__(self):
        self.enum_data = []
        # Генеруємо кілька калібрувальних батчів на основі випадкових/реальних даних
        for _ in range(10):
            sample = np.random.rand(1, 3, 224, 224).astype(np.float32)
            self.enum_data.append({'input': sample})
        self.datasize = len(self.enum_data)

    def get_next(self):
        if len(self.enum_data) > 0:
            return self.enum_data.pop(0)
        return None

# Квантування моделі з FP32 у INT8 (якщо ще не квантована)
if not onnx_int8_path.exists():
    print("Виконується квантування моделі у формат INT8 (Post-Training Quantization)...")
    dr = ResNetCalibrationDataReader()
    quantize_static(
        model_input=onnx.load(str(onnx_fp32_path)),
        model_output=str(onnx_int8_path),
        calibration_data_reader=dr,
        weight_type=QuantType.QInt8
    )
print(f"Квантовану модель збережено: {onnx_int8_path}")  # Убедись, что здесь нет лишнего отступа слева

# Порівняння розмірів файлів
fp32_size_kb = onnx_fp32_path.stat().st_size / 1024
int8_size_kb = onnx_int8_path.stat().st_size / 1024
print(f"Розмір FP32 моделі: {fp32_size_kb:.2f} КБ")
print(f"Розмір INT8 моделі: {int8_size_kb:.2f} КБ")

# 3. Ініціалізація ONNX Runtime сесій для FP32 та INT8
session_fp32 = ort.InferenceSession(str(onnx_fp32_path), providers=['CPUExecutionProvider'])
session_int8 = ort.InferenceSession(str(onnx_int8_path), providers=['CPUExecutionProvider'])

input_name_fp32 = session_fp32.get_inputs()[0].name
output_name_fp32 = session_fp32.get_outputs()[0].name

input_name_int8 = session_int8.get_inputs()[0].name
output_name_int8 = session_int8.get_outputs()[0].name

# 4. Профілювання повного конвеєра (Preprocessing -> Inference -> Postprocessing)
print("\nПрофілювання етапів конвеєра...")

def profile_pipeline(session, input_name, output_name, is_int8=False):
    # Зчитування кадру (OpenCV + Unicode безпека)
    file_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    
    t0 = time.perf_counter()
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Preprocessing
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    # Нормалізація ImageNet
    img_normalized = img_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_normalized = (img_normalized - mean) / std
    img_tensor = img_normalized.transpose(2, 0, 1)[np.newaxis, :, :, :]
    
    if is_int8:
        # Для INT8 вхідні дані часто приводяться до uint8 або залишаються float залежно від конфігурації, 
        # ONNX Runtime автоматично керує кастомними типами квантованих вузлів.
        pass

    t1 = time.perf_counter()
    
    # Inference
    outputs = session.run([output_name], {input_name: img_tensor})[0]
    
    t2 = time.perf_counter()
    
    # Postprocessing (Softmax + Argmax)
    exp_outs = np.exp(outputs - np.max(outputs))
    probabilities = exp_outs / np.sum(exp_outs, axis=1, keepdims=True)
    predicted_class = np.argmax(probabilities, axis=1)[0]
    confidence = np.max(probabilities, axis=1)[0]
    
    t3 = time.perf_counter()
    
    prep_time = (t1 - t0) * 1000
    infer_time = (t2 - t1) * 1000
    post_time = (t3 - t2) * 1000
    total_time = (t3 - t0) * 1000
    
    return prep_time, infer_time, post_time, total_time, predicted_class, confidence

# Профілювання FP32
p_fp32, i_fp32, po_fp32, tot_fp32, cls_f, conf_f = profile_pipeline(session_fp32, input_name_fp32, output_name_fp32, is_int8=False)
# Профілювання INT8
p_int8, i_int8, po_int8, tot_int8, cls_i, conf_i = profile_pipeline(session_int8, input_name_int8, output_name_int8, is_int8=True)

print(f"\n--- Резюме профілювання (FP32) ---")
print(f"Preprocessing: {p_fp32:.2f} мс | Inference: {i_fp32:.2f} мс | Postprocessing: {po_fp32:.2f} мс | Всього: {tot_fp32:.2f} мс")

print(f"\n--- Резюме профілювання (INT8 Quantized) ---")
print(f"Preprocessing: {p_int8:.2f} мс | Inference: {i_int8:.2f} мс | Postprocessing: {po_int8:.2f} мс | Всього: {tot_int8:.2f} мс")