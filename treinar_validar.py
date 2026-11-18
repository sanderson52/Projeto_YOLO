#!/usr/bin/env python3
"""Treina YOLOv8n por 20 épocas e registra métricas de validação."""
from pathlib import Path
import json, shutil
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT/'dataset'/'data.yaml'
RUNS = ROOT/'runs'
MODEL_DIR = ROOT/'modelo'
MODEL_DIR.mkdir(exist_ok=True)

model = YOLO('yolov8n.pt')  # pesos pré-treinados como ponto de partida
train_result = model.train(data=str(DATA), epochs=20, imgsz=320, batch=4,
                           device='cpu', workers=0, project=str(RUNS),
                           name='apple_banana_yolov8n', exist_ok=True,
                           patience=20, plots=True, verbose=True, seed=20261103)
best = RUNS/'apple_banana_yolov8n'/'weights'/'best.pt'
if not best.exists():
    raise FileNotFoundError(f'best.pt não foi salvo em {best}')
shutil.copy2(best, MODEL_DIR/'best.pt')

trained = YOLO(str(best))
val = trained.val(data=str(DATA), split='val', imgsz=320, batch=4, device='cpu',
                 project=str(RUNS), name='validacao_final', exist_ok=True,
                 plots=True, verbose=True)
metrics = {
  'precision': float(val.results_dict.get('metrics/precision(B)', float('nan'))),
  'recall': float(val.results_dict.get('metrics/recall(B)', float('nan'))),
  'mAP50': float(val.results_dict.get('metrics/mAP50(B)', float('nan'))),
  'mAP50_95': float(val.results_dict.get('metrics/mAP50-95(B)', float('nan'))),
  'epochs_configuradas': 20,
  'model': 'YOLOv8n pretrained',
  'device': 'CPU',
  'imgsz': 320,
  'best_weights': str((MODEL_DIR/'best.pt').relative_to(ROOT)),
}
(ROOT/'metricas_validacao.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
print(json.dumps(metrics, indent=2, ensure_ascii=False))
