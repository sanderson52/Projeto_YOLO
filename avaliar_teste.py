#!/usr/bin/env python3
"""Calcula métricas do checkpoint nas cinco imagens inéditas reservadas."""
from pathlib import Path
import json
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parent
model=YOLO(str(ROOT/'modelo'/'best.pt'))
test_images=ROOT/'dataset'/'images'/'test'
r=model.val(data=str(ROOT/'dataset'/'data.yaml'),split='test',imgsz=320,batch=4,device='cpu',
            project=str(ROOT/'runs'),name='teste_holdout',exist_ok=True,plots=True,verbose=True)
metrics={
 'precision':float(r.results_dict.get('metrics/precision(B)',float('nan'))),
 'recall':float(r.results_dict.get('metrics/recall(B)',float('nan'))),
 'mAP50':float(r.results_dict.get('metrics/mAP50(B)',float('nan'))),
 'mAP50_95':float(r.results_dict.get('metrics/mAP50-95(B)',float('nan'))),
 'holdout_images':sum(1 for path in test_images.iterdir()
                      if path.suffix.lower() in {'.jpg','.jpeg','.png'}),
}
path=ROOT/'metricas_teste.json'; path.write_text(json.dumps(metrics,indent=2,ensure_ascii=False))
print(json.dumps(metrics,indent=2,ensure_ascii=False))
