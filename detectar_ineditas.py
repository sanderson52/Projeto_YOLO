#!/usr/bin/env python3
"""Executa inferência nas imagens inéditas reservadas e salva as detecções."""
from pathlib import Path
import csv, json
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
model = YOLO(str(ROOT/'modelo'/'best.pt'))
source = ROOT/'dataset'/'images'/'test'
results = model.predict(source=str(source), imgsz=320, conf=0.20, iou=0.45,
                        device='cpu', save=True, project=str(ROOT/'evidencias'),
                        name='deteccoes_imagens_ineditas', exist_ok=True, verbose=True)
rows=[]
for result in results:
    p=Path(result.path)
    det=[]
    if result.boxes is not None:
        for box in result.boxes:
            cls=int(box.cls.item()); conf=float(box.conf.item())
            xyxy=[round(float(v),2) for v in box.xyxy[0].tolist()]
            det.append({'classe':result.names[cls], 'confianca':round(conf,4), 'xyxy_pixels':xyxy})
            rows.append({'imagem':p.name,'classe':result.names[cls],'confianca':round(conf,4),'x1':xyxy[0],'y1':xyxy[1],'x2':xyxy[2],'y2':xyxy[3]})
    print(json.dumps({'imagem':p.name,'deteccoes':det},ensure_ascii=False))
with (ROOT/'deteccoes.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=['imagem','classe','confianca','x1','y1','x2','y2'])
    w.writeheader(); w.writerows(rows)
(ROOT/'deteccoes.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False))
print(f'Imagens avaliadas: {len(results)}; detecções: {len(rows)}')
