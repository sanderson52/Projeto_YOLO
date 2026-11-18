#!/usr/bin/env python3
"""Baixa arquivos-fonte do subset público usado pelo projeto (aprox. 1,4 GB)."""
from pathlib import Path
from urllib.request import urlopen, Request
import shutil

ROOT = Path(__file__).resolve().parent
DL = ROOT / 'downloads'
DL.mkdir(exist_ok=True)
BASE = 'https://media.githubusercontent.com/media/henningheyen/Fruits-And-Vegetables-Detection-Dataset/main/LVIS_Fruits_And_Vegetables'
FILES = {
    'train_images.zip': f'{BASE}/images/train/train.zip',
    'train_labels.zip': f'{BASE}/labels/train/train.zip',
    'val_images.zip': f'{BASE}/images/val/val.zip',
    'val_labels.zip': f'{BASE}/labels/val/val.zip',
    'original_data.yaml': 'https://raw.githubusercontent.com/henningheyen/Fruits-And-Vegetables-Detection-Dataset/main/LVIS_Fruits_And_Vegetables/data.yaml',
    'LICENSE_dataset.txt': 'https://raw.githubusercontent.com/henningheyen/Fruits-And-Vegetables-Detection-Dataset/main/LICENSE',
}
for name, url in FILES.items():
    out = ROOT / name if name == 'LICENSE_dataset.txt' else DL / name
    if out.exists() and out.stat().st_size:
        print(f'Já existe: {out}')
        continue
    print(f'Baixando: {url} -> {out}')
    req = Request(url, headers={'User-Agent': 'YOLO-academic-project'})
    with urlopen(req, timeout=60) as response, out.open('wb') as f:
        shutil.copyfileobj(response, f, 1024 * 1024)
    print(f'Concluído: {out.stat().st_size:,} bytes')
