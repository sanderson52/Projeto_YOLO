#!/usr/bin/env python3
"""Seleciona e organiza um pequeno subset YOLO de apple e banana.
O dataset de origem é LVIS Fruits and Vegetables, distribuído no GitHub.
"""
from pathlib import Path
import zipfile, random, shutil, json, sys

ROOT = Path(__file__).resolve().parent
DL = ROOT / 'downloads'
DATA = ROOT / 'dataset'
SEED = 20261103
OLD_TO_NEW = {1: 0, 6: 1}  # origem: apple=1, banana=6


def unzip_once(zip_path, dest):
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / '.extracted'
    if marker.exists():
        return
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    marker.touch()


def locate_files(root, suffix):
    return {p.stem: p for p in root.rglob(f'*{suffix}') if p.is_file() and p.name != '.extracted'}


def labeled_candidates(image_dir, label_dir):
    imgs = locate_files(image_dir, '.jpg') | locate_files(image_dir, '.jpeg') | locate_files(image_dir, '.png')
    labs = locate_files(label_dir, '.txt')
    candidates = []
    for stem, image in imgs.items():
        label = labs.get(stem)
        if not label:
            continue
        rows = []
        for line in label.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                coords = list(map(float, parts[1:5]))
            except ValueError:
                continue
            if cls in OLD_TO_NEW and all(0 <= x <= 1 for x in coords):
                rows.append(f"{OLD_TO_NEW[cls]} " + ' '.join(f'{v:.6f}' for v in coords))
        if rows:
            found = {int(r.split()[0]) for r in rows}
            candidates.append((image, rows, found))
    return candidates


def choose_balanced(candidates, n, rng):
    """Greedy balanced selection; allows images containing both target classes."""
    pool = candidates[:]
    rng.shuffle(pool)
    selected, counts = [], {0: 0, 1: 0}
    while pool and len(selected) < n:
        pool.sort(key=lambda item: sum(counts[c] for c in item[2]))
        best = pool.pop(0)
        selected.append(best)
        for c in best[2]:
            counts[c] += 1
    return selected


def clear_split_outputs(split):
    for directory, suffixes in [
        (DATA/f'images/{split}', {'.jpg', '.jpeg', '.png', '.cache'}),
        (DATA/f'labels/{split}', {'.txt', '.cache'}),
    ]:
        directory.mkdir(parents=True, exist_ok=True)
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() in suffixes:
                path.unlink()


def main():
    random.seed(SEED)
    ex = ROOT / 'extracted'
    specs = [
      ('train_images.zip', ex/'train_images'), ('train_labels.zip', ex/'train_labels'),
      ('val_images.zip', ex/'val_images'), ('val_labels.zip', ex/'val_labels')]
    for name, dest in specs:
        archive = DL/name
        if not archive.exists():
            sys.exit(f'Arquivo ausente: {archive}')
        unzip_once(archive, dest)
    tr = labeled_candidates(ex/'train_images', ex/'train_labels')
    va = labeled_candidates(ex/'val_images', ex/'val_labels')
    print(f'Candidatas com apple/banana: treino={len(tr)}, validação={len(va)}')
    if len(tr) < 50 or len(va) < 10:
        sys.exit('Não há candidatos suficientes para o recorte mínimo planejado.')
    rng = random.Random(SEED)
    chosen_tr = choose_balanced(tr, 50, rng)
    chosen_va = choose_balanced(va, 10, rng)
    # Holdout próprio: cinco imagens adicionais do split de origem, nunca copiadas
    # para treino ou validação deste projeto.
    used = {i[0].stem for i in chosen_tr + chosen_va}
    holdout = [item for item in tr if item[0].stem not in used]
    chosen_te = choose_balanced(holdout, 5, rng)
    for split, items in [('train', chosen_tr), ('val', chosen_va), ('test', chosen_te)]:
        clear_split_outputs(split)
        for image, rows, _ in items:
            shutil.copy2(image, DATA/f'images/{split}'/image.name)
            (DATA/f'labels/{split}'/(image.stem+'.txt')).write_text('\n'.join(rows)+'\n')
    yaml = """path: dataset\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: apple\n  1: banana\nnc: 2\n"""
    (DATA/'data.yaml').write_text(yaml)
    counts = {}
    for split in ['train','val','test']:
        images = list((DATA/f'images/{split}').glob('*'))
        class_counts = [0, 0]
        boxes = 0
        for f in (DATA/f'labels/{split}').glob('*.txt'):
            for line in f.read_text().splitlines():
                if line.strip():
                    c = int(line.split()[0]); class_counts[c] += 1; boxes += 1
        counts[split] = {'images':len(images), 'boxes':boxes, 'apple_boxes':class_counts[0], 'banana_boxes':class_counts[1]}
    (ROOT/'dataset_stats.json').write_text(json.dumps(counts,indent=2))
    print(json.dumps(counts, indent=2))
    if counts['train']['images'] + counts['val']['images'] < 60 or counts['test']['images'] < 5:
        sys.exit('Falha: não cumpriu mínimo de 60 imagens de treino/val e cinco de teste.')

if __name__ == '__main__':
    main()
