import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def visdrone2yolo(dir):
    def convert_box(size, box):
        # Convert VisDrone xywh to YOLO normalized cxcywh
        w, h = size
        x, y, bw, bh = box
        return (x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h

    # create labels directory
    label_dir = dir / 'labels'
    label_dir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm((dir / 'annotations').glob('*.txt'), desc=f'Converting {dir}')
    for f in pbar:
        img_file = (dir / 'images' / f.stem).with_suffix('.jpg')
        img_size = Image.open(img_file).size

        lines = []
        with open(f, 'r') as file:
            for row in [x.split(',') for x in file.read().strip().splitlines()]:
                # ignore class 0 in VisDrone
                if row[4] == '0':
                    continue

                cls = int(row[5]) - 1
                box = convert_box(img_size, tuple(map(int, row[:4])))
                lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")

        # write once per file
        out_path = label_dir / f.name
        with open(out_path, 'w') as fl:
            fl.writelines(lines)



dir = Path(r'/home/e706/zhanxiangning/newtask/powerdatasets/VisDrone-Det')
# Convert
for d in 'VisDrone2019-DET-train', 'VisDrone2019-DET-val', 'VisDrone2019-DET-test-dev':
    visdrone2yolo(dir / d)  # convert VisDrone annotations to YOLO labels