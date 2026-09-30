"""Write Pl@ntNet-300K split files in the "relpath label" format read by datasets/herbarium.py.

Download plantnet_300K.zip from https://zenodo.org/records/5645731, unzip it, then:

    python tools/make_plantnet_splits.py --root /path/to/plantnet_300K

This writes plantnet_{train,val,test}.txt and plantnet_label_map.json into --root.
"""
import argparse
import json
import os

import numpy as np

IMG_EXTS = ('.jpg', '.jpeg', '.png')


def list_split(root, split):
    """{species_id: [image file names]} for images/<split>/<species_id>/*."""
    split_dir = os.path.join(root, 'images', split)
    files = {}
    for species_id in sorted(os.listdir(split_dir)):
        class_dir = os.path.join(split_dir, species_id)
        if os.path.isdir(class_dir):
            files[species_id] = sorted(f for f in os.listdir(class_dir) if f.lower().endswith(IMG_EXTS))
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', required=True, help='unzipped plantnet_300K folder (contains images/)')
    parser.add_argument('--many_thr', default=100, type=int, help='many-shot: more train images than this')
    parser.add_argument('--few_thr', default=20, type=int, help='few-shot: fewer train images than this')
    args = parser.parse_args()

    if not os.path.isdir(os.path.join(args.root, 'images', 'train')):
        raise SystemExit("'{}' has no images/train folder".format(args.root))

    train = list_split(args.root, 'train')
    species = sorted(train)
    label_of = {species_id: label for label, species_id in enumerate(species)}

    for split in ('train', 'val', 'test'):
        if not os.path.isdir(os.path.join(args.root, 'images', split)):
            print('skipping {}: images/{} not found'.format(split, split))
            continue
        files = train if split == 'train' else list_split(args.root, split)
        unknown = sorted(set(files) - set(label_of))
        if unknown:
            print('warning: {} {} classes have no train images and are skipped'.format(len(unknown), split))
        out = os.path.join(args.root, 'plantnet_{}.txt'.format(split))
        n = 0
        with open(out, 'w') as f:
            for species_id in sorted(set(files) & set(label_of)):
                for name in files[species_id]:
                    f.write('images/{}/{}/{} {}\n'.format(split, species_id, name, label_of[species_id]))
                    n += 1
        print('{}: {} images -> {}'.format(split, n, out))

    names_file = os.path.join(args.root, 'plantnet300K_species_id_2_name.json')
    names = json.load(open(names_file)) if os.path.isfile(names_file) else {}
    label_map = {label: {'species_id': s, 'name': names.get(s)} for label, s in enumerate(species)}
    with open(os.path.join(args.root, 'plantnet_label_map.json'), 'w') as f:
        json.dump(label_map, f, indent=1)

    counts = np.array([len(train[s]) for s in species])
    print('classes: {}  train images per class: min {}  median {}  max {}'.format(
        len(counts), counts.min(), int(np.median(counts)), counts.max()))
    print('many-shot (>{}): {}  medium-shot: {}  few-shot (<{}): {}'.format(
        args.many_thr, (counts > args.many_thr).sum(),
        ((counts >= args.few_thr) & (counts <= args.many_thr)).sum(),
        args.few_thr, (counts < args.few_thr).sum()))


if __name__ == '__main__':
    main()
