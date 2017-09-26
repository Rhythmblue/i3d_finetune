from __init__ import label_file


def get_label_map(file):
    label_map = []
    with open(file) as f:
        for line in f.readlines():
            label_map.append(line.strip().split(' ')[1])
    return label_map


def trans_label(label_in):
    label_map = get_label_map(label_file)
    if isinstance(label_in, str):
        return label_map.index(label_in)
