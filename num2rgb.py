"""
MS COCO のカテゴリでのセグメンテーションの色の指定をjsonファイルに出力する。

https://github.com/Mikubill/sd-webui-controlnet/discussions/503
"""

import csv
import json

d = {}
d2 = {}
with open("T2i_Segmentation_Color_Ref_v21.csv") as f:
    reader = csv.reader(f)
    i = -1
    for row in reader:
        i += 1
        if i > 4 and row[0] != "":
            print(row)
            num, name, _, r, g, b, hex_str = row
            d[int(num)] = (int(r), int(g), int(b))
            d2[name] = (int(r), int(g), int(b))
print(d)

with open('coco_color.json', 'w') as f:
    json.dump(d, f)

with open('coco_color_name.json', 'w') as f:
    json.dump(d2, f)
