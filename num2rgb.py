import csv
import json

d = {}
with open("T2i_Segmentation_Color_Ref_v21.csv") as f:
    reader = csv.reader(f)
    i = -1
    for row in reader:
        i += 1
        if i > 4 and row[0] != "":
            print(row)
            num, name, _, r, g, b, hex_str = row
            d[int(num)] = (int(r), int(g), int(b))

print(d)

with open('segmentation.json', 'w') as f:
    json.dump(d, f)