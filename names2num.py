from pathlib import Path
from typing import Dict

def get_label2num_dict() -> Dict[str, int]:
    namelist = Path("coconames.txt")

    d = {}
    for i, line in enumerate(namelist.open("rt")):
        line = line.strip()
        d[line] = i+1

    return d

if __name__ == "__main__":
    d = get_label2num_dict()
    print(d)
