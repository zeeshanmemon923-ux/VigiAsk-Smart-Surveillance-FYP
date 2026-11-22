"""Download a pretrained weapon model into models/weapon_yolo.pt.
Usage:  py -3.11 tests/download_weapon_model.py
        py -3.11 tests/download_weapon_model.py --repo Subh775/Threat-Detection-YOLOv8n --file best.pt
Needs:  pip install huggingface_hub
"""
import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

p = argparse.ArgumentParser()
p.add_argument("--repo", default="Subh775/Threat-Detection-YOLOv8n")
p.add_argument("--file", default="best.pt")
p.add_argument("--out", default="models/weapon_yolo.pt")
a = p.parse_args()

src = hf_hub_download(repo_id=a.repo, filename=a.file)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
shutil.copy(src, a.out)
print("Saved to", a.out)