import shutil
from pathlib import Path
import kagglehub

path = kagglehub.dataset_download("plameneduardo/sarscov2-ctscan-dataset")

ziel = Path("data")
ziel.mkdir(exist_ok=True)
shutil.copytree(path, ziel, dirs_exist_ok=True)