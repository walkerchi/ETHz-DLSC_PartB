import os 
import shutil
from glob import glob
from tqdm import tqdm

LATEX_REPO = "../PartB-report"
MATCH = r"*.pdf"

if __name__ == '__main__':
    for srcpath in tqdm(glob(f"images/**/{MATCH}", recursive=True)):
        dstpath = os.path.join(LATEX_REPO, srcpath)
        os.makedirs(os.path.dirname(dstpath), exist_ok=True)        
        shutil.copy(srcpath, dstpath)