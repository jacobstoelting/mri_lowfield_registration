"""Download one subject's scans from the MPI-Leipzig Mind-Brain-Body dataset (OpenNeuro ds000221, CC0)."""
import urllib.request
from pathlib import Path

BASE = "https://s3.amazonaws.com/openneuro.org/ds000221/sub-010001/ses-02/anat/"
FILES = [
    "sub-010001_ses-02_acq-mp2rage_T1w.nii.gz",  # 3T T1-weighted, 1 mm voxels
    "sub-010001_ses-02_inv-2_mp2rage.nii.gz",    # used only to mask the T1w background
    "sub-010001_ses-02_acq-lowres_FLAIR.nii.gz", # FLAIR, 0.43 x 0.43 x 5 mm slices
]

out = Path("data")
out.mkdir(exist_ok=True)
for name in FILES:
    dest = out / name
    if dest.exists():
        print(f"have {name}")
        continue
    print(f"downloading {name}")
    urllib.request.urlretrieve(BASE + name, dest)
