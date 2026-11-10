"""Download the Leiden paired 64 mT / 3T brain MRI dataset (Zenodo record 20281403, MIT license).

van den Broek, Lena & Webb, "Paired 64mT and 3T Brain MRI Scans of Healthy Subjects for
Neuroimaging Research", https://doi.org/10.5281/zenodo.20281403

86 real low-field scans (Hyperfine Swoop, 64 mT); 11 of the volunteers were also scanned at 3T.
Each session has T1-weighted, T2-weighted, FLAIR and diffusion scans.

Run from the repo root:  python download_data.py
Downloads one 1.4 GB zip into data/ and unpacks it to data/leiden/.
"""
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

RECORD = "https://zenodo.org/records/20281403/files/"
ZIP_NAME = "v5 Paired 64mT and 3T Brain MRI Scans of Healthy Subjects for Neuroimaging Research.zip"
PDF_NAME = "Data descriptor to Paired 64mT and 3T Brain MRI Scans.pdf"

DATA = Path("data")
OUT = DATA / "leiden"


def fetch(name):
    dest = DATA / name
    if dest.exists():
        print(f"have {dest}")
        return dest
    url = RECORD + urllib.parse.quote(name) + "?download=1"
    print(f"downloading {name}")

    def progress(blocks, block_size, total):
        if total > 0:
            done = min(blocks * block_size, total)
            sys.stdout.write(f"\r  {done / 1e6:7.0f} / {total / 1e6:.0f} MB")
            sys.stdout.flush()

    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp, progress)
    tmp.rename(dest)  # only a complete download gets the real name
    print()
    return dest


def main():
    DATA.mkdir(exist_ok=True)
    fetch(PDF_NAME)
    zip_path = fetch(ZIP_NAME)
    if not OUT.exists():
        print(f"unzipping to {OUT}")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(OUT)
    n = sum(1 for _ in OUT.rglob("*") if _.is_file())
    print(f"{OUT}: {n} files")


if __name__ == "__main__":
    main()
