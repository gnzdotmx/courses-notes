"""Check / prepare local LFW cache for fetch_lfw_people.

Sklearn downloads from figshare on first run. If you get HTTP 403, download
the four files in a browser and place them in lfw_home/ (see print instructions).
"""

from pathlib import Path
from sklearn.datasets._lfw import FUNNELED_ARCHIVE, TARGETS

LFW_HOME = Path(__file__).resolve().parent / "data" / "sklearn_cache" / "lfw_home"

REQUIRED = [t.filename for t in TARGETS] + [FUNNELED_ARCHIVE.filename]


def main() -> None:
    LFW_HOME.mkdir(parents=True, exist_ok=True)
    missing = [name for name in REQUIRED if not (LFW_HOME / name).exists()]
    extracted = (LFW_HOME / "lfw_funneled").exists()

    if not missing and extracted:
        print(f"LFW cache ready: {LFW_HOME}")
        return

    print("LFW cache incomplete. fetch_lfw_people needs these files:\n")
    print(f"  Directory: {LFW_HOME}\n")
    for target in TARGETS:
        mark = "OK" if (LFW_HOME / target.filename).exists() else "MISSING"
        print(f"  [{mark}] {target.filename}")
        print(f"         {target.url}\n")
    mark = "OK" if (LFW_HOME / FUNNELED_ARCHIVE.filename).exists() else "MISSING"
    print(f"  [{mark}] {FUNNELED_ARCHIVE.filename}  (~200 MB, funneled images)")
    print(f"         {FUNNELED_ARCHIVE.url}\n")
    if not extracted:
        print("  [MISSING] lfw_funneled/  (created automatically after tgz is present)\n")

    print("Download MISSING files in your browser (figshare may block curl/scripts).")
    print("Then run: python code/mod4/FacialRecognitionEigenfaces.py")


if __name__ == "__main__":
    main()
