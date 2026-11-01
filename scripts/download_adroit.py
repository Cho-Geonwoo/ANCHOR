"""Download the public WSRL/Cal-QL Adroit dataset into a chosen directory."""

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

FILE_ID = "1yUdJnGgYit94X_AvV6JJP5Y3Lx2JF30Y"
REQUIRED = ("pen2_sparse.npy", "door2_sparse.npy", "pen_bc_sparse4.npy", "door_bc_sparse4.npy")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data/adroit"))
    args = p.parse_args()
    target = args.data_dir.resolve() / "offpolicy_hand_data"
    if all((target / n).is_file() for n in REQUIRED):
        print(f"Adroit files already present: {target}")
        return
    import gdown

    with tempfile.TemporaryDirectory(prefix="anchor_adroit_") as tmp:
        archive = Path(tmp) / "adroit.zip"
        if not gdown.download(id=FILE_ID, output=str(archive), quiet=False):
            raise RuntimeError("Download failed; use the dataset link in README.md")
        with zipfile.ZipFile(archive) as z:
            candidates = {}
            for info in z.infolist():
                name = Path(info.filename).name
                if name in REQUIRED:
                    if name in candidates:
                        raise ValueError(f"Ambiguous dataset member: {name}")
                    candidates[name] = info
            if set(candidates) != set(REQUIRED):
                raise ValueError("Dataset archive lacks one or more Figure 2 demonstration files")
            target.mkdir(parents=True, exist_ok=True)
            for name, info in candidates.items():
                if (target / name).exists():
                    continue
                with z.open(info) as source, (target / name).open("wb") as dest:
                    shutil.copyfileobj(source, dest)
        manifest = dict(
            source=f"https://drive.google.com/file/d/{FILE_ID}/view",
            archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            files={n: hashlib.sha256((target / n).read_bytes()).hexdigest() for n in REQUIRED},
        )
        (target.parent / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Adroit dataset ready: {target}")


if __name__ == "__main__":
    main()
