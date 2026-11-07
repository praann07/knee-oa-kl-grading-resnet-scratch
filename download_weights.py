#!/usr/bin/env python
"""Download knee_oa_model.pt from the GitHub release (it is stored as 10 small parts) and join it.

Usage: python download_weights.py [--out knee_oa_model.pt]
Uses only the Python standard library and verifies the SHA-256 of the joined file.
"""
import argparse, hashlib, sys, urllib.request

BASE = "https://github.com/praann07/knee-oa-kl-grading-resnet-scratch/releases/download/v1.0.0/knee_oa_model.pt.part{:02d}"
N_PARTS = 10
SHA256 = "f1199e5fac8955d62bb02759325e042756dba9a1c99bcb230c6530b065931dcc"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="knee_oa_model.pt")
    out = ap.parse_args().out
    h = hashlib.sha256()
    with open(out, "wb") as f:
        for i in range(1, N_PARTS + 1):
            print(f"downloading part {i}/{N_PARTS} ...", flush=True)
            with urllib.request.urlopen(BASE.format(i), timeout=120) as r:
                data = r.read()
            h.update(data)
            f.write(data)
    if h.hexdigest() != SHA256:
        sys.exit("checksum mismatch: the download is corrupt, run again")
    print(f"ok -> {out}")


if __name__ == "__main__":
    main()
