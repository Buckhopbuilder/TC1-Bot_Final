import csv, io, sys, zipfile, hashlib
from pathlib import Path

OUT_DIR = Path("reports/data_audit")

def header_signature(cols):
    norm = [c.strip().lower() for c in cols]
    sig = "|".join(norm)
    return hashlib.sha1(sig.encode("utf-8")).hexdigest(), norm

def sniff_header_from_bytes(b):
    text = b.decode("utf-8", errors="replace")
    for line in text.splitlines():
        row = next(csv.reader([line]))
        if row and any(cell.strip() for cell in row):
            return row
    return []

def inspect_zip_folder(zips_dir: Path):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detail_rows = []
    signature_counts = {}

    zip_paths = sorted(Path(zips_dir).glob("*.zip"))
    if not zip_paths:
        print(f"No .zip files found in: {zips_dir}")
        sys.exit(1)

    for zp in zip_paths:
        try:
            with zipfile.ZipFile(zp, "r") as z:
                for name in z.namelist():
                    if not name.lower().endswith(".csv"):
                        continue
                    with z.open(name) as fh:
                        chunk = fh.read(4096)
                        header = sniff_header_from_bytes(chunk)
                        sig_hash, norm_cols = header_signature(header)
                        detail_rows.append({
                            "zip_file": zp.name,
                            "inner_csv": name,
                            "n_cols": len(header),
                            "columns_raw": ",".join(header),
                            "columns_norm": ",".join(norm_cols),
                            "sig_hash": sig_hash,
                        })
                        signature_counts.setdefault(sig_hash, {"count": 0, "columns_norm": norm_cols})
                        signature_counts[sig_hash]["count"] += 1
        except zipfile.BadZipFile:
            print(f"[WARN] Bad zip: {zp}")

    # Write detail
    detail_fp = OUT_DIR / "headers_detail.csv"
    with detail_fp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["zip_file","inner_csv","n_cols","columns_raw","columns_norm","sig_hash"])
        w.writeheader()
        w.writerows(detail_rows)

    # Write signatures
    sig_fp = OUT_DIR / "header_signatures.csv"
    with sig_fp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sig_hash","count","columns_norm"])
        for h, info in signature_counts.items():
            w.writerow([h, info["count"], ",".join(info["columns_norm"])])

    # Console summary
    print("\n=== Header Signature Summary ===")
    for h, info in signature_counts.items():
        print(f"- {h[:8]}…  count={info['count']}  cols=[{', '.join(info['columns_norm'])}]")
    print(f"\nDetails: {detail_fp}")
    print(f"Signatures: {sig_fp}")

def main():
    zips_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("binance_zips")
    inspect_zip_folder(zips_dir)

if __name__ == "__main__":
    main()
