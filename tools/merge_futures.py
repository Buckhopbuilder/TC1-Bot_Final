import csv, io, sys, zipfile
from pathlib import Path
import pandas as pd
import numpy as np

# We always output exactly these 6 columns:
KEEP_COLS = ["time","open","high","low","close","volume"]
TIME_MS_STEP = 5 * 60 * 1000  # 5 minutes in ms

# Full 12-col Binance schema (typical futures kline dump)
BINANCE_12 = [
    "open_time","open","high","low","close","volume",
    "close_time","quote_volume","trades",
    "taker_buy_volume","taker_buy_quote_volume","ignore"
]

# Header aliases if a header row exists
HEADER_ALIASES = {
    "time":   ["time","timestamp","open_time"],
    "open":   ["open","open_price"],
    "high":   ["high","high_price"],
    "low":    ["low","low_price"],
    "close":  ["close","close_price"],
    "volume": ["volume","base_volume","vol"],
}

def looks_like_numeric_row(first_line: str) -> bool:
    """
    Decide if the first non-empty line is a data row (headerless) e.g.
    1600844400000,3.9300,4.4124,3.9300,4.2393,67371,1600844699999,...
    """
    try:
        row = next(csv.reader([first_line]))
        if not row:
            return False
        # If first field is all digits (epoch ms) and row has at least 6 columns, assume data row
        return row[0].strip().isdigit() and len(row) >= 6
    except Exception:
        return False

def read_csv_from_zip(zp: Path, name: str) -> pd.DataFrame:
    """
    Read a CSV from a zip. If the file is headerless, assign the 12-col Binance header
    before parsing. Otherwise, let pandas read its header normally.
    """
    with zipfile.ZipFile(zp, "r") as z:
        with z.open(name) as fh:
            raw = fh.read()
            text = raw.decode("utf-8", errors="replace")
            # Peek first non-empty line
            first_line = ""
            for line in text.splitlines():
                if line.strip():
                    first_line = line
                    break

            if looks_like_numeric_row(first_line):
                # Headerless: inject the 12-col header, then the original text
                header_line = ",".join(BINANCE_12) + "\n"
                text = header_line + text
                return pd.read_csv(io.StringIO(text))
            else:
                # Has a header row already
                return pd.read_csv(io.StringIO(text))

def map_with_aliases(df: pd.DataFrame) -> pd.DataFrame:
    cols_lower = {c.strip().lower(): c for c in df.columns}
    mapped = {}

    def find_col(target):
        # exact match first
        for cand in HEADER_ALIASES[target]:
            if cand in cols_lower:
                return cols_lower[cand]
        return None

    # Try direct/alias mapping
    for target in KEEP_COLS:
        src = find_col(target)
        if src is None:
            raise ValueError(f"Missing required column '{target}'. Found: {list(cols_lower.keys())}")
        mapped[target] = src

    out = df[[mapped[c] for c in KEEP_COLS]].copy()
    out.columns = KEEP_COLS
    return out

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    # If the file was headerless, it now has the 12-col header BINANCE_12.
    # If it had a header, we rely on HEADER_ALIASES to map.
    # Special case: if we see 'open_time' but not 'time', we still map via aliases.

    # Try to map to KEEP_COLS via aliases
    out = map_with_aliases(df)

    # Types
    out["time"] = pd.to_numeric(out["time"], errors="coerce")
    for c in ["open","high","low","close","volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Clean
    out = out.dropna(subset=KEEP_COLS)
    out = out.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)

    # Drop obvious bad rows
    out = out[(out["open"]>0) & (out["high"]>0) & (out["low"]>0) & (out["close"]>0)]
    out = out[out["volume"]>=0]

    return out

def infer_symbol(zp: Path, inner_csv: str):
    # Typical inner names: SYMBOL-5m-YYYY-MM.csv (or similar)
    base = Path(inner_csv).name
    sym = None
    parts = base.replace(".csv","").split("-")
    if len(parts) >= 2 and parts[1].lower().startswith("5m"):
        sym = parts[0]
    if sym is None:
        # Fallback to zip filename
        zbase = zp.stem
        zparts = zbase.split("-")
        if zparts:
            sym = zparts[0]
    return sym

def collect_by_symbol(zips_dir: Path):
    buckets = {}
    zips = sorted(Path(zips_dir).glob("*.zip"))
    if not zips:
        print(f"No zips in {zips_dir}")
        sys.exit(1)

    for zp in zips:
        try:
            with zipfile.ZipFile(zp, "r") as z:
                for name in z.namelist():
                    if not name.lower().endswith(".csv"):
                        continue
                    sym = infer_symbol(zp, name)
                    if sym is None:
                        print(f"[WARN] Could not infer symbol for {zp.name}:{name}, skipping.")
                        continue
                    try:
                        raw = read_csv_from_zip(zp, name)
                        norm = normalize_df(raw)
                    except Exception as e:
                        print(f"[ERROR] {zp.name}:{name} → {e}")
                        continue
                    buckets.setdefault(sym, []).append(norm)
        except zipfile.BadZipFile:
            print(f"[WARN] Bad zip: {zp}")
    return buckets

def validate_spacing(df, symbol):
    diffs = df["time"].diff().dropna().values
    n_gap  = int(np.sum(diffs > TIME_MS_STEP))
    n_weird= int(np.sum((diffs != TIME_MS_STEP) & (diffs <= 0)))
    if n_gap or n_weird:
        print(f"[WARN] {symbol}: gaps={n_gap} weird_steps={n_weird}")
    return n_gap, n_weird

def merge_and_write(buckets, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for sym, frames in buckets.items():
        if not frames:
            continue
        merged = pd.concat(frames, ignore_index=True)
        merged = merged.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
        gaps, weird = validate_spacing(merged, sym)

        out_fp = out_dir / f"{sym}_5m.csv"
        merged.to_csv(out_fp, index=False)
        manifest.append({
            "symbol": sym,
            "rows": len(merged),
            "gaps_detected": gaps,
            "weird_steps": weird,
            "outfile": str(out_fp)
        })
        print(f"[OK] {sym}: rows={len(merged)} → {out_fp}")

    pd.DataFrame(manifest).to_csv(out_dir / "_manifest.csv", index=False)
    print(f"\nManifest: {out_dir / '_manifest.csv'}")

def main():
    in_dir  = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/futures_raw_zips")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/futures_merged_5m")
    buckets = collect_by_symbol(in_dir)
    if not buckets:
        print("No data collected.")
        sys.exit(1)
    merge_and_write(buckets, out_dir)

if __name__ == "__main__":
    main()
