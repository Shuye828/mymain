#!/usr/bin/env python3
"""Prepare, extract, and analyze Revision R3 A-D."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.analysis.revision_r3_afdb_mechanism import prepare, extract, analyze

def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare","extract","analyze","all"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-batches-per-dataset", type=int)
    args=parser.parse_args(); cfg=json.loads(args.config.read_text())
    results=[]
    if args.command in ("prepare","all"): results.append(prepare(cfg, output_override=args.output_dir))
    if args.command in ("extract","all"): results.append(extract(cfg, device_request=args.device, output_override=args.output_dir, max_batches_per_dataset=args.max_batches_per_dataset))
    if args.command in ("analyze","all"): results.append(analyze(cfg, output_override=args.output_dir))
    print(json.dumps(results[-1], ensure_ascii=False, indent=2)); return 0
if __name__ == "__main__": raise SystemExit(main())
