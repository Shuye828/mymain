#!/usr/bin/env python3
"""Run Main M1 disease-axis interpolation stages."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.evaluation.axis_interpolation import extract_oof,select_source,build_final_axis,extract_target,evaluate_targets
def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("command",choices=("oof-extract","select","build-final-axis","target-extract","target-evaluate")); p.add_argument("--config",type=Path,required=True); p.add_argument("--device",default="auto"); p.add_argument("--output-dir",type=Path); p.add_argument("--target"); p.add_argument("--max-batches",type=int); a=p.parse_args(); c=json.loads(a.config.read_text())
 if a.command=="oof-extract": r=extract_oof(c,device_request=a.device,output_override=a.output_dir,max_batches_per_fold=a.max_batches)
 elif a.command=="select": r=select_source(c,output_override=a.output_dir)
 elif a.command=="build-final-axis": r=build_final_axis(c,device_request=a.device,output_override=a.output_dir,max_batches=a.max_batches)
 elif a.command=="target-extract":
  if not a.target: p.error("--target is required")
  r=extract_target(c,target=a.target,device_request=a.device,output_override=a.output_dir,max_batches=a.max_batches)
 else: r=evaluate_targets(c,output_override=a.output_dir)
 print(json.dumps(r,ensure_ascii=False,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
