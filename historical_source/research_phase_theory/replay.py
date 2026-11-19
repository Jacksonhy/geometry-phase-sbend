"""Reproduce into a NEW directory without overwriting frozen analysis."""
import argparse,shutil,sys
from pathlib import Path
SOURCE=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
 dest=args.out.resolve()
 if dest.exists():raise RuntimeError('Replay destination must not exist; preserve all previous results')
 dest.mkdir(parents=True)
 for item in SOURCE.iterdir():
  if item.suffix in ['.py','.md']:shutil.copy2(item,dest/item.name)
 sys.path.insert(0,str(dest))
 import analyze_phase as analysis
 analysis.OUT=dest
 analysis.main()
 import modal_feasibility as modal
 modal.main()
 import summarize_and_plot as post
 post.main()
 import build_report as report
 report.main()
 import finalize as finish
 finish.main()
if __name__=='__main__':main()
