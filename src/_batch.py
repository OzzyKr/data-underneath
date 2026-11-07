import sys, time
from pathlib import Path
from src.run import process
from src.data import ROOT
names = sorted(p.stem for p in (ROOT / "bbox_annotation").glob("*.json"))
a, b = int(sys.argv[1]), int(sys.argv[2])
budget = time.time() + 120
with open("outputs/run_all.log", "a") as log:
    for i in range(a, min(b, len(names))):
        if time.time() > budget:
            print("STOP_AT", i); break
        try:
            dt = process(names[i], Path("outputs/results"))
            log.write(f"{i}\tOK\t{dt:.1f}\t{names[i]}\n"); log.flush()
        except Exception as e:
            log.write(f"{i}\tFAIL\t0\t{names[i]}\t{e}\n"); log.flush()
            print("FAIL", names[i], e)
    else:
        print("DONE_TO", min(b, len(names)))
print(len(names))
