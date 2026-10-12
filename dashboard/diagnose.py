"""Run from project root:  python dashboard/diagnose.py
Checks the Step 24 artifacts and prints each branch's probability for test sentences."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predictor import PROJECT_ROOT, Predictor, artifact_status  # noqa: E402

print("Artifacts:")
for name, path, ok in artifact_status():
    print(f"  [{'OK' if ok else 'MISSING'}] {name}: {path.relative_to(PROJECT_ROOT)}")

p = Predictor()
print("\nMode:", p.mode)
for n in p.notes:
    print(" -", n)
if p.mode != "hybrid":
    sys.exit(1)

tests = [
    "She is good girl",
    "Have a great day, congratulations on the new job!",
    "You are an idiot.",
    "நீ ஒரு முட்டாள்",
]
print(f"\n{'text':52s} {'xlmr':>7s} {'sem':>7s} {'hybrid':>7s}  label (thr={p.threshold:.2f})")
for t in tests:
    r = p.predict([t])[0]
    print(f"{t[:52]:52s} {r['xlmr_prob']:7.3f} {r['semantic_prob']:7.3f} {r['probability']:7.3f}  {r['label']}")
