import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fedsynai.experiments import run_all
o = run_all()
print(f"\n{'ID':<14}{'Acc':>7}{'F1':>7}{'AUC':>7}{'MIA':>7}{'eps':>8}")
for e in o["experiments"]:
    print(f"{e['id']:<14}{e['accuracy']:>7.3f}{e['f1']:>7.3f}{e['auc']:>7.3f}{e['mia_auc']:>7.3f}{(e['epsilon'] or 0):>8.1f}")
print(o["tstr"]); print(round(o["runtime_sec"]), "s")
