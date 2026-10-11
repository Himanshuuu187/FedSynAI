"""Quick sanity tests:  python tests_smoke.py"""
import numpy as np
from fedsynai.data import prepare, partition
from fedsynai.fl import train_federated
from fedsynai.privacy import epsilon

p = prepare(1, max_rows=4000)
iid = partition(p.y_train, 4, None); non = partition(p.y_train, 4, 1.0)
assert sorted(np.concatenate(iid)) == list(range(len(p.y_train))), "partition must cover every row exactly once"
assert abs(p.y_train[non[0]].mean() - p.y_train[non[1]].mean()) >= 0 
_, h, c = train_federated(p, iid, rounds=15)
assert h[-1]["auc"] > 0.8, h[-1]
assert epsilon(1.0, 10) < epsilon(1.0, 20) and epsilon(2.0, 10) < epsilon(1.0, 10)   # more rounds/less noise -> larger eps
print("all smoke tests passed")
