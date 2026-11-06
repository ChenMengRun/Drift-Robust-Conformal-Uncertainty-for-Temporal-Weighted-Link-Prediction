
### HMob provenance
HMob is one of the eight TWLP benchmark datasets released with the SALT
codebase [1]; it is the dataset reported as KAIST in the SALT paper (same
data, code-side name). Per [1], it contains human-mobility GPS traces
(positions updated every 30 s), and edge weights are W_uv = max(0, delta - d(u,v))
with delta = 250 --- consistent with the code's max_thres = 250. Statistics:
92 nodes, 500 snapshots. We download it from the link given in the SALT
repository and use the released snapshot pipeline unchanged; no preprocessing
of ours is applied.
