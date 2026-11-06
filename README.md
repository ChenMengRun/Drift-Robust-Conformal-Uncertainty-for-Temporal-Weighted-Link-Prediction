# Drift-Robust Conformal Uncertainty for Temporal Weighted Link Prediction

Code, prediction caches and result files for the ICEICT 2026 paper
"Drift-Robust Conformal Uncertainty for Temporal Weighted Link Prediction".

The paper asks a simple question on the SALT temporal weighted link prediction
benchmark: when the edge-weight distribution drifts, does static split conformal
calibration still hold? In our experiments it does not -- and a block-level
adaptive conformal wrapper, wrapping a small family of nonconformity scores with
a validation-only selector, brings coverage back online.

## Layout

    SALT/            backbone + probabilistic head. Based on the official SALT
                     codebase (Li et al., CIKM 2025, github.com/liting1024/SALT);
                     the probabilistic heads and the training entry are ours.
      config/        one yaml per training run (131 files)
      model/         official SALT model plus our probabilistic heads
      dataset/, utils/
      inn_train.py   single training entry for every configuration
    Exp/             evaluation pipeline; everything here runs on CPU
      cache/         per-edge (mu, sigma, y) caches for every run in the paper.
                     With these you can reproduce all numbers without training.
      calib/         result CSVs + facts.json -- where the paper's numbers live
      *.py           see below
    figures/         Figure 1 (png + the script that draws it from the CSVs)

    requirements.txt

Key evaluation scripts:

    aci_wrap.py            block-level ACI wrapper, all score members (seed 0)
    aci_wrap_seeds.py      same, over all seeds -> Table II (seed medians)
    block_bootstrap.py     Table I: static vs online coverage + bootstrap CIs
    family_select.py       validation-only member selection
    enbpi_faithful.py      EnbPI baseline (faithful instantiation)
    tail_true_bins.py      tail coverage on the true top decile
    gpd_shape.py           GPD shape parameters (Section III)
    evt_static_tail.py     static GPD tail gain (Section V-C)
    persistence_edge.py    zero-parameter persistence baseline
    gate.py                spot-checks result CSVs against facts.json

Auxiliary scripts (ablations, config generators, earlier probes) are in the
same folder; anything not on the list above is not needed for the paper's
tables.

## Reproducing the paper's numbers

Data: the eight datasets are the SALT benchmark release. Download them from the
figshare link in the official SALT repository and put them under ./Data/. You
do not need the raw data for the analysis-level reproduction -- Exp/cache
already contains every prediction the paper evaluates.

From the caches, about half an hour on CPU:

    python3 Exp/block_bootstrap.py
    python3 Exp/aci_wrap_seeds.py
    python3 Exp/family_select.py
    python3 Exp/enbpi_faithful.py
    python3 Exp/gpd_shape.py
    python3 Exp/evt_static_tail.py
    python3 Exp/gate.py

Training is optional (GPU, roughly a day for the full matrix) and only needed
to rebuild the caches from scratch:

    python3 SALT/inn_train.py --config SALT/config/<dataset>/<config>.yaml
    python3 Exp/cache_musig.py --device cuda    # Prob-run caches
    python3 Exp/cache_cqr.py                    # CQR member caches

Figure 1 regenerates from the same CSVs:

    python3 figures/fig1_camera_gen.py

## Conventions

These are easy to get wrong, so they are fixed everywhere:

* Every interval in the paper is 90% split conformal (alpha = 0.10). That
  includes the static GPD tail claim in Section V-C (evt_static_tail.csv); no
  other nominal level is used.
* Tables report seed medians over the three backbone seeds. The T-Drive s1 slot
  is the beta = 0.3 anchored run -- that seed diverges at beta = 0 and 0.1
  (Appendix D of the paper).
* The EnbPI ratios quoted in the text use the single-seed line (aci_wrap.csv),
  not the medians. EnbPI itself is the faithful version: Xu & Xie 2021,
  alpha/2 two-sided with the finite-sample correction, the 3-seed ensemble
  standing in for the bootstrap ensemble (enbpi_faithful.csv). enbpi.csv is an
  older K-sweep without the correction, kept for reference only.
* The member selector is validation-only (family_final.csv). It picks the same
  member as the seed-median narrowest on 7 of 8 datasets; LyonSchool differs,
  and the paper says where.

## One known caveat

tail_true_bins.csv (the tail-coverage numbers in Section V) is the audited file
the paper was written from. The tail_true_bins.py in this repo was edited after
that file was generated and no longer reproduces it exactly -- if you rerun it,
expect differences in a few INVIS13 rows. Everything else in Exp/calib
regenerates byte-for-byte.

## Acknowledgement

The backbone builds on the official SALT implementation (Li et al., CIKM 2025),
used with the released snapshot pipeline unchanged; no dataset is re-preprocessed.
The probabilistic heads, the conformal evaluation stack and all analysis scripts
are ours. Dataset download and terms of use: see the SALT repository.
