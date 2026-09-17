# Obtaining the real NASA Kepler data

The pipeline runs on synthetic data out of the box so that it is testable
offline. Everything below is what changes when you point it at the archive.

```bash
DATA_SOURCE=real python -m ml.preprocessing.build_dataset
```

If the data cannot be obtained, that command **fails**. It does not fall back to
synthetic data. A silent substitution is the worst failure mode this project can
have, because every downstream number would still look entirely plausible.

---

## 1. Where the data lives

NASA Exoplanet Archive, hosted at IPAC/Caltech:
<https://exoplanetarchive.ipac.caltech.edu/>

No API key, no registration. The archive asks that published work cite it; the
KOI cumulative table carries DOI `10.26133/NEA4`.

The archive has migrated from the legacy `nph-nstedAPI` endpoint to **TAP**
(Table Access Protocol), which speaks ADQL:

```
https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=<ADQL>&format=csv
```

URLs change. `ml/ingest/nasa_tap.py` reads the base URL from configuration
rather than hard-coding it, and checks the requested columns against
`TAP_SCHEMA` before issuing the query, so a renamed column fails loudly instead
of silently arriving as nulls.

---

## 2. Which table

This is the first real decision, and it is not cosmetic.

| Table | Rows | What it is | Use it when |
|---|---:|---|---|
| `cumulative` | ~9,600 | Live. The best current disposition for every KOI, assembled from several deliveries vetted by different pipeline versions. | You want current knowledge and can tolerate a heterogeneous sample. |
| `q1_q17_dr25_koi` | ~8,000 | Frozen. One uniform Robovetter run over the full mission. | You need a statistically uniform sample — anything involving completeness, reliability, or occurrence rates. |
| `q1_q17_dr25_sup_koi` | ~8,000 | Supplemental DR25, with improved stellar parameters. | You care about planet radii. See §5. |
| `q1_q17_dr25_tce` | ~34,000 | Threshold Crossing Events, before vetting. | You want the pre-vetting population. |

**Default here is `cumulative`**, because the goal is classification against the
best current labels rather than an occurrence-rate study.

If you use `cumulative`, run the heterogeneity check: cross-tabulate
`koi_tce_delivname` against `koi_disposition`. If the false-positive rate varies
materially by delivery, the sample is not identically distributed, and any
completeness statement needs `q1_q17_dr25_koi` instead.

```bash
python -m ml.ingest.download_koi --table cumulative
python -m ml.ingest.download_koi --table q1_q17_dr25_koi
```

---

## 3. Provenance is not optional

Every download writes a sidecar next to the CSV:

```
ml/data/raw/cumulative.csv
ml/data/raw/cumulative.csv.provenance.json
```

containing the exact ADQL issued, the UTC timestamp, the row and column counts,
and a SHA-256 of the file.

This matters more here than in most projects. **The `cumulative` table is
live.** Dispositions are revised: objects move from CANDIDATE to CONFIRMED, and
occasionally to FALSE POSITIVE, as follow-up accumulates. "I used the KOI
cumulative table" is therefore not a reproducible statement. "I used the KOI
cumulative table retrieved 2026-03-14T09:22Z, SHA-256 `a3f1…`" is.

`real_data_loader.py` refuses to load a CSV without its sidecar.

---

## 4. Offline after the first download

The CSV is cached. Subsequent runs read from disk and never touch the network:

```bash
python -m ml.ingest.download_koi --table cumulative   # once
DATA_SOURCE=real python -m ml.preprocessing.build_dataset
DATA_SOURCE=real python -m ml.training.train --include-leaky
```

To force a refresh, delete the CSV and its sidecar together. Deleting only one
leaves the loader in a state it will reject, which is intentional.

---

## 5. What will differ from the synthetic run

Expect these, and treat their absence as a reason to check the code rather than
a happy result.

**Scores will be lower.** The synthetic generator has a simple generating
process, so the synthetic PR-AUC near 0.996 is not a forecast. Honest tabular
KOI classification without Robovetter columns lands closer to 0.93–0.96, and the
gap between the physics-informed and baseline feature sets will be smaller.

**Columns will be missing.** Deliveries differ. `ml/config/features.py` resolves
every requested feature list against the schema actually present and logs what
was absent; `koi_smet`, `koi_model_chisq` and `koi_max_sngle_ev` are common
casualties.

**Missingness will not be random.** Run the missingness-by-class report
(`ml/data/processed/missingness_by_class.csv`). Any column missing at materially
different rates across classes carries label information in the *fact* of being
missing, and a fitted imputer will launder that into the values where no
column-name audit will find it.

**The radius valley should appear only in the precise subsample.** The 1.5–2.0
R⊕ deficit (Fulton et al. 2017) resolves only with good stellar radii. Figure 3
from `ml/eda/report.py` plots the full sample against the SME/Astero subset. If
the valley is equally sharp in both panels, something is wrong; if it is absent
from both, check whether the precise subsample is large enough to show it.

**Stellar parameters dominate the error budget.** Roughly 80% of KOIs carry
KIC-provenance stellar parameters with fractional radius uncertainties near 30%.
Gaia parallaxes later revised many Kepler stellar radii substantially, moving
planets across the Earth-size boundary in both directions.
`q1_q17_dr25_sup_koi` carries improved parameters and is worth a comparison run.

---

## 6. Alternative acquisition routes

**Manual CSV.** Download from the archive's web interface and point the pipeline
at it directly:

```bash
python -m ml.preprocessing.build_dataset --csv /path/to/your.csv
```

This bypasses the provenance sidecar, so record the retrieval date yourself.

**Bulk.** The archive publishes downloadable table dumps for users who prefer
not to query. Same caching rules apply.

---

## 7. Citation

If you publish anything using this data:

> This research has made use of the NASA Exoplanet Archive, which is operated by
> the California Institute of Technology, under contract with the National
> Aeronautics and Space Administration under the Exoplanet Exploration Program.

KOI cumulative table DOI: `10.26133/NEA4`.

This project is not affiliated with or endorsed by NASA.
