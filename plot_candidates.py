import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

DATASET = Path("koi_training_data.csv")
RANKED = Path("artifacts_physics_grouped") / "earthlike_candidates.csv"
OUTPUT_DIR = Path("artifacts_physics_grouped")

# Make absolutely sure the output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATASET)
candidates = pd.read_csv(RANKED)

print("=" * 78)
print("EXOVISION AI — EARTH-LIKE CANDIDATE VISUALIZATION")
print("=" * 78)

print(f"Dataset rows: {len(df)}")
print(f"Candidate rows: {len(candidates)}")

plt.figure(figsize=(10, 7))

# Full Kepler/KOI population
plt.scatter(
    df["koi_insol"],
    df["koi_prad"],
    alpha=0.25,
    s=15,
    label="Kepler KOI dataset",
)

# Selected candidates
plt.scatter(
    candidates["koi_insol"],
    candidates["koi_prad"],
    s=70,
    label="Earth-like candidates",
)

# Candidate labels
for _, row in candidates.iterrows():
    plt.annotate(
        str(row["kepoi_name"]).strip(),
        (
            row["koi_insol"],
            row["koi_prad"],
        ),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8,
    )

# Physical screening boundaries
plt.axvline(
    0.5,
    linestyle="--",
    linewidth=1,
)

plt.axvline(
    2.0,
    linestyle="--",
    linewidth=1,
)

plt.axhline(
    0.8,
    linestyle="--",
    linewidth=1,
)

plt.axhline(
    1.25,
    linestyle="--",
    linewidth=1,
)

plt.xscale("log")

plt.xlabel("Stellar Insolation (Earth = 1)")
plt.ylabel("Planet Radius (Earth radii)")

plt.title(
    "ExoVision AI — Earth-Like Candidate Screening"
)

plt.legend()
plt.grid(alpha=0.2)

plt.tight_layout()

# Use an explicit Windows-safe absolute path
output = OUTPUT_DIR.resolve() / "earthlike_candidate_space.png"

plt.savefig(
    str(output),
    dpi=200,
    format="png",
)

plt.close()

print()
print("Plot generated successfully.")
print()
print("Saved:")
print(output)
print("=" * 78)