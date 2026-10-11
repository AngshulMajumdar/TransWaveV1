from pathlib import Path
import csv
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "benchmarks" / "comparison_tables.csv"
OUT = ROOT / "figures" / "benchmarks"
OUT.mkdir(parents=True, exist_ok=True)

display = {
    "arxiv": "arXiv",
    "fineweb_edu": "FineWeb-Edu",
    "pg19": "PG19",
    "wikitext103": "WikiText-103",
}
methods = ["Mamba-2", "Hyena", "RetNet", "TransWave"]

rows = list(csv.DictReader(CSV.open()))
for ds in display:
    block = {r["method"]: float(r["perplexity"]) for r in rows if r["dataset"] == ds}
    vals = [block[m] for m in methods]

    fig = plt.figure(figsize=(7.2, 4.8))
    ax = fig.add_axes([0.12, 0.17, 0.84, 0.75])
    bars = ax.bar(methods, vals)
    bars[-1].set_hatch("//")
    ax.set_yscale("log")
    ax.set_ylabel("Held-out perplexity (log scale)")
    ax.set_title(f"{display[ds]} — recorded benchmark perplexity")
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            value * 1.04,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.savefig(OUT / f"{ds}_perplexity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
