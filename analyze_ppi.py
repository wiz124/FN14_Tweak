"""
analyze_ppi.py — AlphaFold3 PPI analysis CLI
=============================================
Analyses any AF3 prediction from a CIF + summary_confidences JSON pair.
Generates per-residue pLDDT plots, confidence heatmaps, interface contact
maps, a CSV of contact pairs, and a self-contained HTML report.

Usage
-----
# Single model:
python analyze_ppi.py \\
    --cif   fold_fn14_tweak_predict_model_0.cif \\
    --json  fold_fn14_tweak_predict_summary_confidences_0.json \\
    --chain-roles A:TWEAK B:FN14-IDR \\
    --label "TWEAK+FN14 dimer" \\
    --out   results/

# Multiple models in one call (also produces a cross-model comparison chart):
python analyze_ppi.py \\
    --cif   dimer.cif   trimer.cif   fn14_only.cif \\
    --json  dimer.json  trimer.json  fn14_only.json \\
    --chain-roles "A:TWEAK,B:FN14-IDR" \\
                  "A:TWEAK-1,B:TWEAK-2,C:TWEAK-3,D:FN14-IDR-1,E:FN14-IDR-2,F:FN14-IDR-3" \\
                  "A:FN14-IDR-1,B:FN14-IDR-2,C:FN14-IDR-3" \\
    --label "Dimer" "Trimer" "FN14 homotrimer" \\
    --out   results/

Options
-------
--cif           One or more mmCIF paths (required).
--json          One or more summary_confidences JSON paths (required, same order as --cif).
--chain-roles   One role-string per model: comma-separated CHAIN:ROLE pairs, e.g.
                "A:TWEAK,B:FN14-IDR". If omitted, chains are labelled Chain-A, Chain-B, …
--label         Human-readable label for each model (used in filenames and plots).
                Defaults to the CIF stem.
--out           Output directory (default: ./af3_ppi_output).
--cutoff        Cα–Cα distance cutoff in Å for contacts (default: 8.0).
--no-report     Skip HTML report generation.
--color-map     Optional JSON file mapping role names to hex colours.

Dependencies
------------
    pip install numpy pandas matplotlib seaborn
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

# ── Default colour palette ────────────────────────────────────────────────────

DEFAULT_COLORS: dict[str, str] = {
    "TWEAK":      "#185FA5",
    "TWEAK-1":    "#185FA5",
    "TWEAK-2":    "#378ADD",
    "TWEAK-3":    "#85B7EB",
    "FN14-IDR":   "#D85A30",
    "FN14-IDR-1": "#D85A30",
    "FN14-IDR-2": "#EF9F27",
    "FN14-IDR-3": "#BA7517",
}

# Fallback palette for unknown roles (cycles through these)
_FALLBACK_PALETTE = [
    "#185FA5", "#D85A30", "#3B6D11", "#993556",
    "#534AB7", "#BA7517", "#0F6E56", "#A32D2D",
]


def role_color(role: str, color_map: dict[str, str]) -> str:
    merged = {**DEFAULT_COLORS, **color_map}
    if role in merged:
        return merged[role]
    # Derive a stable colour from the role string hash
    idx = abs(hash(role)) % len(_FALLBACK_PALETTE)
    return _FALLBACK_PALETTE[idx]


# ── mmCIF parser ──────────────────────────────────────────────────────────────

def parse_atom_site(cif_path: Path) -> pd.DataFrame:
    """Parse the _atom_site loop from an mmCIF file."""
    with open(cif_path) as fh:
        lines = fh.readlines()

    in_loop = in_atom_site = False
    columns: list[str] = []
    rows: list[list[str]] = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line == "loop_":
            in_loop, in_atom_site, columns = True, False, []
            i += 1
            continue

        if in_loop and line.startswith("_atom_site."):
            in_atom_site = True
            columns.append(line)
            i += 1
            continue

        if in_atom_site and not line.startswith("_"):
            if line in ("", "#") or line.startswith("loop_"):
                if line.startswith("loop_"):
                    in_loop, in_atom_site, columns = True, False, []
                else:
                    in_loop = in_atom_site = False
                i += 1
                continue
            rows.append(line.split())
            i += 1
            continue

        i += 1

    col_names = [c.replace("_atom_site.", "") for c in columns]
    df = pd.DataFrame(rows, columns=col_names)

    numeric = ["Cartn_x", "Cartn_y", "Cartn_z", "B_iso_or_equiv",
               "occupancy", "label_seq_id", "auth_seq_id", "id"]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_ca_atoms(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["label_atom_id"] == "CA"].copy().reset_index(drop=True)


# ── Confidence JSON ───────────────────────────────────────────────────────────

def load_confidences(json_path: Path) -> dict:
    with open(json_path) as fh:
        return json.load(fh)


# ── Chain role parsing ────────────────────────────────────────────────────────

def parse_chain_roles(role_str: Optional[str],
                      actual_chains: list[str]) -> dict[str, str]:
    """
    Parse "A:TWEAK,B:FN14-IDR" into {"A": "TWEAK", "B": "FN14-IDR"}.
    If role_str is None, assigns "Chain-A", "Chain-B", … to actual chains.
    """
    if not role_str:
        return {ch: f"Chain-{ch}" for ch in actual_chains}

    roles: dict[str, str] = {}
    for pair in role_str.split(","):
        pair = pair.strip()
        if ":" not in pair:
            raise ValueError(f"Bad chain-role pair '{pair}'. Expected CHAIN:ROLE.")
        chain, role = pair.split(":", 1)
        roles[chain.strip()] = role.strip()

    # Fill in any chains not explicitly listed
    for ch in actual_chains:
        if ch not in roles:
            roles[ch] = f"Chain-{ch}"

    return roles


# ── Interface analysis ────────────────────────────────────────────────────────

def get_interface_residues(ca_df: pd.DataFrame,
                           chain_a: str, chain_b: str,
                           cutoff: float) -> pd.DataFrame:
    a = ca_df[ca_df["label_asym_id"] == chain_a].reset_index(drop=True)
    b = ca_df[ca_df["label_asym_id"] == chain_b].reset_index(drop=True)

    if a.empty or b.empty:
        return pd.DataFrame()

    coords_a = a[["Cartn_x", "Cartn_y", "Cartn_z"]].values
    coords_b = b[["Cartn_x", "Cartn_y", "Cartn_z"]].values
    dists = np.sqrt(((coords_a[:, None, :] - coords_b[None, :, :]) ** 2).sum(-1))
    contacts = np.argwhere(dists < cutoff)

    rows = []
    for ia, ib in contacts:
        rows.append({
            "chain_A":    chain_a,
            "resname_A":  a.loc[ia, "label_comp_id"],
            "resnum_A":   int(a.loc[ia, "label_seq_id"]),
            "plddt_A":    a.loc[ia, "B_iso_or_equiv"],
            "chain_B":    chain_b,
            "resname_B":  b.loc[ib, "label_comp_id"],
            "resnum_B":   int(b.loc[ib, "label_seq_id"]),
            "plddt_B":    b.loc[ib, "B_iso_or_equiv"],
            "distance_A": round(float(dists[ia, ib]), 2),
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("distance_A").reset_index(drop=True)


def interface_summary(ct: pd.DataFrame,
                      role_a: str, role_b: str) -> dict:
    if ct.empty:
        return {}
    return {
        "pair":            f"{role_a} ↔ {role_b}",
        "n_contact_pairs": len(ct),
        "unique_res_A":    ct["resnum_A"].nunique(),
        "unique_res_B":    ct["resnum_B"].nunique(),
        "min_dist_A":      round(float(ct["distance_A"].min()), 2),
        "mean_dist_A":     round(float(ct["distance_A"].mean()), 2),
        "mean_plddt_A":    round(float(ct["plddt_A"].mean()), 2),
        "mean_plddt_B":    round(float(ct["plddt_B"].mean()), 2),
    }


# ── pLDDT stats ───────────────────────────────────────────────────────────────

def chain_plddt_stats(ca_df: pd.DataFrame,
                      chain_roles: dict[str, str]) -> pd.DataFrame:
    rows = []
    for chain_id, role in chain_roles.items():
        sub = ca_df[ca_df["label_asym_id"] == chain_id]["B_iso_or_equiv"]
        if sub.empty:
            continue
        rows.append({
            "chain":          chain_id,
            "role":           role,
            "n_residues":     len(sub),
            "mean_plddt":     round(float(sub.mean()), 2),
            "median_plddt":   round(float(sub.median()), 2),
            "pct_above_70":   round(float((sub >= 70).mean() * 100), 1),
            "pct_above_90":   round(float((sub >= 90).mean() * 100), 1),
        })
    return pd.DataFrame(rows)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_plddt(ca_df: pd.DataFrame, chain_roles: dict[str, str],
               title: str, out_path: Path,
               color_map: dict[str, str]) -> None:
    fig, ax = plt.subplots(figsize=(12, 3.5))
    x_offset = 0
    xtick_pos, xtick_lab, patches = [], [], []

    for chain_id, role in chain_roles.items():
        sub = ca_df[ca_df["label_asym_id"] == chain_id].reset_index(drop=True)
        if sub.empty:
            continue
        xs = np.arange(len(sub)) + x_offset
        color = role_color(role, color_map)
        ax.bar(xs, sub["B_iso_or_equiv"], width=1.0,
               color=color, alpha=0.85, linewidth=0)
        xtick_pos.append(x_offset + len(sub) // 2)
        xtick_lab.append(f"{role}\n(chain {chain_id})")
        patches.append(mpatches.Patch(color=color, label=f"{role} ({chain_id})"))
        x_offset += len(sub) + 15

    for y, label in [(50, "Low"), (70, "Confident"), (90, "Very high")]:
        ax.axhline(y, color="#888", lw=0.8, linestyle="--", alpha=0.5)
        ax.text(x_offset + 5, y + 1, label, fontsize=7, color="#666", va="bottom")

    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_lab, fontsize=9)
    ax.set_ylabel("pLDDT", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.legend(handles=patches, fontsize=8, loc="lower right", framealpha=0.7)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_plddt_histogram(ca_df: pd.DataFrame, chain_roles: dict[str, str],
                         title: str, out_path: Path,
                         color_map: dict[str, str]) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for chain_id, role in chain_roles.items():
        sub = ca_df[ca_df["label_asym_id"] == chain_id]
        if sub.empty:
            continue
        ax.hist(sub["B_iso_or_equiv"], bins=20, range=(0, 100),
                color=role_color(role, color_map), alpha=0.55,
                label=f"{role} ({chain_id})", density=True,
                edgecolor="white", linewidth=0.3)
    for x, ls in [(50, ":"), (70, "--"), (90, "-.")]:
        ax.axvline(x, color="#555", lw=0.8, linestyle=ls, alpha=0.6)
    ax.set_xlabel("pLDDT", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.7)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_iptm_heatmap(conf: dict, chain_roles: dict[str, str],
                      title: str, out_path: Path) -> None:
    mat = np.array(conf["chain_pair_iptm"])
    n = len(mat)
    labels = [f"{role}\n({cid})" for cid, role in
              list(chain_roles.items())[:n]]
    fig, ax = plt.subplots(figsize=(max(4, n * 1.1), max(3.5, n * 0.95)))
    sns.heatmap(mat, annot=True, fmt=".2f", vmin=0.0, vmax=1.0,
                cmap="Blues", ax=ax,
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor="#eee",
                annot_kws={"size": 9})
    ax.set_title(f"{title}\nchain_pair_iptm", fontsize=10, fontweight="bold", pad=6)
    ax.tick_params(axis="x", labelsize=7, rotation=0)
    ax.tick_params(axis="y", labelsize=7, rotation=0)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_pae_min_heatmap(conf: dict, chain_roles: dict[str, str],
                         title: str, out_path: Path) -> None:
    mat = np.array(conf["chain_pair_pae_min"])
    n = len(mat)
    vmax = max(5.0, float(np.percentile(mat[~np.eye(n, dtype=bool)], 95))
               if n > 1 else 5.0)
    labels = [f"{role}\n({cid})" for cid, role in
              list(chain_roles.items())[:n]]
    fig, ax = plt.subplots(figsize=(max(4, n * 1.1), max(3.5, n * 0.95)))
    sns.heatmap(mat, annot=True, fmt=".1f", vmin=0.5, vmax=vmax,
                cmap="RdYlGn_r", ax=ax,
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor="#eee",
                annot_kws={"size": 9})
    ax.set_title(f"{title}\nchain_pair_pae_min (Å)", fontsize=10, fontweight="bold", pad=6)
    ax.tick_params(axis="x", labelsize=7, rotation=0)
    ax.tick_params(axis="y", labelsize=7, rotation=0)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_interface_contacts(ct: pd.DataFrame,
                            role_a: str, role_b: str,
                            title: str, out_path: Path,
                            cutoff: float) -> None:
    if ct.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    sc = ax.scatter(ct["resnum_A"], ct["resnum_B"],
                    c=ct["distance_A"], cmap="YlOrRd_r",
                    vmin=3, vmax=cutoff, s=20, alpha=0.8, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label("Cα–Cα distance (Å)", fontsize=9)
    ax.set_xlabel(f"Residue — {role_a}", fontsize=9)
    ax.set_ylabel(f"Residue — {role_b}", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    n_uniq = ct["resnum_A"].nunique() + ct["resnum_B"].nunique()
    ax.text(0.01, 0.98,
            f"{len(ct)} contact pairs | {n_uniq} unique residues",
            transform=ax.transAxes, fontsize=8, va="top", color="#555")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(model_labels: list[str],
                    confs: list[dict],
                    out_path: Path) -> None:
    """Bar chart comparing iptm / ptm / ranking_score across models."""
    metrics = ["iptm", "ptm", "ranking_score"]
    palette = (_FALLBACK_PALETTE * 4)[:len(model_labels)]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, metric in zip(axes, metrics):
        vals = [c.get(metric, 0) for c in confs]
        bars = ax.bar(model_labels, vals, color=palette[:len(vals)],
                      width=0.5, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                    f"{v:.2f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.set_title(metric, fontsize=10, fontweight="bold")
        ax.axhline(0.5,  color="#aaa", lw=0.8, linestyle="--")
        ax.axhline(0.75, color="#555", lw=0.8, linestyle="--")
        ax.tick_params(axis="x", labelsize=8, rotation=15)

    fig.suptitle("Confidence metrics across models", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ── HTML report ───────────────────────────────────────────────────────────────

def _img(path: Path, base: Path) -> str:
    rel = path.relative_to(base)
    return (f'<img src="{rel}" '
            f'style="max-width:100%;border-radius:6px;margin:6px 0;">')


def make_report(label: str, conf: dict,
                plddt_stats: pd.DataFrame,
                contact_summaries: list[dict],
                image_paths: list[Path],
                out_path: Path) -> None:
    base = out_path.parent

    def sec(title, text="", imgs=(), table=None):
        h = f"<h2>{title}</h2>\n"
        if text:
            h += f"<p>{text}</p>\n"
        for img in imgs:
            if img.exists():
                h += _img(img, base) + "\n"
        if table is not None and not table.empty:
            h += table.to_html(index=False, border=0, classes="tbl") + "\n"
        return h

    body = ""

    # Global metrics
    gm = (f"iptm={conf.get('iptm','?')} | ptm={conf.get('ptm','?')} | "
          f"ranking_score={conf.get('ranking_score','?')} | "
          f"fraction_disordered={conf.get('fraction_disordered','?')} | "
          f"has_clash={conf.get('has_clash','?')}")
    imgs_conf = [p for p in image_paths if "iptm" in p.name or "pae_min" in p.name]
    body += sec("Global confidence metrics", gm, imgs_conf)

    # pLDDT
    imgs_plddt = [p for p in image_paths if "plddt" in p.name]
    body += sec("Per-residue pLDDT", "", imgs_plddt, plddt_stats)

    # Contacts
    imgs_iface = [p for p in image_paths if "interface" in p.name]
    smry_rows = [s for s in contact_summaries if s]
    smry_text = ""
    if smry_rows:
        smry_df = pd.DataFrame(smry_rows)
        smry_html = smry_df.to_html(index=False, border=0, classes="tbl")
    else:
        smry_html = "<p>No contacts found at the chosen cutoff.</p>"
        smry_df = pd.DataFrame()
    body += sec("Interface contacts", "", imgs_iface)
    body += "<h2>Interface summary table</h2>\n" + smry_html + "\n"

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>AF3 PPI — {label}</title>
<style>
  body  {{ font-family: -apple-system, sans-serif; max-width: 980px;
           margin: 40px auto; padding: 0 20px; color: #222; }}
  h1    {{ font-size: 1.4rem; border-bottom: 2px solid #185FA5;
           padding-bottom: 6px; color: #185FA5; }}
  h2    {{ font-size: 1.05rem; margin-top: 2rem; color: #333; }}
  p     {{ font-size: 0.88rem; color: #555; }}
  .tbl  {{ border-collapse: collapse; width: 100%;
           font-size: 0.83rem; margin: 8px 0 18px; }}
  .tbl td, .tbl th {{ border: 1px solid #ddd; padding: 5px 9px; }}
  .tbl th {{ background: #f0f4fa; font-weight: 600; }}
  .tbl tr:nth-child(even) {{ background: #f9f9f9; }}
</style>
</head>
<body>
<h1>AlphaFold3 PPI analysis — {label}</h1>
{body}
</body>
</html>"""

    with open(out_path, "w") as fh:
        fh.write(html)


# ── Per-model runner ──────────────────────────────────────────────────────────

def run_model(cif_path: Path, json_path: Path,
              chain_roles_str: Optional[str],
              label: str,
              out_dir: Path,
              cutoff: float,
              color_map: dict[str, str],
              no_report: bool) -> tuple[dict, list[dict]]:
    """
    Run the full analysis for a single model.
    Returns (conf_dict, contact_summary_list).
    """
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", label)
    model_dir = out_dir / slug
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*60}")
    print(f"  Model : {label}")
    print(f"  CIF   : {cif_path.name}")
    print(f"  JSON  : {json_path.name}")
    print(f"  Output: {model_dir}")
    print(f"{'─'*60}")

    # Load
    print("  Parsing CIF …")
    atoms = parse_atom_site(cif_path)
    ca    = get_ca_atoms(atoms)
    conf  = load_confidences(json_path)

    actual_chains = sorted(ca["label_asym_id"].unique())
    print(f"  Chains found: {actual_chains}  ({len(ca)} Cα atoms)")

    chain_roles = parse_chain_roles(chain_roles_str, actual_chains)
    print(f"  Chain roles : {chain_roles}")

    # pLDDT stats
    stats = chain_plddt_stats(ca, chain_roles)
    print("\n  pLDDT stats:")
    print(stats.to_string(index=False, col_space=14))
    stats.to_csv(model_dir / "plddt_stats.csv", index=False)

    all_images: list[Path] = []

    # pLDDT plots
    p = model_dir / "plddt_per_residue.png"
    plot_plddt(ca, chain_roles, f"{label}: per-residue pLDDT", p, color_map)
    all_images.append(p)

    p = model_dir / "plddt_histogram.png"
    plot_plddt_histogram(ca, chain_roles, f"{label}: pLDDT distribution", p, color_map)
    all_images.append(p)

    # Confidence heatmaps
    p = model_dir / "chain_pair_iptm.png"
    plot_iptm_heatmap(conf, chain_roles, label, p)
    all_images.append(p)

    p = model_dir / "chain_pair_pae_min.png"
    plot_pae_min_heatmap(conf, chain_roles, label, p)
    all_images.append(p)

    # Interface contacts — every unique chain pair
    all_contacts: list[pd.DataFrame] = []
    contact_summaries: list[dict] = []
    chains = list(chain_roles.keys())

    print("\n  Interface contacts:")
    for ca_id, cb_id in combinations(chains, 2):
        ct = get_interface_residues(ca, ca_id, cb_id, cutoff)
        role_a = chain_roles[ca_id]
        role_b = chain_roles[cb_id]
        smry = interface_summary(ct, role_a, role_b)

        if ct.empty:
            print(f"    {role_a} ↔ {role_b}: no contacts at {cutoff} Å")
        else:
            print(f"    {role_a} ↔ {role_b}: "
                  f"{smry['n_contact_pairs']} pairs, "
                  f"min {smry['min_dist_A']} Å, "
                  f"mean pLDDT {smry['mean_plddt_A']:.1f}/{smry['mean_plddt_B']:.1f}")
            all_contacts.append(ct)
            contact_summaries.append(smry)
            img_name = f"interface_{ca_id}_{cb_id}.png"
            p = model_dir / img_name
            plot_interface_contacts(ct, role_a, role_b,
                                    f"{label}: {role_a} ↔ {role_b}", p, cutoff)
            all_images.append(p)

    if all_contacts:
        merged = pd.concat(all_contacts, ignore_index=True)
        merged.to_csv(model_dir / "interface_contacts.csv", index=False)
        print(f"\n  Saved: interface_contacts.csv  ({len(merged)} rows)")

    # HTML report
    if not no_report:
        report_path = model_dir / "report.html"
        make_report(label, conf, stats, contact_summaries,
                    all_images, report_path)
        print(f"  Saved: report.html")

    return conf, contact_summaries


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AlphaFold3 PPI analysis — single or multi-model CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--cif",    nargs="+", required=True, type=Path,
                   metavar="FILE",
                   help="One or more mmCIF files.")
    p.add_argument("--json",   nargs="+", required=True, type=Path,
                   metavar="FILE",
                   help="One summary_confidences JSON per CIF (same order).")
    p.add_argument("--chain-roles", nargs="+", default=None,
                   metavar="ROLES",
                   help='Chain-role strings, one per model. '
                        'Format: "A:TWEAK,B:FN14-IDR"')
    p.add_argument("--label",  nargs="+", default=None,
                   metavar="LABEL",
                   help="Human-readable label for each model.")
    p.add_argument("--out",    default=Path("af3_ppi_output"), type=Path,
                   metavar="DIR",
                   help="Output directory (default: af3_ppi_output).")
    p.add_argument("--cutoff", default=8.0, type=float,
                   metavar="Å",
                   help="Cα–Cα contact cutoff in Å (default: 8.0).")
    p.add_argument("--no-report", action="store_true",
                   help="Skip HTML report generation.")
    p.add_argument("--color-map", default=None, type=Path,
                   metavar="JSON",
                   help='JSON file mapping role names to hex colours, e.g. '
                        '{"TWEAK": "#185FA5"}')
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    # Validate input counts
    if len(args.cif) != len(args.json):
        sys.exit("ERROR: --cif and --json must have the same number of files.")

    n = len(args.cif)

    labels = args.label or [p.stem for p in args.cif]
    if len(labels) < n:
        labels += [args.cif[i].stem for i in range(len(labels), n)]

    chain_roles_list = args.chain_roles or [None] * n
    if len(chain_roles_list) < n:
        chain_roles_list += [None] * (n - len(chain_roles_list))

    color_map: dict[str, str] = {}
    if args.color_map:
        with open(args.color_map) as fh:
            color_map = json.load(fh)

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  AlphaFold3 PPI analysis")
    print(f"  Models    : {n}")
    print(f"  Cutoff    : {args.cutoff} Å")
    print(f"  Output    : {args.out}")
    print(f"{'='*60}")

    all_confs: list[dict] = []
    all_summaries: list[list[dict]] = []

    for cif, jsn, roles_str, label in zip(
            args.cif, args.json, chain_roles_list, labels):

        if not cif.exists():
            print(f"\nWARNING: CIF not found: {cif} — skipping.")
            continue
        if not jsn.exists():
            print(f"\nWARNING: JSON not found: {jsn} — skipping.")
            continue

        conf, summaries = run_model(
            cif_path=cif,
            json_path=jsn,
            chain_roles_str=roles_str,
            label=label,
            out_dir=args.out,
            cutoff=args.cutoff,
            color_map=color_map,
            no_report=args.no_report,
        )
        all_confs.append(conf)
        all_summaries.append(summaries)

    # Cross-model comparison (only when >1 model)
    if len(all_confs) > 1:
        print(f"\n{'─'*60}")
        print("  Cross-model comparison …")
        comp_path = args.out / "comparison_all_models.png"
        plot_comparison(labels[:len(all_confs)], all_confs, comp_path)
        print(f"  Saved: {comp_path}")

        # Summary CSV
        rows = []
        for label, conf in zip(labels, all_confs):
            rows.append({
                "model":               label,
                "iptm":                conf.get("iptm"),
                "ptm":                 conf.get("ptm"),
                "ranking_score":       conf.get("ranking_score"),
                "fraction_disordered": conf.get("fraction_disordered"),
                "has_clash":           conf.get("has_clash"),
            })
        summary_df = pd.DataFrame(rows)
        summary_csv = args.out / "model_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"  Saved: {summary_csv}")
        print("\n  Model summary:")
        print(summary_df.to_string(index=False))

    print(f"\n{'='*60}")
    print(f"  ✓ Done. All outputs written to: {args.out}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
