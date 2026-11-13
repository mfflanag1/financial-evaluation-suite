"""Render a browsable case catalog from the versioned quality inventory."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACKS = {
    "source_grounded": "Source-grounded analysis",
    "quant100": "Quantitative methods",
    "workflows": "Executable workflows",
    "research58": "Research scenarios",
    "challenges": "Decision core",
    "variants": "Related diagnostic variants",
}


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def render():
    inventory = json.loads((ROOT / "quality/inventory.json").read_text())
    lines = ["# Case catalog", "", "Generated from the case-level quality inventory. "
             "All static evaluator answers are public; historical holdout labels are not secrecy claims.",
             "", "The bank has 224 main cases and six related diagnostic variants. "
             "A numerical check means a separate calculation under the supplied conventions; "
             "it does not mean practitioner review. Pending cases have not received that check.",
             "", "Use the versioned grader's export command to obtain the current response contract. "
             "Links below identify frozen opening packets. For sequential cases, use the coordinator "
             "to obtain later states. See the [quickstart](quickstart.md).",
             "", "Source redistribution is narrower than catalog coverage; see the "
             "[source policy](source-policy.md). Do not redistribute a complete source case merely "
             "because it appears here.", ""]
    for track, title in TRACKS.items():
        rows = [row for row in inventory["cases"] if row["track"] == track]
        lines.extend([f"## {title}", "", f"{len(rows)} cases.", "",
                      "| ID | Task | Area | Turns | Outputs | Numerical check |",
                      "| --- | --- | --- | ---: | ---: | --- |"])
        for row in rows:
            artifact = row["input_artifact"]
            link = f"[{row['id']}](../{artifact['path']})" if artifact else row["id"]
            verified = row["verification"]["independent_numerical_recomputation"]
            status = "Matched" if verified == "matches_stored_reference" else (
                "Discrepancy" if verified == "reference_discrepancy" else "Pending")
            lines.append("| " + " | ".join(map(cell, (link, row["title"], row["area"],
                         row["turn_count"], row["declared_output_count"], status))) + " |")
        lines.append("")
    lines.extend(["## Audit detail", "", "The [machine-readable inventory](../quality/inventory.json) "
                  "records hashes, stage schemas, exposure and review status. "
                  "[Independent checks](../quality/independent_checks.json) record derivations and comparisons. "
                  "The [discrete-field audit](../quality/discrete_fields.json) distinguishes explicit integer "
                  "requirements from quantities that need further interpretation.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    target = ROOT / "docs/case-catalog.md"
    content = render()
    if args.check:
        if not target.exists() or target.read_text() != content:
            raise SystemExit("Case catalog is stale; run python scripts/catalog.py")
    else:
        target.write_text(content)


if __name__ == "__main__":
    main()
