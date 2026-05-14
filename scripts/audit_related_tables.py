"""Audit all Related Information tables in raw_data.txt.

For each sample, list:
1. All Related tables found
2. Each table's extracted abbreviation pair
3. Whether it was KEPT or SKIPPED by the filter
4. Manual verdict (based on matching against sample's own monomers)

This helps catch false negatives (wrong data that slipped through the filter).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from parse_polyinfo_raw import (
    parse_monomer_blocks, parse_sample_blocks,
    build_smiles_lookup, _build_sample_abbrev_set,
    _extract_abbrev_pair, _abbrevs_match,
)


def main() -> None:
    raw_text = Path("raw_data.txt").read_text(encoding="utf-8")
    mono_text = Path("raw_data1.txt").read_text(encoding="utf-8")

    monomers = parse_monomer_blocks(mono_text)
    samples = parse_sample_blocks(raw_text)
    _, _, cuid_to_monomer = build_smiles_lookup(monomers)

    print(f"Audit: {len(samples)} samples, {sum(len(s.related_rows) for s in samples)} related rows")
    print("=" * 80)

    kept_count = 0
    skipped_count = 0
    suspicious: list[str] = []

    for s in samples:
        if not s.related_rows:
            continue

        sample_abbrevs = _build_sample_abbrev_set(s, cuid_to_monomer)
        sample_label = "/".join(sorted(sample_abbrevs)) if sample_abbrevs else "(no abbrev)"

        # Group related_rows by unique (title, header) to avoid repeat per data row
        seen_tables: dict[tuple, int] = {}
        for row in s.related_rows:
            if len(row) == 3:
                title, header, _ = row
            else:
                title, header = "", row[0]
            key = (title, header)
            seen_tables[key] = seen_tables.get(key, 0) + 1

        for (title, header), n_rows in seen_tables.items():
            hdr_abbrevs = _extract_abbrev_pair(header) or _extract_abbrev_pair(title)
            hdr_label = "/".join(sorted(hdr_abbrevs)) if hdr_abbrevs else "?"

            if not hdr_abbrevs or not sample_abbrevs:
                status = "KEPT (no filter — unmatched abbrev)"
                kept_count += n_rows
                verdict = "?"
            elif _abbrevs_match(hdr_abbrevs, sample_abbrevs):
                status = "KEPT"
                kept_count += n_rows
                verdict = "match"
            else:
                status = "SKIPPED"
                skipped_count += n_rows
                verdict = "mismatch"

            # Print summary line
            title_display = title[:55] if title else "(no title)"
            header_display = header[:50]
            print(f"  [{status:8s}] sample {s.sample_id}")
            print(f"    sample_abbrevs: {sample_label}")
            print(f"    header_abbrevs: {hdr_label}")
            print(f"    title: {title_display}")
            print(f"    header: {header_display}")
            print(f"    rows: {n_rows}")

            # Flag suspicious cases
            if status.startswith("KEPT"):
                # Check 1: no filter applied because we couldn't extract abbrevs
                if not hdr_abbrevs:
                    suspicious.append(
                        f"  NO FILTER: sample {s.sample_id} ({sample_label}) kept table with title '{title[:40]}' — couldn't extract abbrev pair"
                    )
                if not sample_abbrevs:
                    suspicious.append(
                        f"  NO SAMPLE ABBREV: sample {s.sample_id} kept table for '{hdr_label}' — couldn't derive sample abbrev"
                    )
            print()

    print("=" * 80)
    print(f"KEPT: {kept_count} rows, SKIPPED: {skipped_count} rows")
    print()
    if suspicious:
        print(f"[SUSPICIOUS] {len(suspicious)} cases to review:")
        for msg in suspicious:
            print(msg)
    else:
        print("No suspicious cases — all Related tables either filtered or matched.")


if __name__ == "__main__":
    main()
