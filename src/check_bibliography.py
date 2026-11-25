#!/usr/bin/env python3
"""
Deep bibliography consistency check.

DOI metadata cannot be fetched from this environment, but a surprising amount of
bibliographic metadata is *derivable* and can therefore be checked without any
network access:

  * JMLR volume numbering is deterministic: volume = year - 1999.
  * NeurIPS proceedings numbering is deterministic: NeurIPS N = year - 1987.
  * arXiv identifiers encode their submission month: YYMM.NNNNN.
  * ICML/PMLR, KDD and CAiSE proceedings carry the year in the series name.
  * Author surnames should be spelled identically across entries.

Anything these rules cannot reach is emitted as a manual worksheet with a
ready-made lookup URL per entry.

Usage:  python check_bibliography.py refs.bib
Output: bib_verification_worksheet.csv
"""
import re
import sys
from urllib.parse import quote

BIB = sys.argv[1] if len(sys.argv) > 1 else "refs.bib"


def parse(text):
    out = []
    for m in re.finditer(r"@(\w+)\{([^,]+),", text):
        typ, key = m.group(1).lower(), m.group(2).strip()
        i, depth = m.end(), 1
        while depth and i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        body = text[m.end():i - 1]
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*\{", body):
            k, j, d = fm.group(1).lower(), fm.end(), 1
            while d and j < len(body):
                if body[j] == "{":
                    d += 1
                elif body[j] == "}":
                    d -= 1
                j += 1
            fields[k] = " ".join(body[fm.end():j - 1].split())
        out.append((typ, key, fields))
    return out


def surnames(author):
    """Crude surname extraction, good enough for spelling-consistency checks."""
    names = []
    for a in author.split(" and "):
        a = a.strip()
        names.append(a.split(",")[0].strip() if "," in a else a.split()[-1])
    return names


def main():
    entries = parse(open(BIB, encoding="utf-8").read())
    print(f"parsed {len(entries)} entries from {BIB}\n")

    findings, worksheet = [], []
    seen_surnames = {}

    for typ, key, f in entries:
        year = f.get("year", "").strip()
        yr = int(year) if year.isdigit() else None
        venue = f.get("journal", "") or f.get("booktitle", "")
        checked = []

        # --- derivable rule 1: JMLR volume = year - 1999 ----------------------
        if "Journal of Machine Learning Research" in venue and yr and "volume" in f:
            exp = yr - 1999
            got = f["volume"].strip()
            ok = got.isdigit() and int(got) == exp
            checked.append("JMLR volume")
            if not ok:
                findings.append((key, f"JMLR volume {got} inconsistent with year "
                                      f"{yr} (expected {exp})"))

        # --- derivable rule 2: NeurIPS proceedings number = year - 1987 -------
        m = re.search(r"Neural Information Processing Systems (\d+)", venue)
        if m and yr:
            exp = yr - 1987
            got = int(m.group(1))
            checked.append("NeurIPS number")
            if got != exp:
                findings.append((key, f"NeurIPS {got} inconsistent with year {yr} "
                                      f"(expected {exp})"))

        # --- derivable rule 3: arXiv identifier encodes YYMM ------------------
        m = re.search(r"arXiv[:\s]*(\d{2})(\d{2})\.\d{4,5}", venue + " " + f.get("note", ""))
        if m and yr:
            exp = 2000 + int(m.group(1))
            checked.append("arXiv id year")
            if not (exp <= yr <= exp + 1):
                findings.append((key, f"arXiv id implies {exp} but year is {yr}"))
            mo = int(m.group(2))
            if not 1 <= mo <= 12:
                findings.append((key, f"arXiv id has impossible month {mo:02d}"))

        # --- derivable rule 4: year appears in the proceedings title ----------
        m = re.search(r"\b(19|20)\d{2}\b", venue)
        if m and yr and typ == "inproceedings":
            checked.append("year in venue")
            if int(m.group(0)) != yr:
                findings.append((key, f"venue names {m.group(0)} but year field is {yr}"))

        # --- derivable rule 5: page range direction and plausibility ----------
        if "pages" in f:
            p = f["pages"].replace("--", "-").strip()
            if re.fullmatch(r"\d+-\d+", p):
                a, b = map(int, p.split("-"))
                checked.append("page range")
                if b < a:
                    findings.append((key, f"reversed page range {f['pages']}"))
                elif b - a > 400:
                    findings.append((key, f"implausible page span {f['pages']}"))

        # --- derivable rule 6: surname spelling consistent across entries -----
        for s in surnames(f.get("author", "")):
            norm = re.sub(r"[^a-z]", "", s.lower())
            if norm and norm in seen_surnames and seen_surnames[norm] != s:
                findings.append((key, f"surname spelled '{s}' here but "
                                      f"'{seen_surnames[norm]}' elsewhere"))
            elif norm:
                seen_surnames[norm] = s

        # --- worksheet row ----------------------------------------------------
        title = f.get("title", "").strip("{}")
        first = surnames(f.get("author", ""))[:1]
        query = quote(f"{title} {first[0] if first else ''}".strip())
        worksheet.append(dict(
            key=key, type=typ, year=year, venue=venue[:60],
            volume=f.get("volume", ""), pages=f.get("pages", ""),
            has_doi="yes" if "doi" in f else "NO",
            auto_checked="; ".join(checked) if checked else "none",
            lookup=(f"https://doi.org/{f['doi']}" if "doi" in f
                    else f"https://scholar.google.com/scholar?q={query}"),
        ))

    print("=" * 72)
    if findings:
        print(f"INCONSISTENCIES FOUND ({len(findings)})")
        for k, msg in findings:
            print(f"  {k:18s} {msg}")
    else:
        print("No inconsistency found by the derivable rules.")
    print("=" * 72)

    auto = sum(1 for w in worksheet if w["auto_checked"] != "none")
    doi = sum(1 for w in worksheet if w["has_doi"] == "yes")
    print(f"\nentries with a DOI (verifiable in one click) : {doi}/{len(worksheet)}")
    print(f"entries checked by a derivable rule           : {auto}/{len(worksheet)}")
    covered = sum(1 for w in worksheet
                  if w["has_doi"] == "yes" or w["auto_checked"] != "none")
    print(f"entries covered by at least one of the two    : {covered}/{len(worksheet)}")
    print(f"entries needing fully manual checking         : {len(worksheet)-covered}")

    try:
        import pandas as pd
        df = pd.DataFrame(worksheet).sort_values(["has_doi", "auto_checked"])
        df.to_csv("bib_verification_worksheet.csv", index=False)
        print("\nwrote bib_verification_worksheet.csv")
        rest = df[(df.has_doi == "NO") & (df.auto_checked == "none")]
        if len(rest):
            print("\nFully manual (open the lookup URL and confirm volume/pages):")
            for _, r in rest.iterrows():
                print(f"  {r.key:18s} {r.venue}")
    except ImportError:
        import csv
        with open("bib_verification_worksheet.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(worksheet[0]))
            w.writeheader()
            w.writerows(worksheet)
        print("\nwrote bib_verification_worksheet.csv")


if __name__ == "__main__":
    main()
