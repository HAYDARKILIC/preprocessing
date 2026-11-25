# Publishing this repository

## Anonymity — decide this first

Under single-blind review a named repository is fine, because reviewers already
know who you are. **Check the review model on the target venue's "Guide for
Authors" page before you publish**, because under double-blind review a named
repository breaks anonymity.

If you need an anonymised version:

```bash
# strip identifying material
rm CITATION.cff
sed -i 's/Haydar Kilic/Author (name withheld for review)/g' LICENSE README.md
sed -i '/orcid/Id' README.md
```

Then publish under a neutral account, or use https://anonymous.4open.science,
which proxies a GitHub repository behind an anonymous URL and is accepted by most
publishers.

## Creating the repository

```bash
cd preprocessing
git init
git add -A
git commit -m "Replication code and results"
git branch -M main
git remote add origin https://github.com/HAYDARKILIC/preprocessing.git
git push -u origin main
```

Repository size is about 16 MB, well inside GitHub's limits. No file approaches
the 100 MB per-file cap, so Git LFS is not needed.

Suggested repository description:

> Replication code and per-cell results for a factorial study calibrating
> preprocessing effort against learner choice across 95 tabular datasets.

Suggested topics: `reproducible-research`, `benchmark`, `tabular-data`,
`data-preprocessing`, `automl`, `meta-learning`.

## Archiving for the DOI

The data-availability statement needs a DOI, which GitHub does not provide.
Link the repository to Zenodo (Zenodo → GitHub → enable the repository), then cut
a release; Zenodo mints a DOI automatically and archives that snapshot.

Alternatively upload the standalone archive prepared separately
(`preprocessing-artefacts.zip`), which already carries `.zenodo.json`,
`CITATION.cff` and a SHA-256 manifest.

Once you have the DOI, add it to the top of `README.md`:

```markdown
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
```

## What is deliberately absent

- **The manuscript.** Code, data and figures only, as intended.
- **The benchmark datasets.** Not redistributed; they are identified by permanent
  OpenML identifier and served revision, with SHA-256 fingerprints in
  `data/dataset_fingerprints.csv` so identity can be confirmed after retrieval.
- **Checkpoint files.** `results_checkpoint*.csv` duplicate the final result
  tables and were excluded to keep the repository lean.

## A note for the editor

The repository is self-checking. `notebooks/02_analysis_and_verification.ipynb`
re-derives every quantity reported in the article from `data/` and asserts that
each matches, ending with a pass/fail summary. A reviewer can confirm the paper's
numerical claims in about two minutes without reading any analysis code.
