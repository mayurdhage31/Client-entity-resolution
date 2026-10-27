# Publication preparation

Prepared on 2026-10-28 for a public GitHub portfolio repository.

## Repository identity

GitHub repository: [mayurdhage31/Client-entity-resolution](https://github.com/mayurdhage31/Client-entity-resolution)

Suggested description:

> Python and Streamlit portfolio: synthetic company-record deduplication, interpretable matching, leakage-safe evaluation and separate corporate hierarchy exploration.

Suggested topics: `entity-resolution`, `record-linkage`, `data-quality`,
`machine-learning`, `scikit-learn`, `streamlit`, `graph-analysis`, `synthetic-data`.

The project uses the [MIT license](../LICENSE), selected by the owner. It includes
original synthetic artifacts and dated public company references. No real
customer data or real banking-system records are included. Public names and
source pages retain their respective rights; there is no company endorsement.

## Included and excluded files

The public file set includes source, tests, documentation, small model binaries,
synthetic/reference CSVs, saved reports and static HTML graph examples. These
artifacts allow the app to run without retraining. The two approximately 4.6 MiB
HTML files contain self-contained Plotly views and belong to the frozen evidence.
Download and open them locally for interactive viewing. Their bundled
JavaScript is excluded from source diffs and whitespace checks.

Virtual environments, Python caches, encoder weights, package downloads, local
test/server logs and common credential files are ignored by the existing
`.gitignore`. The historical embedding installation log is intentionally included
because it is protected benchmark evidence; the publication screen found no
credential or private machine-path patterns in it.

The original `.gitignore` is one of the 78 frozen files and remains unchanged.
`.gitattributes` disables automatic text conversion for every file so a clone
preserves the checksums even with `core.autocrlf=true`. Generated data, model and
report artifacts are marked as generated for GitHub language statistics.

## Validation performed

The preparation checks and their outcomes are recorded below. Current commands
are in the [README](../README.md). Logs and temporary validation files stay in
ignored `.cache/` paths.

- The standard-library publication checker verifies 131 unique saved artifact
  hashes across the original freeze, completion records, saved reproducibility
  rerun and Indian demo. All matched.
- The publishable text scan found no screened credential patterns or private
  machine paths. No environment/cache paths or files at least 50 MiB were in
  the public file set. This is a limited pattern scan, not an exhaustive
  secret-detection guarantee.
- Git ignore checks covered local caches, the virtual environment, `.env`,
  `.env.local` and `.streamlit/secrets.toml`.
- All 26 current Python sources parsed successfully. The candidate preparation
  suite passed: **28 tests, 0 failures**.
- The original installed environment's 70 packages passed `uv pip check`.
- A fresh dependency resolution and installation of `requirements-dev.txt`
  succeeded in an isolated Python 3.12 environment. All 44 installed packages
  passed the compatibility check; no torch or sentence-transformers was installed.
- The workflow YAML parsed with the expected events, seven steps and Windows/Linux
  matrix. All local links in the current documentation resolve.
- An isolated Git export made with `core.autocrlf=true` passed the publication
  checker, confirming the saved bytes survive the Git staging/export path.
  Negative checks in that isolated copy rejected a changed frozen CSV and a
  dummy credential pattern without printing its value.

Full-suite execution was attempted in the existing environment, including a
retry outside the filesystem sandbox. Both attempts encountered the Windows
Application Control error **"DLL load failed while importing _ctypes"**. Pandas
and Streamlit-dependent modules could not load. This does not constitute a
passing full-suite result or a current visual/browser validation. Historical
results in the development notes and guide are labelled as historical.

The added GitHub Actions workflow runs snapshot checks and the offline unit and
Streamlit tests on Windows and Linux with Python 3.12. It has not run on GitHub
during local preparation. Its first successful run remains a publication check.
Actions have read-only repository permissions, credentials are not persisted,
and action versions are pinned to full commit hashes. Release references:
[checkout v4.2.2](https://github.com/actions/checkout/releases/tag/v4.2.2) and
[setup-python v5.6.0](https://github.com/actions/setup-python/releases/tag/v5.6.0).
Dependabot is configured to propose GitHub Actions updates. Python dependency
upgrades require separate model-compatibility review.

## First publication

The local repository is initialized on `main`. Review the file set and ensure
your Git author name/email are what you want visible in a public commit.
Then run from the project root with the virtual environment's Python:

```text
python scripts/check_publication.py
git status --short
```

The target GitHub repository has already been created. Preserve any existing
remote history and use a normal, non-forced push. After reviewing the files,
make the publication commit and connect the repository (reuse `origin` if it
already has this exact URL):

```text
git add .
git diff --cached --stat
git diff --cached --check
git commit -m "Publish banking client entity resolution portfolio"
git remote add origin https://github.com/mayurdhage31/Client-entity-resolution.git
git push -u origin main
```

Check the first Actions run and
launch the app in an unrestricted Python 3.12 environment before presenting the
repository as fully validated. Enable private vulnerability reporting in GitHub
if you want the reporting route described in [SECURITY.md](../SECURITY.md).
