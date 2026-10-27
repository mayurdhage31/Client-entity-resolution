# Contributing

Use Python 3.12. Create a virtual environment and install
`requirements-dev.txt`; see the [README](README.md) for platform-specific setup.

Before opening a pull request, run from the project root:

```text
python scripts/check_publication.py
python -B -m unittest discover -s . -p "test_*.py" -v
```

The tests use temporary fixtures. They exercise the evaluation lifecycle without
rerunning the saved held-out benchmark or downloading an encoder.

The v1 benchmark, Indian demonstration data, model binaries, manifests and
protected pipeline sources are a historical snapshot. Keep them byte-for-byte
intact when changing documentation, packaging or the app. `.gitignore` is also
part of the freeze. `.gitattributes` disables automatic line-ending conversion
because the integrity checks hash raw bytes.

For model, retrieval or data changes, create a separate experiment with separate
artifacts. Do not update old hashes to make changed results pass. Do not tune on
the published held-out results. The original evaluator's strict inventory check
predates the added app and publication files; use `scripts/check_publication.py`
for snapshot verification rather than creating or rerunning the original freeze.

Keep observed matching inputs separate from ground truth and corporate reference
relationships. Preserve missing values as missing, identifiers as text, and the
distinction between duplicate companies and related legal entities.

Describe the behavior changed and checks run in your pull request. Use synthetic
examples only; never attach customer data or credentials. See
[SECURITY.md](SECURITY.md) for sensitive findings.
