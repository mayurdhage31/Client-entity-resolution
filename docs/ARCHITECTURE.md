# Architecture and data

The app reads saved, hash-checked artifacts and scores optional record comparisons.
It does not run the training, data-generation or evaluation entry points.

## Matching pipeline

1. Normalize Unicode, punctuation, company suffixes, address tokens and country
   variants, while preserving identifiers as strings and missing values as empty.
2. Keep entire corporate families in one data partition before pair generation.
3. Retrieve candidate pairs through exact blocking plus bounded fuzzy fallbacks.
   Missing values do not form agreement keys.
4. Build 21 observed-only features: fuzzy company/address similarities, exact
   field agreements, registration disagreement and missingness indicators.
5. Compare illustrative rules with a training-only StandardScaler/logistic
   regression pipeline. The selected model uses a saved threshold of 0.5.
6. Build an undirected graph of direct predicted matches and take connected
   components. Include isolated records; flag conflicting identifiers for review
   without changing membership.
7. Join truth only for evaluation or explicitly labelled teaching expectations.
   Render supplied corporate references through a separate graph.

Record IDs are opaque join keys. Ground-truth entity IDs, canonical reference
names, family IDs, split labels and corporate relationships are not matching
features. Transitive group membership does not create a direct scored edge.

## Two distinct datasets

| Dataset | Observations | Truth | Corporate reference |
| --- | --- | --- | --- |
| Original benchmark | 757 fictional records, 300 fictional entities, five simulated source systems | Separate canonical entities and record mappings; family-grouped train/validation/test split | Fully synthetic parent relationships |
| Indian teaching demo | 24 simulated records using six real public company names | Separate teaching labels; no performance evaluation | One supplied, dated public parent/subsidiary relationship with official source URL |

Both observed datasets use the same eight fields:

```text
record_id,source_system,company_name,registration_id,address,city,postcode,country
```

The Indian demo's addresses, PIN codes and `DEMO-IN` identifiers are invented.
Names and relationship references are dated public facts. Sources and retrieval
dates are in `data/india_demo/reference/` and the [app guide](../APP_GUIDE.md).
The original generator's claim that every company is fictional applies to the
original benchmark, not to the separate Indian reference names.

## Artifacts and trust

`portfolio_ui/artifacts.py` validates runtime files against the original freeze
and completion manifest. `portfolio_ui/india.py` validates the separate teaching
run. Neither changes a freeze. `scripts/check_publication.py` verifies all saved
protected hashes, original and rerun completion outputs, and Indian-run inputs
without importing scientific libraries or unpickling a model.

`.gitignore` itself belongs to the 78-file original freeze. It already excludes
environments, caches and common credential files. `.gitattributes` disables text
normalization so cloning on Windows or Linux preserves the raw bytes covered by
these hashes. The small model binaries and synthetic artifacts are intentionally
included so the app works after dependency installation without training.

The saved installation log and package inventories are historical experiment
evidence included in the freeze. Local cache/server/test logs stay under the
ignored `.cache/` directory. A publication scan checks the included text for
common credential patterns and private machine paths.

## Reproduction and experiments

Launching the app and checking existing artifacts reproduce the saved portfolio
experience. Unit tests use isolated temporary data for pipeline behavior. They do
not repeat the original held-out benchmark.

The older evaluator also enforces an exact inventory and package environment
from before the app was added. That strict historical entry point is not the
publication checker and should not be rerun in this expanded checkout.

To explore a different model or dataset, copy the pipeline into a separate
experiment and use new versioned artifact paths. The historical
[development notes](DEVELOPMENT_HISTORY.md) explain the generator, blocking,
training, optional embedding comparison, graph construction and evaluation
lifecycle. Install `requirements-dev.txt` for Faker-based generation. The app
and offline tests need only `requirements.txt`; the optional real encoder
experiment has its own frozen dependency file and requires model downloads.
