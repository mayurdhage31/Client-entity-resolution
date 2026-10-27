# Indian banking/finance demonstration

This local personal project uses **real public company names with simulated
records and errors**. It contains no actual customer data or genuine bank system
records and has no affiliation with, or endorsement by, the named companies.
All addresses, six-digit PIN codes and `DEMO-IN-...` identifiers are invented for
teaching. They are not verified addresses, genuine CINs, IFSCs or account numbers.

## Launch

First create a Python 3.12 virtual environment and install `requirements.txt`
as described in the [README](README.md). Then run from the project root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\app.py
```

Open **http://127.0.0.1:8501**. Stop the local server with `Ctrl+C`. `requirements.txt` is separate from the optional embedding experiment. The app still works without torch or sentence-transformers.
On Linux the launch command is `./.venv/bin/python -m streamlit run app.py`.
Nothing has been published or deployed.
If Streamlit is already running, stop and restart it to load these changes;
automatic file watching is disabled in the local configuration.

## A short walkthrough using the finished records

1. **Start here:** the opening cards show `HDFC Bank Limited` from `DEMO_CORE` and
   `HDFC Bnak Ltd.` from `DEMO_CRM`. Select **Try an example**.
2. **Compare two company records:** the guided example selects these actual
   dropdown entries:
   - `HDFC Bank Limited | DEMO_CORE | 6c031199`
   - `HDFC Bnak Ltd. | DEMO_CRM | 37f8a88b`
   The default mode is **Choose existing records**. Fields fill automatically and
   are read-only. Select **Compare selected records** to see the actual frozen
   model decision, uncalibrated score, saved threshold and observed evidence.
3. Change the **Guided example** dropdown to try the other cases below. The
   educational expectation comes from separate simulation labels; it is not a
   promise that the model will agree. Model mistakes are not overwritten.
4. **Explore duplicate groups:** select an observed company name, source system,
   or group. For example, choose `HDFC Bank Limited` in **Company name as recorded**.
   The selected group still shows every saved member, including other spellings
   and sources. Try **Single-record groups** or **Only groups flagged for review**.
   Full technical IDs and link evidence are in an expander.
5. **Explore corporate relationships:** select
   `Bajaj Finserv Limited | 2 separate companies`. The arrow points to
   `Bajaj Finance Limited`. Open the official source link and note the retrieval
   date. This is supplied public reference information, not a matching-model
   prediction or an exhaustive ownership structure.

| Guided example | Actual Record A label | Actual Record B label | Educational expectation only |
| --- | --- | --- | --- |
| Same entity with spelling differences | HDFC Bank Limited \| DEMO_CORE \| 6c031199 | HDFC Bnak Ltd. \| DEMO_CRM \| 37f8a88b | Same legal entity |
| Same entity with missing details | ICICI Bank Limited \| DEMO_CORE \| 386ecbe0 | ICICI Bank \| DEMO_LEGACY \| 571aa876 | Same legal entity |
| Different entities | State Bank of India \| DEMO_CORE \| a3b1799d | Axis Bank Limited \| DEMO_CORE \| f50bea63 | Different legal entities |
| Related but separate companies | Bajaj Finserv Limited \| DEMO_CORE \| b45ed1f0 | Bajaj Finance Limited \| DEMO_CORE \| 89463e85 | Different legal entities, despite the supplied relationship |

Both record dropdowns are searchable and populated from the Indian CSV, with
labels `company name | source system | short record ID`. Record A is excluded
from Record B. Changing records, scenarios, mode or edits clears the prior score.
There is one comparison button and no separate Apply buttons.

To experiment with input errors, choose **Advanced: edit a copy** and open the
section with that name. Edits change only temporary session copies. They never
write back to CSVs or change saved predictions. Switching back to **Choose
existing records** restores the selected saved fields. Empty copies are rejected.
Direct comparisons bypass candidate blocking; they do not update any benchmark.

## Five pages in plain English

Each page begins with **What this page does**, **What to select**, and **How to
interpret the result**. The navigation is:

1. Start here
2. Compare two company records
3. Explore duplicate groups
4. Explore corporate relationships
5. How it works and limitations

Duplicate groups show saved model predictions. Solid graph lines are direct
matches; a path A–B–C also puts A and C in the same group without creating a new
A–C match line. One wrong link can join separate companies. Singletons remain
visible. Review flags identify conflicting nonempty demo identifiers in one
country and never change membership. Downloads contain the selected group's
original observed records. Import identifier and PIN columns as text in a
spreadsheet to preserve their text representation.

The company filter uses **observed names as recorded**, including typos, rather
than assigning a canonical company identity using truth. Filters select whole
groups with at least one matching member. The group is never reduced to just the
matching rows. Simulation truth is available only in the explicitly labelled
educational expander and is not used for grouping or hierarchy linking.

## Data, predictions and provenance

The Indian demo has **24 records**, four simulated source records for each of
six named companies. Its one unchanged-model batch run generated **66 candidates,
28 direct predicted match links and 8 predicted groups**, including **3 singletons
and 1 review flag**. These are descriptive counts, **not accuracy measurements**.
No Indian-demo accuracy, precision, recall or F1 was calculated or invented.

| Path | Contents |
| --- | --- |
| `data/india_demo/observed/client_records.csv` | Eight observed fields using the original schema: record ID, demo source system, company name, fake registration ID, address, city, postcode, country. |
| `data/india_demo/reference/companies.csv` | Six public canonical names, official source URLs, retrieval dates and verification notes. Used for public reference display, not model features. |
| `data/india_demo/reference/public_relationships.csv` | The one verified parent-to-child relationship, type, source, source URL and retrieval date. |
| `data/india_demo/ground_truth/record_entity_map.csv` | Separate simulation labels for teaching expectations only. |
| `data/india_demo/education/guided_examples.json` | Four pointers to actual record pairs and explicitly educational expected outcomes. |
| `data/india_demo/dataset_info.json` | Scope, seed, counts, identifier/address policies and disclaimer. |
| `outputs/india_demo/v1/cleaned_records.csv` | Original Indian fields plus the existing normalizer's fields. |
| `outputs/india_demo/v1/candidate_pairs.csv`, `candidate_predictions.csv` | Fresh candidates with provenance; observed-only features, model scores and decisions. No truth targets are included. |
| `outputs/india_demo/v1/clusters/` | Fresh deterministic memberships, direct links, summaries and review flags for all Indian records. |
| `outputs/india_demo/v1/prediction_run.json` | Frozen model hash, threshold, feature order, unchanged retrieval settings, input/output hashes and one-run provenance. |

`build_india_demo.py` was run **once** to create these separate files. It uses the
existing normalizers, 21 features, exact/fuzzy candidate rules and selected
Stage 3 model with its saved **0.5** threshold. The same connected-component and
review-flag functions build fresh Indian groups. No original predictions or
memberships were copied or relabelled. The prediction function accepts only
observed records; truth and public relationships are not inputs. Teaching labels
are saved after prediction and never repair its results. No model was retrained
or configuration tuned in response to the new outputs.

The builder refuses to overwrite an existing dataset or output run. To verify
saved hashes **without rerunning predictions**, use:

```powershell
.\.venv\Scripts\python.exe -B .\build_india_demo.py --verify
```

Original v1 records, models, reports, thresholds, manifests and evaluation outputs
remain unchanged. On **Start here**, historical metrics are inside **Original v1
benchmark details — separate from the Indian demo** and explicitly labelled as
original v1 results, never Indian-demo performance. The older Stage 6 strict
inventory check predates additive UI/demo scripts; do not edit its manifest or
rerun test evaluation to launch the app.

## Official public sources

All source rows were verified and retrieved on **2026-10-26**. Only company names
and the specified relationship were taken from these pages. No addresses,
identifiers or customer/system records were copied.

| Public canonical name | Official name source |
| --- | --- |
| State Bank of India | [Official contact page](https://sbi.bank.in/en/web/customer-care/contact-us) |
| HDFC Bank Limited | [Official company overview](https://www.hdfc.bank.in/about-us/who-we-are) |
| ICICI Bank Limited | [Official bank page identifying the company as incorporated in India](https://www.icicibank.hk/about_us) |
| Axis Bank Limited | [Official investor FAQs](https://www.axis.bank.in/shareholders-corner/investor-faqs) |
| Bajaj Finance Limited | [Official Bajaj Finance overview](https://www.aboutbajajfinserv.com/finance-about-us) |
| Bajaj Finserv Limited | [Official Bajaj Finserv overview](https://www.aboutbajajfinserv.com/about-us) |

The [Bajaj Finance overview](https://www.aboutbajajfinserv.com/finance-about-us)
describes Bajaj Finance as a subsidiary of Bajaj Finserv. That supports the
supplied arrow **Bajaj Finserv Limited → Bajaj Finance Limited**. No extra
relationships were inferred from similar names. This is a dated public reference
snapshot; it does not claim a complete or real-time ownership register.

## Checks and limitations

```powershell
.\.venv\Scripts\python.exe -B -m unittest test_india_demo test_app -v
```

Recorded result from the app development run: **31 tests passed** in 64.049
seconds. This is historical evidence, not a fresh-clone validation result; see
[publication notes](docs/PUBLICATION.md) for the current preparation checks. Checks cover all five pages,
record/dropdown population, autofill, A/B exclusion, scenario selection, stale
results, edited-copy isolation, empty inputs, actual Indian model inputs,
incorrect predictions remaining visible, whole-group filters, singleton/review
controls, source dates, dataset separation, frozen hashes and safe labels. The app
and live scoring also pass with optional encoder imports blocked. Fixture tests
use fake model outputs to test prediction plumbing; the saved Indian dataset has
only one real batch run. The original held-out evaluation was not rerun.

This is a teaching demo, not banking policy, verified company identity or a
production system. The original model learned from a different synthetic
distribution; abbreviations such as SBI and sparse records may fail retrieval or
matching. Missing evidence is not positive evidence. Scores are uncalibrated,
and graph transitivity can amplify errors. Public names and simulated records do
not establish real-bank performance or permission to merge records.
