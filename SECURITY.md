# Security and data handling

This is an educational portfolio demonstration using synthetic records and
dated public company references. It has no authentication or production banking
controls. Use the default local server binding for the demo.

Do not commit real customer records, private bank documents, API keys, passwords,
environment files or model-download caches. The app requires no credentials.
`scripts/check_publication.py` checks the Git publication file set for common
credential patterns and private machine paths. This is a limited automated
check, not proof that every possible secret has been detected.

The saved `.joblib` models are Python pickle-based artifacts. Load only the
trusted project copies. Hash checks detect differences from the saved snapshot;
they do not establish that an arbitrary replacement model or manifest is safe.

Report vulnerabilities using GitHub's private vulnerability reporting feature
when enabled on the repository. If it is unavailable, ask the maintainer for a
private reporting channel without posting sensitive details in a public issue.
Ordinary reproducible bugs with synthetic inputs can use public issues.

There is no production support or security response-time commitment. Any future
public deployment needs a separate review of access controls, data handling,
dependency updates and hosting configuration.
