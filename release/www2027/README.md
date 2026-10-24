# WWW 2027 source and core-result release

- `framework_source.zip`: source, synthetic tests, current diagram assets, and the final reference schedule.
- `core_results.zip`: ten manuscript-aligned aggregate tables (87 rows), documentation and table manifest.
- `SOURCE_MANIFEST.json`: source file sizes and SHA-256 hashes.
- `SHA256SUMS`: hashes for published files and both release archives.
- `VALIDATION.json`: executed checks and their scope.

The source archive excludes the result tables; obtain `core_results.zip` alongside
it for those tables. The framework and result archives contain no raw task
records, model answers, individual trajectories, credentials, or manuscript
source files. The release publishes the experimental framework, result tables,
reference schedules, and architecture figures.
Aggregate numbers retain manuscript display precision.

Reference accounting is identical across the repository and both archives:
1,786 sampled alternatives; 1,429 retained alternatives; 165 Native chains;
1,594 final references. `references/reference_counts.json` provides the ledger.
