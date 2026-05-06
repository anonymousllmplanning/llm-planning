# Paper Result Artifacts

Sanitized aggregate CSV files used to audit the reported results. Files are grouped by paper section or appendix role:

- `main_tables/`: headline open-weight GAIA tables and tool-clean diagnostics.
- `appendix_c_data_profile/`: Augmented GAIA construction, accounting, and reference-validated parallelism summaries.
- `appendix_e_replay_filter/`: Gemma 4 replay-filter retention and outcome typology summaries.
- `appendix_closed_model_extension/`: OpenAI/Gemini closed-model extension aggregates.
- `appendix_cross_benchmark/`: TaskBench and UltraTool auxiliary cross-benchmark summaries.
- `reference_diagnostics/`: best-reference and threshold diagnostics for flexible-order scoring.
- `tool_scoring_audits/`: aggregate audits for tool and parameter-value scoring.

The repository intentionally excludes raw GAIA records, attachments, per-task qualitative examples, and paper source/PDF files.
