# Manuscript reference data

`paper_final_success.csv` contains the 42 displayed per-task/method aggregate endpoint
values in the supplied manuscript, Appendix L, table `tab:final_perform`. Units are
success fractions (0–1), at 400,000 online environment steps. The manuscript reports
mean ± standard error over five seeds. These are reported aggregate observations,
not raw seed measurements. No seed-level observations or learning curves were inferred.

`paper_final_success_source.tex` preserves the exact source table, including its
rounded aggregate rows. `provenance.json` identifies the original source and its hash.
The CSV is a numeric transcription of the six task rows in the source column order;
LaTeX emphasis was removed. Rows were not filtered based on performance.

The training/plotting commands do not use these target values. Each new experiment
exports its own evaluation observations and exact plotted aggregates.
