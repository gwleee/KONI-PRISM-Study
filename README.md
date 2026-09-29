# Audit artifact package

Engineering and Auditing an On-Premise Multimodal RAG System: Performance, Failure Modes, and
Lessons Learned (Journal of Systems and Software, submitted).

Archived at https://doi.org/10.5281/zenodo.23029708 · Source: https://github.com/gwleee/KONI-PRISM-Study

This package contains the evaluation scripts, the benchmarks in both their original and rebuilt
forms, the audits that invalidated the originals, the metric-control validation, the latency
harness, and the export that fixes every number in the manuscript to a file on disk. It does not
contain the platform implementation, which is subject to ongoing technology-transfer
negotiations.

## Layout

| Directory | What it holds |
|---|---|
| `common/` | Shared harness (provenance, statistics), the embedding-distance conflict check, the NLI faithfulness scorer |
| `synconflict_v1/` | Original synthetic conflict set (N=100) and its generator. Superseded, kept on purpose: its generator was built so that the deployed threshold is optimal |
| `synconflict_v2/` | Rebuilt conflict set (N=600), generator, evaluation of both v1 and v2, threshold curves |
| `external_eval/` | Contradiction detection on SNLI, MultiNLI and VitaminC samples (1000 pairs per class) |
| `hmmr_v2/` | MultiTurnSet v2: segment builder, question/caption generation, assembly, independence audit, retrieval run, analysis |
| `ablation_v2/` | Grounded evaluation set, component ablation, metric-control validation |
| `mars_attribution/` | Attribution evaluation set, threshold sweep, random and within-query-shuffle controls |
| `sensitivity/` | Parameter sweeps and the reachability analysis of the trust-dominant branch |
| `latency/` | Latency and cold-start harness, the original run and two re-runs, solver re-measurement, container overlay |
| `ilp_fix/` | Results of the corrected routing read-back (the solver itself is withheld; see `ilp_fix/README.md`) |
| `config_snapshot/` | The deployed routing registry and configuration, read from the platform code |
| `export/` | `paper_numbers.csv` (every value the manuscript cites, with source file, JSON path and SHA-256 prefix) and `manifest.json` |
| `figures/` | Regeneration of Fig. 2 and Fig. 3 from the export |

## Manuscript to artifact

| Manuscript | Script | Artifact |
|---|---|---|
| §4.2 metric calibration | `ablation_v2/validate_metric.py` | `ablation_v2/results/metric_validation.json` |
| §5.1, Fig. 2 | `latency/run_latency.py`, `figures/regen_jss_figs.py` | `latency/results/results_a100_1.json`, `latency/results_rerun_20260928*/` |
| §5.2 cold start | `latency/run_latency.py` | same, `cold_start_encoder` block |
| §5.3, Table 2, Fig. 3 | (solver withheld), `latency/remeasure_dmar.py` | `ilp_fix/results/ilp_fix_summary.json`, `latency/results/dmar_remeasure.json` |
| §5.4 conflict screening | `synconflict_v2/evaluate_cmae_v2.py`, `external_eval/run_external_eval.py` | `synconflict_v2/results/results.json`, `external_eval/results/results.json` |
| §5.5 memory retrieval | `hmmr_v2/run_hmmr_v2.py`, `hmmr_v2/analyze.py`, `hmmr_v2/audit_dataset.py` | `hmmr_v2/results/analysis.json`, `memory_state_seed*.json`, `dataset_audit.json` |
| §5.6 attribution | `mars_attribution/run_mars_eval.py`, `mars_attribution/analyze_query_level.py` | `mars_attribution/results/results.json`, `query_level.json` |
| §5.7 ablation | `ablation_v2/run_ablation.py`, `ablation_v2/analyze_results.py` | `ablation_v2/results/analysis.json` (`query_level` block: unit = query, n=30) |
| §6 L1 (trust-dominant branch) | `sensitivity/analyze_theta_dom.py` | `sensitivity/results/theta_dom_reachability.json` |
| Table 1, §3.3, §5.2 (model footprints, quality floors) | `config_snapshot/snapshot_registry.py` | `config_snapshot/results/registry_snapshot.json`, `synconflict_v2/results/results.json` (settings block), `ilp_fix/results/ilp_fix_summary.json` |

`export/paper_numbers.csv` gives the exact file and JSON path of each cited value. To check that
the shipped artifacts still yield those values, run `python export_paper_numbers.py` from the
package root; it rewrites the CSV from the artifacts. The SHA-256 prefixes in the CSV refer to the
internal copies of the artifacts, which differ from the shipped ones only in redacted provenance
fields (host names, local paths, branch names); the values are identical.

## Environment

Python 3.11 is required.

```bash
conda create -y -n artifact python=3.11
pip install torch==2.5.1 transformers==4.47.1 FlagEmbedding==1.3.5 \
    sentence-transformers==3.3.1 numpy==2.1.3 scipy==1.14.1 pulp==2.9.0 \
    pandas matplotlib scikit-learn datasets==4.8.4 qdrant-client==1.12.1 \
    httpx==0.28.1 structlog==24.4.0 pypdf
```

Measurements were taken on Ubuntu 22.04 with CUDA 12.1 on an NVIDIA A100 80 GB PCIe.

## What runs without the platform

The following run on their own:

- dataset builders: `synconflict_v1/build_conflict_set.py`, `synconflict_v2/build_synconflict_v2.py`,
  `ablation_v2/build_evalset.py`, `mars_attribution/build_mars_evalset.py`,
  `hmmr_v2/build_segments.py`, `hmmr_v2/gen_llm.py` (needs vLLM and the Qwen models),
  `hmmr_v2/assemble.py`;
- metric-control validation: `ablation_v2/validate_metric.py` (downloads
  `joeddav/xlm-roberta-large-xnli`);
- analysis: `ablation_v2/analyze_results.py`, `mars_attribution/analyze_query_level.py`, `hmmr_v2/analyze.py`;
- the export and the figures: `export_paper_numbers.py`, `figures/regen_jss_figs.py`.

The evaluation harnesses import the platform's gateway package at run time (`common/harness.py`,
`bootstrap_gateway`), so they measure the deployed code rather than a reimplementation. They do
not run without it: `synconflict_v2/evaluate_cmae_v2.py`, `external_eval/run_external_eval.py`,
`common/cmae_eval.py`, `sensitivity/*`, `ablation_v2/run_ablation.py`,
`mars_attribution/run_mars_eval.py`, `hmmr_v2/run_hmmr_v2.py`, `hmmr_v2/audit_dataset.py`,
`config_snapshot/snapshot_registry.py`.
`latency/*` additionally needs the platform's services (language and speech containers, vector
store); `latency/docker-compose.latency.yml` documents the overlay used. Their inputs and raw
outputs are shipped, so every reported number can be re-derived without re-running them.

The corrected routing read-back behind Table 2 and Fig. 3 is withheld because it is a modified
copy of the platform's solver; its results are included in `ilp_fix/results/`.

## Data sources

- SynConflict v1 and v2, the ablation and attribution evaluation sets, and MultiTurnSet v2 turns
  are generated by the scripts in this package with fixed seeds.
- `external_eval/data/` holds samples of SNLI, MultiNLI and VitaminC, redistributed under the
  licenses of those datasets.
- MultiTurnSet v2 is built from public KLUE reading-comprehension passages, ten arXiv papers, and synthetic chart images. Its segments quote the
  passages and papers under their original licenses; the source documents themselves are not
  included.

## Licensing

Code is released under the MIT License; synthetic benchmark data under CC BY 4.0.

## Notes

- Source comments are in Korean; this README and the manuscript describe each script.
- Generated answer texts (`raw_records`) are omitted for size; the per-query metric tables and
  all derived results are included.
