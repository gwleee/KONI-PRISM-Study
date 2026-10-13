"""
Freeze the numbers the manuscript is allowed to cite.

Every value is read from an experiment artifact on disk and written to
`paper_experiments/export/paper_numbers.csv` with the file it came from, that file's sha256, and
the commit the export ran at. Nothing is typed in by hand; a number that is not in this file has
no artifact behind it and must not appear in the manuscript.

  python export_paper_numbers.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "export"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent,
                          capture_output=True, text=True).stdout.strip()


def jload(rel: str):
    p = ROOT / rel
    return json.loads(p.read_text(encoding="utf-8")), p


def dig(obj, path: str):
    """Path segments are separated by '/' because artifact keys contain dots and pipes."""
    cur = obj
    for k in path.split("/"):
        cur = cur[int(k)] if isinstance(cur, list) else cur[k]
    return cur


SPEC = [
    # ---- operational characterisation (A100, measured) ----
    ("lat.dmar_ms", "latency/results/results_a100_1.json", "stage_latency_ms/dmar_route/mean", 2, "sec5.1 stage table"),
    ("lat.dmar_p95_ms", "latency/results/results_a100_1.json", "stage_latency_ms/dmar_route/p95", 2, "sec5.1"),
    ("lat.cmae_ms", "latency/results/results_a100_1.json", "stage_latency_ms/cmae_encode/mean", 2, "sec5.1"),
    ("lat.cmae_p95_ms", "latency/results/results_a100_1.json", "stage_latency_ms/cmae_encode/p95", 2, "sec5.1"),
    ("lat.hmmr_ms", "latency/results/results_a100_1.json", "stage_latency_ms/hmmr_retrieve/mean", 2, "sec5.1"),
    ("lat.hmmr_p95_ms", "latency/results/results_a100_1.json", "stage_latency_ms/hmmr_retrieve/p95", 2, "sec5.1"),
    ("lat.mars_ms", "latency/results/results_a100_1.json", "stage_latency_ms/mars_score/mean", 2, "sec5.1"),
    ("lat.mars_p95_ms", "latency/results/results_a100_1.json", "stage_latency_ms/mars_score/p95", 2, "sec5.1"),
    ("lat.specialist_ms", "latency/results/results_a100_1.json", "stage_latency_ms/specialist_infer/mean", 2, "sec5.1"),
    ("lat.generate_ms", "latency/results/results_a100_1.json", "stage_latency_ms/llm_generate/mean", 2, "sec5.1"),
    ("lat.total_ms", "latency/results/results_a100_1.json", "stage_latency_ms/total/mean", 2, "sec5.1 denominator"),
    ("lat.total_p95_ms", "latency/results/results_a100_1.json", "stage_latency_ms/total/p95", 2, "sec5.1"),
    ("lat.total_std_ms", "latency/results/results_a100_1.json", "stage_latency_ms/total/std", 2, "sec5.1"),
    ("lat.n_queries", "latency/results/results_a100_1.json", "reps", 0, "sec4.1"),
    ("lat.warmup_excluded", "latency/results/results_a100_1.json", "warmup_excluded", 0, "sec4.1"),
    ("cold.encoder_load_ms", "latency/results/results_a100_1.json", "cold_start_encoder/load_plus_first_infer_ms", 1, "sec5.2"),
    ("cold.encoder_warm_ms", "latency/results/results_a100_1.json", "cold_start_encoder/warm_infer_ms/mean", 2, "sec5.2"),
    ("cold.encoder_ratio", "latency/results/results_a100_1.json", "cold_start_encoder/cold_warm_ratio", 1, "sec5.2 headline ratio"),
    ("cold.encoder_warm_p50_ms", "latency/results/results_a100_1.json", "cold_start_encoder/warm_infer_ms/p50", 2, "sec5.2 ratio denominator"),
    # ---- secondary: re-verification runs (same host, same script and arguments, 2026-09-28) ----
    # rerun_a: first run after a reboot, OS page cache empty; rerun_b: immediately after, page cache warm.
    ("rerun_a.cold.encoder_load_ms", "latency/results_rerun_20260928/results_a100_1.json", "cold_start_encoder/load_plus_first_infer_ms", 1, "sec5.2 page-cache dependence"),
    ("rerun_a.cold.encoder_warm_p50_ms", "latency/results_rerun_20260928/results_a100_1.json", "cold_start_encoder/warm_infer_ms/p50", 2, "sec5.2"),
    ("rerun_b.cold.encoder_load_ms", "latency/results_rerun_20260928_b/results_a100_1.json", "cold_start_encoder/load_plus_first_infer_ms", 1, "sec5.2 page-cache dependence"),
    ("rerun_b.cold.encoder_warm_p50_ms", "latency/results_rerun_20260928_b/results_a100_1.json", "cold_start_encoder/warm_infer_ms/p50", 2, "sec5.2"),
    ("rerun_a.lat.total_ms", "latency/results_rerun_20260928/results_a100_1.json", "stage_latency_ms/total/mean", 2, "sec5.1 re-run consistency"),
    ("rerun_b.lat.total_ms", "latency/results_rerun_20260928_b/results_a100_1.json", "stage_latency_ms/total/mean", 2, "sec5.1 re-run consistency"),
    # ---- routing under constraint: shipped vs corrected (Table 2, all cells) ----
    *[(f"dmar.{fam}.{tag}.{field}", "ilp_fix/results/ilp_fix_summary.json",
       f"{prof}|{fam}/{key}/mean", nd, "Table 2")
      for prof, tag in (("a100_1", "a100"), ("a6000_1", "a6000"), ("rtx4090_simulated", "4090"))
      for fam in ("distinct", "repeated")
      for key, field, nd in (("shipped_q", "shipped_qbar", 3), ("corrected_q", "corrected_qbar", 3),
                             ("shipped_v", "shipped_violation_rate", 3),
                             ("corrected_v", "corrected_violation_rate", 3),
                             ("shipped_fallback", "fallback_rate", 3))],
    *[(f"dmar.alpha.{tag}", "ilp_fix/results/ilp_fix_summary.json", f"{prof}|distinct/alpha/mean", 3,
       "Table 1 configuration")
      for prof, tag in (("a100_1", "a100"), ("a6000_1", "a6000"), ("rtx4090_simulated", "4090"))],
    *[(f"dmar.vmax.{tag}", "ilp_fix/results/ilp_fix_summary.json", f"{prof}|distinct/v_max/mean", 0,
       "Table 1 configuration")
      for prof, tag in (("a100_1", "a100"), ("a6000_1", "a6000"), ("rtx4090_simulated", "4090"))],
    *[(f"dmar.solve_ms_k{k}", "latency/results/dmar_remeasure.json", f"by_k/{k-1}/mean", 2,
       "sec5.3 solver cost") for k in (1, 5)],
    ("dmar.solve_reps", "latency/results/dmar_remeasure.json", "reps_per_k", 0, "sec5.3"),
    ("dmar.solve_ms_k3", "latency/results/dmar_remeasure.json", "by_k/2/mean", 2, "sec5.3 solver cost"),
    # ---- audit: conflict screening, external corpora (sec 6.1) ----
    *[(f"ext.{d}.{field}", "external_eval/results/results.json",
       f"results/{d}/primary_contradiction_vs_entailment/at_production_tau/{key}", 3, "sec6.1 Table 3")
      for d in ("snli", "mnli", "vitaminc")
      for key, field in (("f1", "f1"), ("precision", "precision"), ("recall", "recall"))],
    *[(f"ext.{d}.neutral_f1", "external_eval/results/results.json",
       f"results/{d}/with_neutral_as_clean/at_production_tau/f1", 3, "sec6.1 neutral-as-clean")
      for d in ("snli", "mnli", "vitaminc")],
    ("ext.n_per_class", "external_eval/results/results.json", "n_per_class", 0, "sec6.1"),
    ("ext.path_agreement", "external_eval/results/results.json",
     "results/snli/cmae_path_verification/agreement_rate", 3, "sec6.1 path check"),
    *[(f"ext.{d}.best_{f}", "external_eval/results/results.json",
       f"results/{d}/primary_contradiction_vs_entailment/best_threshold/{f}", 3,
       "sec6.1 threshold transfer")
      for d in ("snli", "mnli", "vitaminc") for f in ("tau", "f1")],
    *[(f"cmae.type.{t}", "synconflict_v2/results/results.json",
       f"results/v2/by_type/{t}/detected_rate", 3, "sec6.1 by conflict type")
      for t in ("numerical", "temporal", "terminological", "paraphrastic", "domain_mismatch",
                "complementary", "paraphrase", "unrelated_consistent")],
    *[(f"cmae.register.{r}_{f}", "synconflict_v2/results/results.json",
       f"results/v2/by_register/{r}/{f}", 3, "sec6.1 register confound")
      for r in ("same", "cross") for f in ("recall", "precision", "n")],
    ("cmae.v1_auc", "synconflict_v2/results/results.json", "results/v1/roc_auc", 3, "sec6.1 / L2"),
    ("cmae.v1_n", "synconflict_v2/results/results.json", "results/v1/n", 0, "sec6.1 / L2"),
    ("cmae.v2_overlap", "synconflict_v2/results/results.json", "results/v2/separability_overlap", 3, "sec6.1"),
    ("cmae.v1_overlap", "synconflict_v2/results/results.json", "results/v1/separability_overlap", 3, "sec6.1 / L2"),
    # ---- audit: memory, by target modality and coverage (sec 6.2) ----
    *[(f"hmmr.{m}.{tag}", "hmmr_v2/results/analysis.json", f"by_target_modality/{m}/{cfg}", 3, "sec6.2")
      for m in ("text", "document", "image")
      for cfg, tag in (("flat_rag", "flat_mrr"), ("hmmr@default", "full_mrr"))],
    ("hmmr.coverage_missed", "hmmr_v2/results/analysis.json",
     "coverage/hmmr@default/target_not_returned", 4, "sec6.2 coverage"),
    ("hmmr.flat_returned", "hmmr_v2/results/analysis.json", "coverage/flat_rag/mean_n_returned", 0, "sec6.2"),
    ("hmmr.hmmr_returned", "hmmr_v2/results/analysis.json", "coverage/hmmr@default/mean_n_returned", 0, "sec6.2"),
    # ---- deployed configuration (Table 1); read from the settings block of a run artifact ----
    *[(f"cfg.{k}", "synconflict_v2/results/results.json", f"settings/{k}", 3, "Table 1")
      for k in ("cmae_tau_conflict", "cmae_gamma", "cmae_tau_t", "cmae_theta_dom",
                "hmmr_lambda1", "hmmr_lambda2", "hmmr_lambda3", "hmmr_delta",
                "hmmr_omega_s", "hmmr_omega_e", "hmmr_omega_l", "hmmr_theta_imp",
                "mars_theta_cite", "mars_theta_sent", "dmar_alpha_ref", "dmar_beta",
                "dmar_vram_reserve_gb", "dmar_qmin_default")],
    # ---- routing registry constants (Table 1, sec3.3, sec5.2); not part of the settings block ----
    *[(f"reg.vram.{tag}", "config_snapshot/results/registry_snapshot.json",
       f"model_registry/{mid}/vram_gb", 1, "sec3.3 / sec5.2 model footprints")
      for tag, mid in (("vlm", "vlm_7b"), ("stt", "stt_large"), ("doc", "doc_processor"),
                       ("llm", "llm_7b"))],
    ("reg.vram.specialist_sum", "config_snapshot/results/registry_snapshot.json",
     "specialist_vram_gb_sum", 1, "sec3.3 / sec5.2 pooled footprint"),
    *[(f"reg.qmin.{m}", "config_snapshot/results/registry_snapshot.json", f"q_min/{m}", 2,
       "Table 1 quality floor") for m in ("text", "image", "audio", "video", "document")],
    ("reg.qmin.min", "config_snapshot/results/registry_snapshot.json", "q_min_range/0", 2, "Table 1"),
    ("reg.qmin.max", "config_snapshot/results/registry_snapshot.json", "q_min_range/1", 2, "Table 1"),
    # ---- benchmark debiasing audit (sec 7 / L2) ----
    ("audit.modality_chi2_p", "hmmr_v2/results/dataset_audit.json",
     "run_0.json/A_contingency/p", 3, "sec7 L2"),
    ("audit.modalmatch_target", "hmmr_v2/results/dataset_audit.json",
     "run_0.json/B_mean_modalmatch_target", 4, "sec7 L2"),
    ("audit.modalmatch_distractor", "hmmr_v2/results/dataset_audit.json",
     "run_0.json/B_mean_modalmatch_distractors", 4, "sec7 L2"),
    # ---- unfired branch: trust-dominant strategy (sec 7 / L1) ----
    ("dom.theta_dom", "sensitivity/results/theta_dom_reachability.json", "theta_dom", 2, "sec7 L1"),
    ("dom.gap_required_k3", "sensitivity/results/theta_dom_reachability.json",
     "theory_min_trust_gap/3/min_trust_gap_required", 3, "sec7 L1"),
    ("dom.max_trust_k3", "sensitivity/results/theta_dom_reachability.json",
     "empirical_max_trust/3/max_trust_max", 3, "sec7 L1"),
    ("dom.pct_exceeding_k3", "sensitivity/results/theta_dom_reachability.json",
     "empirical_max_trust/3/pct_exceeding_theta_dom", 3, "sec7 L1"),
    ("dom.max_trust_k2", "sensitivity/results/theta_dom_reachability.json",
     "empirical_max_trust/2/max_trust_max", 3, "sec7 L1"),
    # ---- audit: conflict screening ----
    ("cmae.v1_f1", "synconflict_v2/results/results.json", "results/v1/at_production_tau/f1", 3, "sec6.1"),
    ("cmae.v2_f1", "synconflict_v2/results/results.json", "results/v2/at_production_tau/f1", 3, "sec6.1 headline"),
    ("cmae.v2_auc", "synconflict_v2/results/results.json", "results/v2/roc_auc", 3, "sec6.1"),
    ("cmae.v2_n", "synconflict_v2/results/results.json", "results/v2/n", 0, "sec6.1"),
    # ---- audit: memory ----
    ("hmmr.flat_mrr", "hmmr_v2/results/analysis.json", "configs/flat_rag/rr/mean", 3, "sec6.2"),
    ("hmmr.nodelta_mrr", "hmmr_v2/results/analysis.json", "configs/hmmr_nodelta@default/rr/mean", 3, "sec6.2"),
    ("hmmr.full_mrr", "hmmr_v2/results/analysis.json", "configs/hmmr@default/rr/mean", 3, "sec6.2"),
    ("hmmr.n_probes", "hmmr_v2/results/analysis.json", "configs/flat_rag/rr/n", 0, "sec6.2"),
    # ---- target-level memory comparison (unit = target turn, seeds averaged) ----
    ("hmmr.t.n_targets", "hmmr_v2/results/analysis.json", "target_level/n_targets", 0, "sec5.5"),
    ("hmmr.t.n_targets_all_seeds", "hmmr_v2/results/analysis.json", "target_level/n_targets_all_seeds", 0, "sec5.5"),
    ("hmmr.t.n_probes", "hmmr_v2/results/analysis.json", "target_level/n_probes", 0, "sec5.5"),
    ("hmmr.t.mrr_flat", "hmmr_v2/results/analysis.json", "target_level/mrr/flat_rag", 3, "sec5.5"),
    ("hmmr.t.mrr_nodelta", "hmmr_v2/results/analysis.json", "target_level/mrr/hmmr_nodelta@default", 3, "sec5.5"),
    ("hmmr.t.mrr_full", "hmmr_v2/results/analysis.json", "target_level/mrr/hmmr@default", 3, "sec5.5"),
    ("hmmr.t.icc_flat", "hmmr_v2/results/analysis.json", "target_level/icc1_flat_rr", 2, "sec5.5"),
    ("hmmr.t.icc_full_minus_flat", "hmmr_v2/results/analysis.json", "target_level/icc1_full_minus_flat_rr", 2, "sec5.5"),
    ("hmmr.t.full_vs_flat", "hmmr_v2/results/analysis.json", "target_level/contrasts/hmmr@default - flat_rag/mean_delta", 4, "sec5.5"),
    ("hmmr.t.full_vs_flat_ci_lo", "hmmr_v2/results/analysis.json", "target_level/contrasts/hmmr@default - flat_rag/bootstrap_ci95/0", 4, "sec5.5"),
    ("hmmr.t.full_vs_flat_ci_hi", "hmmr_v2/results/analysis.json", "target_level/contrasts/hmmr@default - flat_rag/bootstrap_ci95/1", 4, "sec5.5"),
    ("hmmr.t.full_vs_flat_p", "hmmr_v2/results/analysis.json", "target_level/contrasts/hmmr@default - flat_rag/permutation_p", 3, "sec5.5"),
    ("hmmr.t.full_vs_flat_nonzero", "hmmr_v2/results/analysis.json", "target_level/contrasts/hmmr@default - flat_rag/n_targets_nonzero", 0, "sec5.5"),
    ("hmmr.t.nodelta_vs_flat", "hmmr_v2/results/analysis.json", "target_level/contrasts/hmmr_nodelta@default - flat_rag/mean_delta", 4, "sec5.5"),
    ("hmmr.t.nodelta_vs_flat_p", "hmmr_v2/results/analysis.json", "target_level/contrasts/hmmr_nodelta@default - flat_rag/permutation_p", 3, "sec5.5"),
    ("hmmr.t.bonus", "hmmr_v2/results/analysis.json", "target_level/contrasts/hmmr@default - hmmr_nodelta@default/mean_delta", 4, "sec5.5"),
    ("hmmr.t.bonus_nonzero", "hmmr_v2/results/analysis.json", "target_level/contrasts/hmmr@default - hmmr_nodelta@default/n_targets_nonzero", 0, "sec5.5"),
    ("hmmr.delta_vs_flat", "hmmr_v2/results/analysis.json", "paired/hmmr@default - flat_rag/rr/delta", 4, "sec6.2"),
    ("hmmr.delta_vs_flat_p", "hmmr_v2/results/analysis.json", "paired/hmmr@default - flat_rag/rr/wilcoxon_p", 3, "sec6.2"),
    ("hmmr.delta_vs_nodelta", "hmmr_v2/results/analysis.json",
     "paired/hmmr@default - hmmr_nodelta@default/rr/delta", 4, "sec6.2 modality bonus"),
    ("hmmr.delta_vs_nodelta_p", "hmmr_v2/results/analysis.json",
     "paired/hmmr@default - hmmr_nodelta@default/rr/wilcoxon_p", 3, "sec6.2 modality bonus"),
    ("hmmr.nodelta_vs_flat", "hmmr_v2/results/analysis.json",
     "paired/hmmr_nodelta@default - flat_rag/rr/delta", 4, "sec6.2"),
    # ---- audit: attribution ----
    ("mars.shipped_theta_citations", "mars_attribution/results/results.json", "summary/theta_0.70/queries_with_any_citation", 0, "sec6.3 zero citations"),
    ("mars.n_queries", "mars_attribution/results/results.json", "summary/theta_0.70/n_queries", 0, "sec6.3"),
    ("mars.corrected_precision_rel", "mars_attribution/results/results.json", "summary/theta_0.10/precision_relevance/mean", 3, "sec6.3"),
    ("mars.random_precision", "mars_attribution/results/results.json", "summary/theta_0.10/random_precision_baseline/mean", 3, "sec6.3"),
    ("mars.control_within_precision_rel", "mars_attribution/results/results.json", "summary/control_shuffle_within/precision_relevance/mean", 3, "sec6.3 control"),
    ("mars.corrected_recall_unit", "mars_attribution/results/results.json", "summary/theta_0.10/recall_unit/mean", 3, "sec6.3"),
    *[(f"mars.cited_at_{t}", "mars_attribution/results/results.json",
       f"summary/theta_{t}/queries_with_any_citation", 0, "sec6.3 threshold sweep")
      for t in ("0.15", "0.20", "0.25", "0.30")],
    ("mars.recall_at_015", "mars_attribution/results/results.json",
     "summary/theta_0.15/recall_unit/mean", 3, "sec6.3 threshold sweep"),
    ("mars.recall_at_020", "mars_attribution/results/results.json",
     "summary/theta_0.20/recall_unit/mean", 3, "sec6.3 threshold sweep"),
    ("mars.grade_high_at_shipped", "mars_attribution/results/results.json", "summary/theta_0.70/grade_distribution/High", 0, "sec6.3 / L1"),
    # ---- audit: ablation ----
    ("abl.full_faithfulness", "ablation_v2/results/analysis.json", "aggregated_from_results_json/full/mean", 4, "sec6.4 / Table 3"),
    ("abl.trustflow_delta", "ablation_v2/results/analysis.json", "paired_vs_full/no_trust_flow/mean_delta", 4, "sec6.4 headline"),
    ("abl.trustflow_p", "ablation_v2/results/analysis.json", "paired_vs_full/no_trust_flow/wilcoxon_p", 3, "sec6.4"),
    ("abl.trustflow_ci_lo", "ablation_v2/results/analysis.json", "paired_vs_full/no_trust_flow/ci95/0", 4, "sec6.4"),
    ("abl.trustflow_ci_hi", "ablation_v2/results/analysis.json", "paired_vs_full/no_trust_flow/ci95/1", 4, "sec6.4"),
    ("abl.nohmmr_delta", "ablation_v2/results/analysis.json", "paired_vs_full/no_hmmr/mean_delta", 4, "sec6.4"),
    ("abl.nohmmr_p", "ablation_v2/results/analysis.json", "paired_vs_full/no_hmmr/wilcoxon_p", 3, "sec6.4"),
    ("abl.nohmmr_ci_lo", "ablation_v2/results/analysis.json",
     "paired_vs_full/no_hmmr/ci95/0", 4, "sec6.4"),
    ("abl.nohmmr_ci_hi", "ablation_v2/results/analysis.json",
     "paired_vs_full/no_hmmr/ci95/1", 4, "sec6.4"),
    ("abl.nomars_delta", "ablation_v2/results/analysis.json",
     "paired_vs_full/no_mars/mean_delta", 4, "sec6.4 structural zero"),
    ("abl.nomars_identical", "ablation_v2/results/analysis.json",
     "paired_vs_full/no_mars/n_identical_to_full", 0, "sec6.4 structural zero"),
    ("abl.trustflow_identical", "ablation_v2/results/analysis.json",
     "paired_vs_full/no_trust_flow/n_identical_to_full", 0, "sec6.4"),
    ("abl.notrust_mean", "ablation_v2/results/analysis.json",
     "aggregated_from_results_json/no_trust_flow/mean", 4, "sec6.4"),
    ("abl.full_std", "ablation_v2/results/analysis.json",
     "aggregated_from_results_json/full/std", 4, "sec6.4"),
    ("abl.full_run_min", "ablation_v2/results/analysis.json",
     "aggregated_from_results_json/full/per_run_means/0", 4, "sec6.4 run-to-run spread"),
    ("abl.full_run_max", "ablation_v2/results/analysis.json",
     "aggregated_from_results_json/full/per_run_means/1", 4, "sec6.4 run-to-run spread"),
    ("abl.n_pairs", "ablation_v2/results/analysis.json", "n_pairs", 0, "sec6.4"),
    # ---- deployed strategy selection with an empty conflict set (supplement, Algorithm S2 note) ----
    ("abl.strat.n_instances", "ablation_v2/results/analysis.json", "strategy_audit/n_instances_per_config", 0, "supp S2 note"),
    ("abl.strat.n_configs", "ablation_v2/results/analysis.json", "strategy_audit/n_configs", 0, "supp S2 note"),
    ("abl.strat.n_empty_P_explicit", "ablation_v2/results/analysis.json", "strategy_audit/n_empty_P_explicit_conflict", 0, "supp S2 note"),
    ("abl.strat.share_empty_P_explicit", "ablation_v2/results/analysis.json", "strategy_audit/share_empty_P_explicit_conflict", 2, "supp S2 note"),
    ("abl.strat.n_pairs_differ", "ablation_v2/results/analysis.json", "strategy_audit/n_pairs_strategy_or_P_differs_across_configs", 0, "supp S2 note"),
    # ---- query-level analysis (unit = query, n=30; 3 orderings averaged) ----
    ("abl.q.n_queries", "ablation_v2/results/analysis.json", "query_level/n_queries", 0, "sec4.3 / sec5.7 unit of analysis"),
    ("abl.q.n_orderings", "ablation_v2/results/analysis.json", "query_level/n_orderings", 0, "sec4.4"),
    ("abl.q.n_bootstrap", "ablation_v2/results/analysis.json", "query_level/n_bootstrap", 0, "sec4.3"),
    ("abl.q.n_permutation", "ablation_v2/results/analysis.json", "query_level/n_permutation", 0, "sec4.3"),
    ("abl.q.full_icc", "ablation_v2/results/analysis.json", "query_level/full_icc1_across_orderings", 2, "sec4.4 / sec5.7 ordering variance"),
    ("abl.q.full_sd_median", "ablation_v2/results/analysis.json", "query_level/full_within_query_sd_median", 3, "sec5.7"),
    ("abl.q.trustflow_delta", "ablation_v2/results/analysis.json", "query_level/contrasts/no_trust_flow/mean_delta", 4, "sec5.7 / Table 3"),
    ("abl.q.trustflow_ci_lo", "ablation_v2/results/analysis.json", "query_level/contrasts/no_trust_flow/bootstrap_ci95/0", 4, "sec5.7 / Table 3"),
    ("abl.q.trustflow_ci_hi", "ablation_v2/results/analysis.json", "query_level/contrasts/no_trust_flow/bootstrap_ci95/1", 4, "sec5.7 / Table 3"),
    ("abl.q.trustflow_perm_p", "ablation_v2/results/analysis.json", "query_level/contrasts/no_trust_flow/permutation_p", 3, "sec5.7 / Table 3"),
    ("abl.q.trustflow_scores_identical", "ablation_v2/results/analysis.json", "query_level/contrasts/no_trust_flow/n_scores_identical_instances", 0, "sec5.7"),
    ("abl.q.trustflow_answers_identical", "ablation_v2/results/analysis.json", "query_level/contrasts/no_trust_flow/n_answers_byte_identical_instances", 0, "sec5.7"),
    ("abl.q.trustflow_worse", "ablation_v2/results/analysis.json", "query_level/contrasts/no_trust_flow/n_queries_worse", 0, "sec5.7"),
    ("abl.q.trustflow_better", "ablation_v2/results/analysis.json", "query_level/contrasts/no_trust_flow/n_queries_better", 0, "sec5.7"),
    ("abl.q.nohmmr_delta", "ablation_v2/results/analysis.json", "query_level/contrasts/no_hmmr/mean_delta", 4, "sec5.7"),
    ("abl.q.nohmmr_ci_lo", "ablation_v2/results/analysis.json", "query_level/contrasts/no_hmmr/bootstrap_ci95/0", 4, "sec5.7"),
    ("abl.q.nohmmr_ci_hi", "ablation_v2/results/analysis.json", "query_level/contrasts/no_hmmr/bootstrap_ci95/1", 4, "sec5.7"),
    ("abl.q.nohmmr_perm_p", "ablation_v2/results/analysis.json", "query_level/contrasts/no_hmmr/permutation_p", 3, "sec5.7"),
    ("mars.q.n_queries", "mars_attribution/results/query_level.json", "n_queries", 0, "sec5.6"),
    ("mars.q.n_orderings", "mars_attribution/results/query_level.json", "n_orderings", 0, "sec5.6"),
    ("mars.q.cited_all_010", "mars_attribution/results/query_level.json", "cited_by_theta/0.10/queries_cited_all_orderings", 0, "sec5.6 sweep"),
    ("mars.q.cited_all_015", "mars_attribution/results/query_level.json", "cited_by_theta/0.15/queries_cited_all_orderings", 0, "sec5.6 sweep"),
    ("mars.q.cited_any_015", "mars_attribution/results/query_level.json", "cited_by_theta/0.15/queries_cited_any_ordering", 0, "sec5.6 sweep"),
    ("mars.q.cited_all_020", "mars_attribution/results/query_level.json", "cited_by_theta/0.20/queries_cited_all_orderings", 0, "sec5.6 sweep"),
    ("mars.q.cited_any_020", "mars_attribution/results/query_level.json", "cited_by_theta/0.20/queries_cited_any_ordering", 0, "sec5.6 sweep"),
    ("mars.q.cited_any_070", "mars_attribution/results/query_level.json", "cited_by_theta/0.70/queries_cited_any_ordering", 0, "sec5.6 shipped"),
    ("mars.q.high_all", "mars_attribution/results/query_level.json", "shipped_grade_high_all_orderings", 0, "sec5.6 shipped"),
    ("mars.q.precision", "mars_attribution/results/query_level.json", "at_theta_0.10/precision_relevance/mean", 3, "sec5.6"),
    ("mars.q.precision_ci_lo", "mars_attribution/results/query_level.json", "at_theta_0.10/precision_relevance/bootstrap_ci95/0", 3, "sec5.6"),
    ("mars.q.precision_ci_hi", "mars_attribution/results/query_level.json", "at_theta_0.10/precision_relevance/bootstrap_ci95/1", 3, "sec5.6"),
    ("mars.q.recall", "mars_attribution/results/query_level.json", "at_theta_0.10/recall_unit/mean", 3, "sec5.6"),
    ("mars.q.recall_ci_lo", "mars_attribution/results/query_level.json", "at_theta_0.10/recall_unit/bootstrap_ci95/0", 3, "sec5.6"),
    ("mars.q.recall_ci_hi", "mars_attribution/results/query_level.json", "at_theta_0.10/recall_unit/bootstrap_ci95/1", 3, "sec5.6"),
    ("mars.q.recall_015", "mars_attribution/results/query_level.json", "recall_unit_by_theta/0.15/mean", 3, "sec5.6 sweep"),
    ("mars.q.recall_020", "mars_attribution/results/query_level.json", "recall_unit_by_theta/0.20/mean", 3, "sec5.6 sweep"),
    ("mars.q.random", "mars_attribution/results/query_level.json", "at_theta_0.10/random_precision_baseline/mean", 3, "sec5.6"),
    ("mars.q.shuffle", "mars_attribution/results/query_level.json", "control_shuffle_within/precision_relevance/mean", 3, "sec5.6"),
    ("mars.q.shuffle_ci_lo", "mars_attribution/results/query_level.json", "control_shuffle_within/precision_relevance/bootstrap_ci95/0", 3, "sec5.6"),
    ("mars.q.shuffle_ci_hi", "mars_attribution/results/query_level.json", "control_shuffle_within/precision_relevance/bootstrap_ci95/1", 3, "sec5.6"),
    # ---- metric validation (sec 4.2) ----
    ("metric.copy", "ablation_v2/results/metric_validation.json", "conditions/copy/faithfulness_mean", 3, "sec4.2 control"),
    ("metric.unrelated", "ablation_v2/results/metric_validation.json", "conditions/unrelated/faithfulness_mean", 3, "sec4.2 control"),
    ("metric.negated", "ablation_v2/results/metric_validation.json",
     "conditions/negated/faithfulness_mean", 3, "sec4.2 control"),
    ("metric.n_instances", "ablation_v2/results/metric_validation.json", "n_instances", 0, "sec4.2"),
    ("metric.rejected_copy", "ablation_v2/results/metric_validation.json",
     "history/mDeBERTa-base + chunk premise/copy", 3, "sec4.2 rejected configuration"),
    ("metric.rejected_unrelated", "ablation_v2/results/metric_validation.json",
     "history/mDeBERTa-base + chunk premise/unrelated", 3, "sec4.2 rejected configuration"),
    ("metric.separation", "ablation_v2/results/metric_validation.json", "separation_copy_minus_unrelated", 3, "sec4.2"),
]


# Values that are not a single key in an artifact but a reduction over one. Each entry names the
# artifact it reads and the reducer that produces the number, so the value is still computed from
# a file on disk rather than typed in.
def _promotion(state_glob: str):
    """E-tier promotion rate and mean importance per modality set, over the memory states."""
    from collections import defaultdict
    tot, pro, imp = defaultdict(int), defaultdict(int), defaultdict(list)
    for f in sorted(ROOT.glob(state_glob)):
        for proj in json.loads(f.read_text(encoding="utf-8")):
            seen: dict = {}
            for r in proj["stored"]:
                if r.get("turn_key") is None:      # generated summaries, not promoted turns
                    continue
                k = (f.name, proj["project_id"], r["turn_key"])
                seen.setdefault(k, [tuple(r["modalities"]), False, r["importance"]])
                if r["tier"] == "E":
                    seen[k][1] = True
            for m, e, i in seen.values():
                tot[m] += 1
                pro[m] += int(e)
                imp[m].append(i)
    return {"+".join(m): {"n": tot[m], "rate": pro[m] / tot[m],
                          "mean_importance": sum(imp[m]) / len(imp[m])} for m in tot}


def _analytic(rel: str) -> tuple[float, float]:
    d = jload(rel)[0]
    s = sum(dig(d, f"stage_latency_ms/{k}/mean")
            for k in ("dmar_route", "cmae_encode", "hmmr_retrieve", "mars_score"))
    return s, s / dig(d, "stage_latency_ms/total/mean")


def _ratio_p50(rel: str) -> float:
    d = jload(rel)[0]
    return (dig(d, "cold_start_encoder/load_plus_first_infer_ms")
            / dig(d, "cold_start_encoder/warm_infer_ms/p50"))


DERIVED = [
    ("hmmr.promote.text_rate", "hmmr_v2/results/memory_state_seed0.json",
     lambda: _promotion("hmmr_v2/results/memory_state_seed*.json")["text"]["rate"], 3, "sec6.2 / L1"),
    ("hmmr.promote.text_doc_rate", "hmmr_v2/results/memory_state_seed0.json",
     lambda: _promotion("hmmr_v2/results/memory_state_seed*.json")["text+document"]["rate"], 3, "sec6.2 / L1"),
    ("hmmr.promote.text_image_rate", "hmmr_v2/results/memory_state_seed0.json",
     lambda: _promotion("hmmr_v2/results/memory_state_seed*.json")["text+image"]["rate"], 3, "sec6.2 / L1"),
    ("hmmr.promote.text_imp", "hmmr_v2/results/memory_state_seed0.json",
     lambda: _promotion("hmmr_v2/results/memory_state_seed*.json")["text"]["mean_importance"], 3, "sec6.2 / L1"),
    ("hmmr.promote.text_doc_imp", "hmmr_v2/results/memory_state_seed0.json",
     lambda: _promotion("hmmr_v2/results/memory_state_seed*.json")["text+document"]["mean_importance"], 3, "sec6.2 / L1"),
    ("hmmr.promote.text_image_imp", "hmmr_v2/results/memory_state_seed0.json",
     lambda: _promotion("hmmr_v2/results/memory_state_seed*.json")["text+image"]["mean_importance"], 3, "sec6.2 / L1"),
    ("lat.analytic_sum_ms", "latency/results/results_a100_1.json",
     lambda: sum(dig(jload("latency/results/results_a100_1.json")[0],
                     f"stage_latency_ms/{k}/mean")
                 for k in ("dmar_route", "cmae_encode", "hmmr_retrieve", "mars_score")), 2,
     "sec5.1 four analytical stages"),
    ("lat.analytic_share", "latency/results/results_a100_1.json",
     lambda: sum(dig(jload("latency/results/results_a100_1.json")[0],
                     f"stage_latency_ms/{k}/mean")
                 for k in ("dmar_route", "cmae_encode", "hmmr_retrieve", "mars_score"))
     / dig(jload("latency/results/results_a100_1.json")[0], "stage_latency_ms/total/mean"), 4,
     "sec5.1 share of end-to-end"),
    # generation time against each analytical stage (sec5.1): range over the four stage means
    ("lat.gen_over_slowest_stage", "latency/results/results_a100_1.json",
     lambda: dig(jload("latency/results/results_a100_1.json")[0], "stage_latency_ms/llm_generate/mean")
     / max(dig(jload("latency/results/results_a100_1.json")[0], f"stage_latency_ms/{k}/mean")
           for k in ("dmar_route", "cmae_encode", "hmmr_retrieve", "mars_score")), 0,
     "sec5.1 generation vs analytical stages"),
    ("lat.gen_over_fastest_stage", "latency/results/results_a100_1.json",
     lambda: dig(jload("latency/results/results_a100_1.json")[0], "stage_latency_ms/llm_generate/mean")
     / min(dig(jload("latency/results/results_a100_1.json")[0], f"stage_latency_ms/{k}/mean")
           for k in ("dmar_route", "cmae_encode", "hmmr_retrieve", "mars_score")), 0,
     "sec5.1 generation vs analytical stages"),
    # cold/warm ratio against the warm median: one slow warm call can move the mean several-fold
    ("cold.encoder_ratio_p50", "latency/results/results_a100_1.json",
     lambda: _ratio_p50("latency/results/results_a100_1.json"), 1, "sec5.2 / L3 original conditions"),
    ("rerun_a.cold.encoder_ratio_p50", "latency/results_rerun_20260928/results_a100_1.json",
     lambda: _ratio_p50("latency/results_rerun_20260928/results_a100_1.json"), 1, "sec5.2 / L3 page cache empty"),
    ("rerun_b.cold.encoder_ratio_p50", "latency/results_rerun_20260928_b/results_a100_1.json",
     lambda: _ratio_p50("latency/results_rerun_20260928_b/results_a100_1.json"), 1, "sec5.2 / L3 page cache warm"),
    ("rerun_a.lat.analytic_sum_ms", "latency/results_rerun_20260928/results_a100_1.json",
     lambda: _analytic("latency/results_rerun_20260928/results_a100_1.json")[0], 2, "sec5.1 re-run consistency"),
    ("rerun_a.lat.analytic_share", "latency/results_rerun_20260928/results_a100_1.json",
     lambda: _analytic("latency/results_rerun_20260928/results_a100_1.json")[1], 4, "sec5.1 re-run consistency"),
    ("rerun_b.lat.analytic_sum_ms", "latency/results_rerun_20260928_b/results_a100_1.json",
     lambda: _analytic("latency/results_rerun_20260928_b/results_a100_1.json")[0], 2, "sec5.1 re-run consistency"),
    ("rerun_b.lat.analytic_share", "latency/results_rerun_20260928_b/results_a100_1.json",
     lambda: _analytic("latency/results_rerun_20260928_b/results_a100_1.json")[1], 4, "sec5.1 re-run consistency"),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, missing = [], []
    cache: dict[str, tuple] = {}
    for name, rel, path, nd, use in SPEC:
        p = ROOT / rel
        if not p.exists():
            missing.append((name, rel))
            continue
        if rel not in cache:
            cache[rel] = jload(rel)
        obj, _ = cache[rel]
        try:
            val = dig(obj, path)
        except (KeyError, IndexError, TypeError) as e:
            missing.append((name, f"{rel}:{path} ({type(e).__name__})"))
            continue
        rows.append({"name": name, "value": round(val, nd) if isinstance(val, float) else val,
                     "artifact": rel, "json_path": path, "artifact_sha256": sha256(p)[:16],
                     "used_for": use})
    for name, rel, fn, nd, use in DERIVED:
        p = ROOT / rel
        if not p.exists():
            missing.append((name, rel))
            continue
        val = fn()
        rows.append({"name": name, "value": round(val, nd) if isinstance(val, float) else val,
                     "artifact": rel, "json_path": "(derived; see DERIVED in this script)",
                     "artifact_sha256": sha256(p)[:16], "used_for": use})

    with (OUT / "paper_numbers.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "value", "artifact", "json_path",
                                          "artifact_sha256", "used_for"])
        w.writeheader()
        w.writerows(rows)
    manifest = {"export_commit": commit(), "n_values": len(rows),
                "artifacts": sorted({r["artifact"] for r in rows}),
                "missing": [{"name": n, "reason": r} for n, r in missing]}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"{len(rows)} values -> {OUT/'paper_numbers.csv'}   commit {manifest['export_commit'][:8]}")
    for n, r in missing:
        print(f"  MISSING {n}: {r}")


if __name__ == "__main__":
    main()
