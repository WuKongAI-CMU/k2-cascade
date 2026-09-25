import math

from k2cascade.projector.confidence import analyse, auroc, brier, ece, spearman, wasserstein1


def test_metric_basics():
    assert auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-9
    assert abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-9
    assert ece([1.0, 1.0], [1, 1]) == 0.0 and brier([1.0, 0.0], [1, 0]) == 0.0
    assert wasserstein1([0, 0, 0], [1, 1, 1]) == 1.0


def _row(i, text_e, proj_e, pg_t, pg_p, f1_t=1.0, f1_p=1.0):
    return {"i": i, "text": {"entropy": text_e, "p_gold": pg_t, "f1": f1_t, "em": 1.0, "logp": -0.1, "jsd_text": 0.0, "p_counter": 0.05},
            "project": {"entropy": proj_e, "p_gold": pg_p, "f1": f1_p, "em": 1.0, "logp": -0.2, "jsd_text": 0.1, "p_counter": 0.1}}


def test_analyse_tracks_sender_and_drop_ratio():
    n = 40
    clean = {i: _row(i, 0.1 * i, 0.1 * i + 0.05, 0.9, 0.8) for i in range(n)}
    removed = {i: _row(i, 0.1 * i, 0.1 * i, 0.3, 0.5) for i in range(n)}
    se = {"clean": {i: {"i": i, "se": 0.1 * i} for i in range(n)}}
    out = analyse({"clean": clean, "removed": removed}, se)
    assert out["n_kept"] == n
    assert out["sender_tracking"]["clean"]["auroc_project"] > 0.95
    # text drops 0.6, project drops 0.3 -> ratio 0.5
    assert abs(out["drop_ratio"]["project"] - 0.5) < 1e-9
    assert "p_gold_minus_text_ci" in out["per_variant"]["clean"]["project"]
    assert 0 < out["per_variant"]["clean"]["project"]["memorisation_ratio"] < 1


def test_belief_structure_metric():
    def row(i, pc_text, pc_proj):
        r = _row(i, 1.0, 1.2, 0.8, 0.7)
        r["sender_cands"] = [["a", 0.6], ["b", 0.3], ["c", 0.1]]
        r["text"]["p_cands"] = pc_text; r["project"]["p_cands"] = pc_proj
        return r
    clean = {i: row(i, [0.5, 0.3, 0.1], [0.1, 0.3, 0.5]) for i in range(10)}
    out = analyse({"clean": clean}, None)
    b = out["belief"]["clean"]
    assert b["text"]["spearman_sender_freq"] > 0.99 and b["project"]["spearman_sender_freq"] < -0.99
    assert abs(b["text"]["runner_up_share"] - 0.3 / 0.9) < 1e-9
