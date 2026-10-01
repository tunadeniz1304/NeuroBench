"""nbfscil.cost_check must never run on test data, and aggregates per-seed costs as mean/std."""
import pytest

from nbfscil.cost_check import check_split, summarize


@pytest.mark.parametrize("split", ["base_test", "evaluation", "base_train"])
def test_cost_check_refuses_other_splits(split):
    with pytest.raises(ValueError):
        check_split(split)


def test_cost_check_allows_base_val():
    assert check_split("base_val") == "base_val"


def test_summarize_mean_std():
    keys = ("acc", "footprint", "extra_footprint_bytes", "activation_sparsity", "dense", "eff_macs", "eff_acs")
    runs = [{"seed": s, **{k: float(s) for k in keys}} for s in (0, 2)]
    out = summarize("c.yaml", runs)
    assert out["seeds"] == [0, 2] and out["eff_acs"] == [1.0, 1.0]
