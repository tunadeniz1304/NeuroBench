"""Per-seed backbone selection in nbfscil.official_eval and the final system config."""
import os

import yaml

from nbfscil.official_eval import seed_config

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_seed_config_expands_seed_and_env(monkeypatch):
    monkeypatch.setenv("CKPT_DIR", "/ck")
    cfg = {"system": "proto", "checkpoint": "${CKPT_DIR}/b_s{seed}.pt", "learner": {"name": "x"}}
    assert seed_config(cfg, 2)["checkpoint"] == "/ck/b_s2.pt"
    assert cfg["checkpoint"] == "${CKPT_DIR}/b_s{seed}.pt"          # the shared config is not mutated


def test_seed_config_without_checkpoint_is_unchanged():
    cfg = {"system": "upstream_proto_snn"}
    assert seed_config(cfg, 1) is cfg


def test_final_config_matches_pseudo_learner(monkeypatch):
    monkeypatch.setenv("CKPT_DIR", "/ck")
    cfg = yaml.safe_load(open(os.path.join(REPO, "configs", "final", "clip_cl2n_8bit.yaml")))
    learner = yaml.safe_load(open(os.path.join(REPO, "configs", "learners", "cl2n_8bit.yaml")))
    assert cfg["system"] == "proto" and cfg["learner"] == learner
    assert seed_config(cfg, 0)["checkpoint"] == "/ck/cos_1024_amp_clip_s0.pt"
