"""LadderNet, training without singly charged ions, and pausing/resuming a training run."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _features(seed=0, n_peaks=3000):
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(1, 4, 32768)
    idx = torch.randint(0, 32768, (n_peaks,), generator=g)
    x[0, 0, idx] = torch.rand(n_peaks, generator=g) * 5.0
    x[0, 1] = torch.rand(32768, generator=g)
    return x


def test_untrained_ladder_net_is_the_comb():
    """With its learned part at zero, LadderNet's charge scores are the comb's evidence."""
    from lcmsdeconv.nn.features import comb_evidence
    from lcmsdeconv.nn.model import build_model

    model = build_model("small", z_max=100, in_ch=4, arch="ladder").eval()
    x = _features()
    with torch.no_grad():
        logits = model(x)["charge_logits"][0]
    ch0 = x[0, 0].numpy().astype(np.float64)
    ev = comb_evidence(ch0, 2e-5, model.z_ladder) / 25.0  # the comb's evidence, scaled
    level = x[0, 0]
    kth = torch.topk(level, model.active_top).values[-1]
    active = ((level > model.active_level) & (level >= kth)).numpy()
    got = logits[1:model.z_ladder + 1].numpy()[:, active]
    want = 8.0 * ev[:, active]
    assert np.abs(got - want).max() < 1e-4


def test_ladder_net_sparse_and_dense_paths_agree():
    from lcmsdeconv.nn.model import build_model

    torch.manual_seed(0)
    model = build_model("small", z_max=100, in_ch=4, arch="ladder")
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    model.eval()
    x = _features(1)
    with torch.no_grad():
        sparse = model(x)
        model.dense = True
        dense = model(x)
    for k in ("charge_logits", "apex_logit"):
        assert torch.allclose(sparse[k], dense[k], atol=1e-4), k


def test_ignored_charges_carry_no_loss_weight():
    from lcmsdeconv.nn.losses import ChargeLoss

    L = 64
    feats = torch.zeros(1, 4, L)
    feats[0, 0] = 3.0
    topk_z = torch.zeros(1, L, 3, dtype=torch.long)
    topk_w = torch.zeros(1, L, 3)
    topk_z[0, :32, 0], topk_w[0, :32, 0] = 1, 1.0  # first half singly charged
    topk_z[0, 32:, 0], topk_w[0, 32:, 0] = 5, 1.0
    batch = {"features": feats, "topk_z": topk_z, "topk_w": topk_w, "heat": torch.zeros(1, L)}
    logits = torch.zeros(1, 101, L, requires_grad=True)
    out = {"charge_logits": logits, "apex_logit": torch.full((1, 1, L), -4.6)}
    ChargeLoss(ignore_charges=(1,))(out, batch)["loss"].backward()
    assert float(logits.grad[0, :, :32].abs().max()) == 0.0  # z = 1 bins: no gradient at all
    assert float(logits.grad[0, :, 32:].abs().max()) > 0.0


def test_generator_can_drop_singly_charged_ions():
    from lcmsdeconv.synth.compounds import DEFAULT_CLASSES
    from lcmsdeconv.synth.config import SynthConfig
    from lcmsdeconv.synth.frames import generate_frame

    cfg = SynthConfig(classes=list(DEFAULT_CLASSES), mode="rplc", polarity=1,
                      background_ions_range=(30, 30), singly_charged="drop")
    for seed in range(3):
        fs = generate_frame(cfg, np.random.default_rng(seed))
        assert not (fs.topk_z == 1).any()


def test_training_pauses_and_resumes(tmp_path):
    """A run stopped after one step leaves a state that a resumed run completes."""
    from lcmsdeconv.nn.train import TrainConfig, train
    from lcmsdeconv.synth.compounds import ClassConfig
    from lcmsdeconv.synth.config import SynthConfig

    synth = SynthConfig(classes=[ClassConfig("peptide", (8000.0, 20000.0))], mode="rplc",
                        polarity=1)
    out = tmp_path / "net.pt"
    cfg = dict(arch="ladder", model_size="small", epochs=2, train_len=4, val_len=2,
               batch_size=2, num_workers=0, threads=1, out=str(out), log_every=100)
    train(TrainConfig(max_seconds=1e-6, **cfg), synth_config=synth)  # stops after one step
    resume = out.with_suffix(".resume.pt")
    state = torch.load(resume, weights_only=False)
    assert (state["epoch"], state["step"]) == (0, 1)
    _model, history = train(TrainConfig(resume=True, **cfg), synth_config=synth)
    assert [h["epoch"] for h in history] == [0, 1]
    assert out.exists()
    assert torch.load(resume, weights_only=False)["epoch"] == 2
