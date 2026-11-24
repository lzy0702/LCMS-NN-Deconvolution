"""Recipe v4 (``--recipe v4``): v3 with lone multiply charged oligonucleotides.

The option behind it, ``SynthConfig.oligo_single_state_prob``, is off by default and then draws
nothing, so default and v3 frames, and their resume fingerprints, stay as they were (the digests
of ``tests/test_recipe_v3.py`` check the frames). Here: what v4 sets, and what the option does
where it applies.
"""

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lcmsdeconv.chem.adducts import AdductLibrary
from lcmsdeconv.synth.adduct_sampler import member_distributions
from lcmsdeconv.synth.config import SynthConfig, settings_repr
from lcmsdeconv.synth.spec import Compound

ROOT = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location("train_charge_net",
                                                  ROOT / "scripts" / "train_charge_net.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_v4_is_v3_with_single_state_oligos_on_orbitrap_only():
    script = _script()
    v3, v4 = script.mixture_v3(), script.mixture_v4()
    assert [w for w, _c in v4] == [w for w, _c in v3]
    for (_w3, c3), (_w4, c4) in zip(v3, v4):
        if c4.instrument_kind == "orbitrap":
            assert c4.oligo_single_state_prob == script.V4_SINGLE_STATE == 0.4
            assert c4.oligo_real_like_prob == 0.5  # the option acts in real-like frames only
        else:
            assert c4.oligo_single_state_prob == 0.0
        assert replace(c4, oligo_single_state_prob=0.0) == c3
    # v3's resume fingerprint text does not mention the new setting; v4's does
    assert "oligo_single_state_prob" not in settings_repr(v3)
    assert "oligo_single_state_prob=0.4" in settings_repr(v4)


def _members(mass: float, cls: str = "dna", impurities: int = 2):
    main = Compound(mass, cls, None, name="main", kind="main")
    out = [(main, 1.0)]
    for k in range(impurities):
        out.append((Compound(mass - 300.0 * (k + 1), cls, None, name=f"imp{k}", parent_id=0,
                             kind="impurity"), 0.05))
    return out


def _draw(cfg, members, real_like, seed):
    lib = AdductLibrary.from_mode(cfg.mode, cfg.polarity, max_per_type=cfg.adduct_max_per_type,
                                  max_total=cfg.adduct_max_total)
    rng = np.random.default_rng(seed)
    out = member_distributions(cfg, members, lib, {}, rng, real_like=real_like)
    return [cd for cd, _a in out], rng.random()


def test_single_state_draws_nothing_while_off_or_outside_real_like_frames():
    members = _members(4512.8)
    off = SynthConfig(instrument_kind="orbitrap")
    on = replace(off, oligo_single_state_prob=1.0)
    for seed in range(5):
        # off: exactly the draws of a config without the option
        assert _draw(off, members, True, seed) == _draw(SynthConfig(instrument_kind="orbitrap"),
                                                         members, True, seed)
        # on, but the frame is not real-like: no draw, the same distributions
        assert _draw(on, members, False, seed) == _draw(off, members, False, seed)


def test_single_state_oligos_show_one_charge_their_impurities_the_same():
    cfg = replace(SynthConfig(instrument_kind="orbitrap"), oligo_single_state_prob=1.0)
    seen = set()
    for seed in range(40):
        mass = 1000.0 + 275.0 * seed
        dists, _ = _draw(cfg, _members(mass), True, seed)
        main = dists[0]
        assert len(main) == 1 and list(main.values()) == [1.0]
        z = next(iter(main))
        apex = 0.0769 * mass ** 0.445
        assert max(1, round(0.8 * 0.9 * apex)) <= z <= round(1.25 * 1.1 * apex) + 1
        for imp in dists[1:]:
            assert imp == main  # impurities show the same single state
        seen.add(z)
    assert len(seen) >= 4  # z = 2 ... 6 over 1-12 kDa


def test_single_state_share_and_other_classes():
    cfg = replace(SynthConfig(instrument_kind="orbitrap"), oligo_single_state_prob=0.4)
    single = 0
    n = 400
    for seed in range(n):
        dists, _ = _draw(cfg, _members(5000.0, impurities=0), True, seed)
        single += len(dists[0]) == 1
    assert single / n == pytest.approx(0.4, abs=0.07)
    # other classes are left alone, drawn exactly as without the option
    off = SynthConfig(instrument_kind="orbitrap")
    for seed in range(20):
        prot = _members(15000.0, cls="peptide", impurities=0)
        assert _draw(cfg, prot, True, seed) == _draw(off, prot, True, seed)
