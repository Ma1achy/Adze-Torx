from adze_t.config import REFERENCE_SMALL_V0


def test_reference_small_v0():
    cfg = REFERENCE_SMALL_V0
    assert cfg.carrier.C == 32
    assert cfg.carrier.h_dim == 64
    assert cfg.carrier.L_max == 4
    assert cfg.packing.M_max == 32
    assert cfg.packing.K == 8
