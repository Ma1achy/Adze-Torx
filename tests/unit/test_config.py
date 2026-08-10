from adze_t.config import CarrierConfig, LoopConfig


def test_carrier_length_categories_include_zero():
    cfg = CarrierConfig(max_bytes_per_site=8, length_categories=9)
    assert cfg.length_categories == cfg.max_bytes_per_site + 1


def test_effective_core_applications():
    cfg = LoopConfig(physical_blocks=4, core_cycles=3)
    assert cfg.effective_core_applications == 12
