"""Independent host-side grammar oracle; no shared symbolic reconstruction helper."""

import numpy as np

import run_phase_g11_validation as v


def independent_parse(data):
    arrays = {k: np.asarray(x) for k, x in data.items()}
    for row, source in enumerate(arrays["source"]):
        slots = np.pad(source, (0, 64)).reshape(32, 4)
        extents = np.asarray([sum(int(x) != 0 for x in site) for site in slots], np.int32)
        q = int(arrays["prompt"][row, 1]) - 1
        assert q == int(arrays["q"][row])
        assert slots[q].tolist() == [3, 4, 4, 4]
        ends = [i for i, site in enumerate(slots) if int(site[0]) == 1 and i > q]
        assert ends, "no observable future END"
        peer = ends[0]
        assert peer == int(arrays["peer"][row]), "peer identity disagrees with raw observation"
        assert extents[q] == extents[peer] == 4 and 1 <= peer - q <= 5
        expected = source.copy()
        expected[4 * q + 1 : 4 * q + 4] = slots[peer, 1:]
        assert np.array_equal(expected, arrays["target"][row]), (
            "target disagrees with observed peer"
        )
        boundary = np.asarray(
            [int(site[0] == 1 and extents[i] > 0) for i, site in enumerate(slots)], np.int32
        )
        boundary[-1] = 1
        assert np.array_equal(boundary, arrays["boundary"][row])
        assert np.array_equal(extents, arrays["length"][row])
        nonzero = source[source != 0]
        assert np.all((nonzero >= 1) & (nonzero <= 32))
        assert np.array_equal(source != 0, arrays["source_mask"][row])
    return len(arrays["source"])


def main():
    checked = {
        name: independent_parse(v.make_examples(seed, count, id_offset=offset))
        for name, seed, count, offset in (
            ("train", 1140, 32, 0),
            ("validation", 1141, 16, 10000),
            ("protected_test_symbolic_only", 1142, 16, 20000),
        )
    }
    v.dump(
        v.OUT / "independent_data_audit.json",
        {
            "passed": True,
            "source_sha256": v.sha(__file__),
            "checked": checked,
            "total": sum(checked.values()),
            "shared_symbolic_helper_used": False,
            "neural_model_executed": False,
        },
    )


if __name__ == "__main__":
    main()
