import pandas as pd

from lithology.dash.derived import salt_table, wells_table


def _logs(*rows):
    cols = ["log_id", "title", "well_group", "depth_ref", "top_ft", "bottom_ft", "kind"]
    return pd.DataFrame(rows, columns=cols)


def _beds(log_id, beds):
    return pd.DataFrame([{"log_id": log_id, "top_depth": t, "bottom_depth": b, "rock_family": f} for t, b, f in beds])


def test_salt_body_base_ignores_stringer_and_bridges_inclusions():
    logs = _logs(("a", "Well A", "A", "MD", 1000, 5000, "colour"))
    beds = _beds(
        "a",
        [
            (1000, 1500, "Shale"),
            (1500, 2500, "Salt"),
            (2500, 2530, "Shale"),  # 30 ft inclusion: same body
            (2530, 3000, "Salt"),
            (3000, 3200, "Shale"),
            (3200, 3204, "Salt"),  # stringer 200 ft below the base: mixed zone, not the base
            (3204, 3300, "Sand"),
        ],
    )
    s = salt_table(logs, beds).iloc[0]
    assert s.salt_top_ft == 1500 and s.top_seen
    assert s.salt_base_ft == 3000
    assert s.mixed_zone_base_ft == 3204
    assert s.sand_below_base_ft == 96
    assert s.main_log


def test_log_starting_inside_salt_does_not_claim_a_top():
    logs = _logs(("a", "Well A", "A", "TVD", 8700, 9000, "colour"))
    beds = _beds("a", [(8700, 8703, "Other"), (8750, 8900, "Salt"), (8900, 9000, "Shale")])
    assert not salt_table(logs, beds).iloc[0].top_seen


def test_one_main_log_per_well_and_well_aliases():
    logs = _logs(
        ("a", "G1", "G1", "TVD", 0, 0, "colour"),
        ("b", "MDN-1 ST00", "G1", "TVD", 0, 0, "colour"),
    )
    beds = pd.concat([_beds("a", [(100, 400, "Salt")]), _beds("b", [(100, 600, "Salt")])])
    s = salt_table(logs, beds)
    assert s.main_log.sum() == 1 and s[s.main_log].log_title.item() == "MDN-1 ST00"
    w = wells_table(logs).iloc[0]
    assert w.n_logs == 2 and "MDN-1 ST00" in w.other_names
