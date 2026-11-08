"""`lith run` flags: box source and the tier-2 guard."""

from __future__ import annotations

import yaml

from lithology import cli


def test_detect_boxes_and_tier_flags_reach_the_worker_config(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_docs_from", lambda inputs: [])  # stop right after the config is built
    rc = cli.main(["run", str(tmp_path), "--out", str(tmp_path / "o"), "--detect-boxes", "--tier", "1"])
    assert rc == 2  # no documents (by design here)
    forced = yaml.safe_load((tmp_path / "o" / "_config_run_overrides.yaml").read_text())
    assert forced["layout"]["source"] == "detect" and forced["ai"]["tier"] == 1


def test_run_refuses_to_start_when_tier2_is_configured_but_unavailable(tmp_path, monkeypatch):
    import lithology.ai.embed as embed

    monkeypatch.setattr(embed, "availability", lambda: "torch is not installed")
    reached = []
    monkeypatch.setattr(cli, "_docs_from", lambda inputs: reached.append(1) or [])
    assert cli.main(["run", str(tmp_path), "--out", str(tmp_path / "o")]) == 2
    assert not reached  # stopped before any work: no silent classical-only run
    assert cli.main(["run", str(tmp_path), "--out", str(tmp_path / "o1"), "--tier", "1"]) == 2
    assert reached  # classical-only on purpose passes the guard
