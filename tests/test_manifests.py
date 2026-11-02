from pathlib import Path

from ai_trader.evaluation.replay import load_manifest


def test_all_five_session_manifests_are_valid() -> None:
    directory = Path(__file__).parents[1] / "examples" / "replays"
    manifests = [load_manifest(path) for path in sorted(directory.glob("*.json"))]
    assert [manifest.session_date for manifest in manifests] == [
        "2026-10-08",
        "2026-10-16",
        "2026-10-22",
        "2026-10-27",
        "2026-10-29",
    ]
    for manifest in manifests:
        assert manifest.symbols
        assert manifest.expected.min_daily_r == -3.0
        assert manifest.expected.zero_counter_trend_entries
        assert manifest.expected.zero_cooldown_violations
        assert manifest.expected.no_entries_before_15m_cascade
