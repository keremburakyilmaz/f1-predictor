"""Basic import smoke tests for Phase 1 modules."""


def test_phase_1_imports() -> None:
    """Phase 1 data modules import successfully."""
    from f1predictor.data import fastf1_loader, openf1_client, schedule

    assert schedule.get_completed_rounds
    assert fastf1_loader.fetch_race
    assert openf1_client.OpenF1Client
