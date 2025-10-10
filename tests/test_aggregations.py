import pandas as pd
import pytest

from src.analysis.aggregations import MetricsCalculator


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "call_id": [1, 2, 3],
            "outcome": ["connected", "missed", "connected"],
            "duration": [120, 60, 180],
            "revenue": [100.0, 0.0, 150.0],
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "agent_id": ["agent-1", "agent-2", "agent-1"],
            "campaign": ["alpha", "beta", "alpha"],
        }
    )


def test_calculate_basic_metrics(sample_dataframe: pd.DataFrame) -> None:
    calculator = MetricsCalculator()
    metrics = calculator.calculate_basic_metrics(sample_dataframe)

    assert metrics["total_calls"] == 3
    assert metrics["connected_calls"] == 2
    assert metrics["connection_rate"] == pytest.approx(66.6666, rel=1e-3)
    assert metrics["avg_duration"] == pytest.approx(120.0)
    assert metrics["total_duration"] == pytest.approx(360.0)
    assert metrics["median_duration"] == pytest.approx(120.0)
    assert metrics["total_revenue"] == pytest.approx(250.0)
    assert metrics["avg_revenue"] == pytest.approx(83.3333, rel=1e-3)
    assert metrics["revenue_per_minute"] == pytest.approx(41.6666, rel=1e-3)


def test_calculate_all_metrics_groups_expected_sections(sample_dataframe: pd.DataFrame) -> None:
    calculator = MetricsCalculator()
    grouped = calculator.calculate_all_metrics(sample_dataframe)

    overview = grouped["overview"]
    assert overview["total_calls"] == 3
    assert overview["connected_calls"] == 2
    assert overview["connection_rate"] == pytest.approx(66.6666, rel=1e-3)

    duration_metrics = grouped["duration_metrics"]
    assert duration_metrics["avg_duration"] == pytest.approx(120.0)
    assert duration_metrics["median_duration"] == pytest.approx(120.0)
    assert duration_metrics["p95"] == pytest.approx(174.0)

    revenue_metrics = grouped["revenue_metrics"]
    assert revenue_metrics["total_revenue"] == pytest.approx(250.0)
    assert revenue_metrics["revenue_per_minute"] == pytest.approx(41.6666, rel=1e-3)

    outcome_counts = grouped["outcome_distribution"]
    assert outcome_counts["connected_calls"] == 2
    assert outcome_counts["missed_calls"] == 1

    outcome_percent = grouped["outcome_distribution_percentage"]
    assert outcome_percent["connected_pct"] == pytest.approx(66.6666, rel=1e-3)
    assert outcome_percent["missed_pct"] == pytest.approx(33.3333, rel=1e-3)

    time_metrics = grouped["time_metrics"]
    assert time_metrics["active_days"] == 3
    assert time_metrics["calls_per_day"] == pytest.approx(1.0)
    assert time_metrics["first_call"] == "2024-01-01"
    assert time_metrics["last_call"] == "2024-01-03"

    entity_metrics = grouped["entity_metrics"]
    assert entity_metrics["unique_agents"] == 2
    assert entity_metrics["active_campaigns"] == 2


def test_calculate_agent_metrics_returns_sorted_dataframe(sample_dataframe: pd.DataFrame) -> None:
    calculator = MetricsCalculator()
    agent_metrics = calculator.calculate_agent_metrics(sample_dataframe)

    assert list(agent_metrics.index) == ["agent-1", "agent-2"]
    assert agent_metrics.loc["agent-1", "total_calls"] == 2
    assert agent_metrics.loc["agent-2", "total_calls"] == 1
    assert agent_metrics.loc["agent-1", "duration_mean"] == pytest.approx(150.0)
    assert agent_metrics.loc["agent-1", "revenue_sum"] == pytest.approx(250.0)
    assert agent_metrics.loc["agent-1", "connection_rate"] == pytest.approx(100.0)
    assert agent_metrics.loc["agent-2", "connection_rate"] == pytest.approx(0.0)


def test_calculate_all_metrics_handles_empty_dataframe() -> None:
    calculator = MetricsCalculator()
    grouped = calculator.calculate_all_metrics(pd.DataFrame())

    assert grouped == {"overview": {"total_calls": 0, "connected_calls": 0, "connection_rate": 0.0}}
