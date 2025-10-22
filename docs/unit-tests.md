# Unit Tests

I keept unit tests in this project so I can trust the core analytics features before I push anything.

## Table of Contents
- [1. Test layout](#1-test-layout)
- [2. Metrics tests](#2-metrics-tests)
- [3. Text helper tests with coverage](#3-text-helper-tests-with-coverage)
- [4. Running everything](#4-running-everything)

## 1. Test layout

- Metrics maths live in `tests/test_aggregations.py` where I reuse one fake DataFrame fixture across the checks (`tests/test_aggregations.py:7-91`).
- Text clean-up and similarity helpers live in `tests/test_text_processing.py` (`tests/test_text_processing.py:6-42`).
- The Ant build always generates a JUnit XML report from the `test` target (`build.xml:39-45`).
- Jenkins publishes that XML by calling `junit 'test-reports/*.xml'`, so the pipeline fails on any red test (`Jenkinsfile:88-94`).

## 2. Metrics tests

`tests/test_aggregations.py:7-34`

```python
@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "call_id": [1, 2, 3],
            "outcome": ["connected", "missed", "connected"],
            "duration": [120, 60, 180],
            "revenue": [100.0, 0.0, 150.0],
        }
    )

def test_calculate_basic_metrics(sample_dataframe: pd.DataFrame) -> None:
    calculator = MetricsCalculator()
    metrics = calculator.calculate_basic_metrics(sample_dataframe)

    assert metrics["total_calls"] == 3
    assert metrics["connection_rate"] == pytest.approx(66.6666, rel=1e-3)
    assert metrics["revenue_per_minute"] == pytest.approx(41.6666, rel=1e-3)
```

- The fixture builds a tiny table with only the columns needed for the assertions.
- The test feeds that table into `MetricsCalculator` and checks totals, averages, and rate calculations.
- I use `pytest.approx` so floating-point rounding never causes flaky failures.

`tests/test_aggregations.py:87-91`

```python
def test_calculate_all_metrics_handles_empty_dataframe() -> None:
    calculator = MetricsCalculator()
    grouped = calculator.calculate_all_metrics(pd.DataFrame())

    assert grouped == {"overview": {"total_calls": 0, "connected_calls": 0, "connection_rate": 0.0}}
```

- This guards against regressions when the analytics job receives no rows.
- Returning a predictable dictionary makes the API easy to handle in dashboards.

## 3. Text helper tests with coverage

`tests/test_text_processing.py:6-42`

```python
def test_clean_text_removes_noise() -> None:
    text = " Hello, WORLD! Call 1234. "
    cleaned = tp.clean_text(
        text,
        remove_punctuation=True,
        lowercase=True,
        remove_numbers=True,
    )
    assert cleaned == "hello world call"

@pytest.mark.parametrize("method", ["jaccard", "cosine", "levenshtein"])
def test_calculate_similarity_identical_texts(method: str) -> None:
    text = "call analytics system"
    score = tp.calculate_similarity(text, text, method=method)
    assert score == pytest.approx(1.0)
```

- The first test proves that optional flags in `clean_text` really strip punctuation, digits, and uppercase.
- Parametrised similarity tests loop through three algorithms and make sure identical strings always hit a perfect score.
- `test_calculate_similarity_invalid_method` catches bad inputs by expecting a `ValueError`, so the API fails fast when misused.
- `test_mask_pii_replaces_sensitive_tokens` confirms masking for email, phone, and card data so logs stay safe.

## 4. Running everything

```bash
source .venv/bin/activate
pytest -q --maxfail=1
```

- Pytest writes `test-reports/junit.xml` automatically because I am passing the `--junitxml` flag in the Ant target (`build.xml:39-45`).
- The Ant helper stays available with `ant -noinput -buildfile build.xml test` with the same run profile Jenkins uses.
- Jenkins collects that XML in the `post` block (`Jenkinsfile:88-94`) and exposes the JUnit report in the UI.

Keeping these checks green ensures the build pipeline catches regressions early and keeps the release flow smooth.
