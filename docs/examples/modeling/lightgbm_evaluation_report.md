# LightGBM Evaluation Report

Source: `docs/examples/modeling/synthetic_supervised_rows.csv`

Rows evaluated: 1464
Validation windows: 1

## Final test

Window: `final_test_2025-02`
Period: 2025-02-01T00:00:00+11:00 to 2025-03-01T00:00:00+11:00
Training rows: 1488

| Metric | Value |
| --- | ---: |
| Row count | 672 |
| MAE | 0.9500 |
| RMSE | 1.4500 |
| WAPE | 0.0550 |

## Model comparison

| Window | Model | Rows | MAE | RMSE | WAPE | Relative WAPE improvement |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| final_test_2025-02 | LightGBM | 672 | 0.9500 | 1.4500 | 0.0550 | 45.00% |
| final_test_2025-02 | Seasonal Naive | 672 | 1.8000 | 2.3000 | 0.1000 | n/a |
| validation_2025-01 | LightGBM | 744 | 1.0000 | 1.5000 | 0.0650 | 27.78% |
| validation_2025-01 | Seasonal Naive | 744 | 1.4000 | 1.9000 | 0.0900 | n/a |

## Validation windows

| Window | Period | Training rows | Rows | MAE | RMSE | WAPE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| validation_2025-01 | 2025-01-01T00:00:00+11:00 to 2025-02-01T00:00:00+11:00 | 744 | 744 | 1.0000 | 1.5000 | 0.0650 |

## Metric comparison charts

```mermaid
xychart-beta
    title "MAE by evaluation window"
    x-axis ["validation_2025-01", "final_test_2025-02"]
    y-axis "MAE" 0 --> 1.1000
    bar [1.0000, 0.9500]
```

```mermaid
xychart-beta
    title "RMSE by evaluation window"
    x-axis ["validation_2025-01", "final_test_2025-02"]
    y-axis "RMSE" 0 --> 1.6500
    bar [1.5000, 1.4500]
```

```mermaid
xychart-beta
    title "WAPE by evaluation window"
    x-axis ["validation_2025-01", "final_test_2025-02"]
    y-axis "WAPE" 0 --> 0.0715
    bar [0.0650, 0.0550]
```

## Final test by horizon

| Horizon | Rows | MAE | RMSE | WAPE |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 672 | 0.9500 | 1.4500 | 0.0550 |
