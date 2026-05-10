# SmartShrimp Output Analysis Report

Generated analysis date: 2026-05-11

## Overview

This report summarizes the generated outputs for the SmartShrimp genetic algorithm feeding optimizer. The outputs include the latest consecutive 7-day optimization window, the GA-vs-fixed feeding comparison, GA convergence history, visualization files, and the rolling-window evaluation across all possible consecutive 7-day periods in the IoT dataset.

The key result is that the GA consistently reduces feed use compared with the fixed 150 g/day baseline while maintaining or improving estimated ABW in the model simulation.

## Output Files Reviewed

| File | Purpose |
| --- | --- |
| `seven_day_feeding_schedule.csv` | Latest consecutive 7-day feeding schedule with water conditions, safe feed limits, GA feed, and fixed feed |
| `final_comparison_summary.csv` | GA vs fixed-feeding metrics for the latest 7-day window |
| `ga_history.csv` | Best and average GA fitness score per generation |
| `rolling_window_comparison.csv` | GA vs fixed-feeding metrics for every possible consecutive 7-day window |
| `rolling_window_summary.csv` | High-level summary of all rolling-window results |
| `fitness_curve.png` | GA convergence chart |
| `feeding_schedule_comparison.png` | Latest 7-day GA feed vs fixed feed and safe feed limit |
| `estimated_growth_trend.png` | Estimated ABW trend for GA vs fixed feed |
| `abw_treatment_comparison.png` | Treatment vs no-treatment ABW EDA chart |
| `rolling_window_feed_savings.png` | Feed savings over all rolling windows |
| `rolling_window_environment_vs_savings.png` | Environment score compared with GA feed savings |
| `rolling_window_savings_distribution.png` | Distribution of feed savings across rolling windows |

`SmartShrimp_GA.executed.ipynb` is an ignored old executed-notebook artifact. It is not part of the tracked project outputs and is not needed for the current analysis.

## Latest 7-Day Window Result

The latest default optimization window is:

| Day | Date | Environment Score | Safe Feed Limit | GA Feed | Fixed Feed |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | 2024-06-25 | 0.915 | 166.12 g | 136.17 g | 150.00 g |
| 2 | 2024-06-26 | 0.915 | 166.12 g | 136.12 g | 150.00 g |
| 3 | 2024-06-27 | 0.915 | 166.12 g | 136.18 g | 150.00 g |
| 4 | 2024-06-28 | 0.915 | 166.12 g | 136.20 g | 150.00 g |
| 5 | 2024-06-29 | 0.915 | 166.12 g | 136.16 g | 150.00 g |
| 6 | 2024-06-30 | 0.915 | 166.12 g | 136.15 g | 150.00 g |
| 7 | 2024-07-01 | 0.915 | 166.12 g | 136.19 g | 150.00 g |

The latest 7-day window has stable water conditions. Because the environmental inputs are identical across the 7 days, the GA schedule is also nearly flat. This is expected behavior.

### Latest Window Comparison

| Metric | GA Optimized | Fixed Feeding | Difference |
| --- | ---: | ---: | ---: |
| Total feed used | 953.18 g | 1050.00 g | 96.82 g saved |
| Feed saved percent | 9.22% | 0.00% | 9.22 percentage points |
| Feed cost | 26.45 pesos | 29.14 pesos | 2.69 pesos saved |
| Estimated final ABW | 22.366 g | 22.230 g | 0.136 g higher |
| Fitness score | 1923.890 | 1899.084 | 24.806 higher |
| Growth target status | Target Growth Achieved | Target Growth Achieved | Same status |

Interpretation: the latest-window result is good but modest. The GA uses less feed and slightly improves estimated ABW. The result is not visually dramatic because the water-condition data in this final 7-day window is flat.

## GA Convergence

The GA history shows improvement from the first generation to the final generations:

| Metric | Value |
| --- | ---: |
| Best generation | 197 |
| First-generation best fitness | 1907.716 |
| Final best fitness | 1923.890 |

Interpretation: the GA converges toward a stronger feeding schedule over time. Most gains happen before the end, with small refinements near the final generations.

## Rolling 7-Day Window Analysis

The rolling-window analysis tests every possible consecutive 7-day window in the daily IoT data.

| Metric | Value |
| --- | ---: |
| Windows tested | 177 |
| Date coverage | 2024-01-01 to 2024-07-01 |
| Target achieved windows | 149 / 177 |
| Target achieved rate | 84.18% |
| Windows with feed savings below zero | 0 |
| Windows with ABW improvement below zero | 0 |
| Windows with fitness improvement below zero | 0 |

This is the strongest output in the project because it does not depend on a single selected week. It shows the GA advantage across the available data period.

### Rolling-Window Summary

| Metric | Value |
| --- | ---: |
| Average feed saved | 128.76 g |
| Median feed saved | 125.95 g |
| Minimum feed saved | 94.78 g |
| Maximum feed saved | 180.38 g |
| Average feed saved percent | 12.26% |
| Average cost saved | 3.57 pesos |
| Average ABW improvement | 0.165 g |
| Minimum ABW improvement | 0.133 g |
| Maximum ABW improvement | 0.211 g |
| Average fitness improvement | 54.40 |

Interpretation: across all tested windows, the GA consistently outperforms the fixed 150 g/day feeding schedule in this simulation.

## Best Rolling Windows

### Best Feed-Saving Windows

| Rank | Window | Feed Saved | Feed Saved % | ABW Improvement | Average Environment Score | GA Schedule |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | 2024-02-13 to 2024-02-19 | 180.38 g | 17.18% | 0.211 g | 0.650 | 124.25; 124.14; 124.26; 124.29; 124.24; 124.23; 124.21 |
| 2 | 2024-02-04 to 2024-02-10 | 180.37 g | 17.18% | 0.211 g | 0.650 | 124.20; 124.24; 124.20; 124.24; 124.25; 124.25; 124.26 |
| 3 | 2024-02-07 to 2024-02-13 | 180.36 g | 17.18% | 0.211 g | 0.650 | 124.20; 124.24; 124.22; 124.27; 124.21; 124.23; 124.28 |

### Best ABW-Improvement Windows

| Rank | Window | Feed Saved | Feed Saved % | ABW Improvement | Average Environment Score | GA Schedule |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | 2024-02-20 to 2024-02-26 | 180.19 g | 17.16% | 0.211 g | 0.650 | 124.27; 124.25; 124.25; 124.26; 124.26; 124.26; 124.27 |
| 2 | 2024-02-03 to 2024-02-09 | 180.18 g | 17.16% | 0.211 g | 0.650 | 124.24; 124.26; 124.26; 124.26; 124.27; 124.26; 124.27 |
| 3 | 2024-02-17 to 2024-02-23 | 180.16 g | 17.16% | 0.211 g | 0.650 | 124.25; 124.27; 124.24; 124.26; 124.27; 124.28; 124.26 |

## Lowest Feed-Saving Windows

| Rank | Window | Feed Saved | Feed Saved % | ABW Improvement | Average Environment Score | GA Schedule |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | 2024-03-06 to 2024-03-12 | 94.78 g | 9.03% | 0.134 g | 0.921 | 136.46; 136.49; 136.44; 136.47; 136.49; 136.45; 136.43 |
| 2 | 2024-03-12 to 2024-03-18 | 94.78 g | 9.03% | 0.134 g | 0.921 | 136.45; 136.47; 136.46; 136.46; 136.46; 136.46; 136.45 |
| 3 | 2024-03-08 to 2024-03-14 | 94.80 g | 9.03% | 0.133 g | 0.921 | 136.46; 136.45; 136.50; 136.46; 136.45; 136.49; 136.39 |

Interpretation: the lowest savings occur during high environment-score windows. In these windows, feeding conditions are strong, so the GA stays closer to the fixed feeding level. Even then, the GA still saves feed and improves estimated ABW in the model.

## Environment Relationship

The rolling-window data shows a strong inverse relationship between average environment score and feed savings:

| Relationship | Correlation |
| --- | ---: |
| Average environment score vs feed saved | -1.000 |
| Average environment score vs ABW improvement | -1.000 |
| Feed saved vs ABW improvement | 1.000 |
| Feed saved vs fitness improvement | 0.845 |

Interpretation: lower environment-score periods produce larger feed savings because the GA avoids overfeeding when pond conditions are weaker. Higher environment-score periods still produce savings, but the schedule is closer to fixed feeding.

This behavior is consistent with the model design: `safe_feed_limit` and water-condition penalties discourage heavier feeding under weaker pond conditions.

## Chart Interpretation

### `fitness_curve.png`

Shows that the GA improves the best schedule over generations and stabilizes near the final best score.

### `feeding_schedule_comparison.png`

Shows the latest 7-day GA feed schedule compared with fixed feeding and safe feed limits. Since the latest water conditions are stable, the GA line is almost flat.

### `estimated_growth_trend.png`

Shows the GA and fixed-feeding estimated ABW trends. Both reach the target, but the GA ends slightly higher while using less feed.

### `abw_treatment_comparison.png`

Shows the source ABW difference between treatment groups. This chart is part of the EDA and is independent of the GA rolling-window change.

### `rolling_window_feed_savings.png`

Shows feed savings across all 177 tested windows. This is useful for seeing which periods benefit most from adaptive feeding.

### `rolling_window_environment_vs_savings.png`

Shows that feed savings are larger when average environment scores are lower.

### `rolling_window_savings_distribution.png`

Shows that all rolling windows have positive feed savings, mostly around 9% to 17%.

## Overall Assessment

The outputs are aligned with the project goal. The model now demonstrates both:

1. A concrete latest 7-day feeding recommendation.
2. A broader rolling-window evaluation across all possible 7-day periods.

The latest 7-day result is modest because the environment data is stable, but the rolling-window analysis gives stronger evidence that the GA is useful across the dataset.

The result should be described as a simulated decision-support result, not as a real farm recommendation. The growth model remains simplified and would need validation with farm-specific biomass, stocking density, mortality, feed conversion ratio, and production records before operational use.

