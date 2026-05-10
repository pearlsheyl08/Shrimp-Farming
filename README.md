# SmartShrimp: Genetic Algorithm-Based Feeding Optimization

SmartShrimp is a smart agriculture and aquaculture AI project that uses a Genetic Algorithm (GA) to optimize shrimp feeding decisions.

The project focuses on a **7-day consecutive feeding optimization window**. This does not represent the full shrimp grow-out cycle, which normally takes several months. Instead, the 7-day schedule is treated as a short-term feeding management plan that can be repeated weekly as new water-quality and shrimp-growth data become available.

## Project Objective

The objective is to optimize shrimp feed amounts so the model can:

- support shrimp growth toward a target average body weight (ABW)
- reduce unnecessary feed use
- reduce feed cost
- reduce possible feed waste
- respond to water-quality conditions such as temperature, pH, and dissolved oxygen

## Main Files

- `SmartShrimp_GA.ipynb`: Jupyter Notebook used for project explanation, data loading, preprocessing, exploratory data analysis, visualization, GA simulation, result interpretation, discussion, and conclusion.
- `ga_feeding_optimizer.py`: clean Python implementation of the Genetic Algorithm. It contains the reusable GA functions and can be run directly from the terminal.
- `requirements.txt`: Python libraries needed to run the notebook and script.

## Datasets Used

The project uses local datasets stored in the `data/` folder:

- `data/Data_Model_IoTMLCQ_2024.xlsx`: aquaculture IoT/environment dataset used for temperature, pH, dissolved oxygen, and daily water-condition inputs.
- `data/shrimp_measurement.csv`: shrimp measurement dataset with treatment group, average body weight, length, and volume.

The no-treatment shrimp ABW is used as the baseline starting condition. The with-treatment shrimp ABW is used as the target growth reference.

## Genetic Algorithm Method

The GA searches for a good 7-day feeding schedule using consecutive daily water-condition records. By default, the script uses the latest available 7-day window in the IoT dataset. A specific window can be selected by passing `window_start` to `load_data()`.

- **Chromosome:** one complete 7-day feeding schedule
- **Gene:** feed amount for one day
- **Population:** many possible feeding schedules
- **Fitness function:** scores each schedule using estimated ABW, total feed used, feed cost, and environmental penalties
- **Selection:** chooses stronger schedules as parents
- **Crossover:** combines two parent schedules
- **Mutation:** randomly adjusts feed values
- **Elitism:** keeps the best schedules for the next generation

The optimized GA schedule is compared with a fixed feeding schedule that gives the same feed amount every day.

## Algorithms Used

The project uses the following algorithms and evaluation methods:

- **Genetic Algorithm (GA):** main optimization algorithm used to search for an improved 7-day shrimp feeding schedule.
- **Fitness-based optimization:** each candidate schedule is scored using estimated final ABW, target gap, total feed used, feed cost, excess feed, and water-condition penalties.
- **Tournament selection:** parent schedules are selected by randomly sampling candidate schedules and choosing the strongest candidate from each tournament.
- **Single-point crossover:** two parent schedules exchange part of their 7-day feed pattern at one crossover point to create new child schedules.
- **Gaussian mutation:** individual daily feed values are randomly adjusted using normally distributed noise, then clipped within the allowed feed range.
- **Elitism:** the best schedules from each generation are preserved so strong solutions are not lost during crossover and mutation.
- **Environment scoring:** temperature, pH, and dissolved oxygen are converted into a 0-to-1 water-condition score that affects safe feeding limits and fitness penalties.
- **Growth simulation:** each candidate schedule is simulated over the 7-day window to estimate final average body weight (ABW).
- **Fixed-schedule baseline comparison:** the GA schedule is compared against a fixed 150 g/day feeding schedule.
- **Rolling 7-day window evaluation:** the GA is tested across every possible consecutive 7-day window in the IoT dataset to measure average feed savings, ABW improvement, and fitness improvement.

## Short-Term Optimization Concept

The 7-day schedule represents a short-term feeding optimization window, not the full shrimp grow-out cycle. In actual aquaculture, feeding decisions may be adjusted daily or weekly. Therefore, this 7-day GA schedule can be repeatedly applied across the grow-out period as new water-quality and shrimp-growth data become available.

The script includes `run_weekly_optimization()` to demonstrate how the same 7-day optimizer can be reused over time with updated weekly data.

The script also includes `run_rolling_window_analysis()` to evaluate every possible consecutive 7-day window in the IoT dataset. With the current 183 daily records, this tests 177 rolling windows. This prevents the project result from depending only on one selected week and gives stronger evidence about average feed savings, ABW improvement, and fitness improvement across the available data period.

## Growth Target Status

The project uses `growth_target_status` instead of a harvest decision.

- `Target Growth Achieved`: estimated final ABW reached or exceeded the short-term target ABW.
- `Continue Optimization`: estimated final ABW did not reach the short-term target ABW.

This status does not claim that the shrimp are ready for harvest. It only describes whether the short-term simulated growth target was achieved.

## Installation

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

## How To Run

To run the notebook, open and execute all cells in:

```text
SmartShrimp_GA.ipynb
```

To run the clean Python script from the terminal:

```bash
python ga_feeding_optimizer.py
```

The terminal version prints GA generation progress, the best generation, best fitness score, final GA schedule, feed saved, feed cost saved, ABW improvement, and growth target status.

## Outputs Generated

Running the notebook or script saves results in the `outputs/` folder:

- `outputs/ga_history.csv`: best and average fitness score per generation
- `outputs/seven_day_feeding_schedule.csv`: optimized feed, fixed feed, and water-condition values for the 7-day window
- `outputs/final_comparison_summary.csv`: GA vs fixed schedule comparison with `growth_target_status`
- `outputs/fitness_curve.png`: fitness score per generation
- `outputs/feeding_schedule_comparison.png`: 7-day GA feeding window vs fixed schedule
- `outputs/estimated_growth_trend.png`: estimated ABW trend during the 7-day optimization window
- `outputs/abw_treatment_comparison.png`: EDA chart comparing ABW by treatment group
- `outputs/rolling_window_comparison.csv`: GA vs fixed-feeding metrics for every consecutive 7-day window
- `outputs/rolling_window_summary.csv`: high-level summary of all rolling-window results
- `outputs/rolling_window_feed_savings.png`: feed savings across all tested 7-day windows
- `outputs/rolling_window_environment_vs_savings.png`: relationship between water-condition score and GA feed savings
- `outputs/rolling_window_savings_distribution.png`: distribution of feed savings across all tested windows

## Project Notes

This is an educational AI optimization project. The growth model is simplified and is intended to demonstrate Genetic Algorithm concepts for smart aquaculture. It should not be used as a professional shrimp farming recommendation system without validation using real farm data, stocking density, feed conversion ratio, biomass, mortality, and expert aquaculture guidance.
