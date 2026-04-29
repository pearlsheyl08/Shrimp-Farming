# SmartShrimp: Genetic Algorithm-Based Feeding Optimization

SmartShrimp is a smart agriculture and aquaculture AI project that uses a Genetic Algorithm (GA) to optimize shrimp feeding decisions.

The project focuses on a **7-day feeding optimization window**. This does not represent the full shrimp grow-out cycle, which normally takes several months. Instead, the 7-day schedule is treated as a short-term feeding management plan that can be repeated weekly as new water-quality and shrimp-growth data become available.

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

The GA searches for a good 7-day feeding schedule.

- **Chromosome:** one complete 7-day feeding schedule
- **Gene:** feed amount for one day
- **Population:** many possible feeding schedules
- **Fitness function:** scores each schedule using estimated ABW, total feed used, feed cost, and environmental penalties
- **Selection:** chooses stronger schedules as parents
- **Crossover:** combines two parent schedules
- **Mutation:** randomly adjusts feed values
- **Elitism:** keeps the best schedules for the next generation

The optimized GA schedule is compared with a fixed feeding schedule that gives the same feed amount every day.

## Short-Term Optimization Concept

The 7-day schedule represents a short-term feeding optimization window, not the full shrimp grow-out cycle. In actual aquaculture, feeding decisions may be adjusted daily or weekly. Therefore, this 7-day GA schedule can be repeatedly applied across the grow-out period as new water-quality and shrimp-growth data become available.

The script includes `run_weekly_optimization()` to demonstrate how the same 7-day optimizer can be reused over time with updated weekly data.

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

## Project Notes

This is an educational AI optimization project. The growth model is simplified and is intended to demonstrate Genetic Algorithm concepts for smart aquaculture. It should not be used as a professional shrimp farming recommendation system without validation using real farm data, stocking density, feed conversion ratio, biomass, mortality, and expert aquaculture guidance.
