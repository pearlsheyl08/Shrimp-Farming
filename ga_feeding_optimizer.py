# This script implements a Genetic Algorithm for optimizing shrimp feeding
# schedules based on environmental conditions and growth targets. The model uses
# a 7-day optimization window that can be repeated weekly.
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


IOT_DATA_PATH = Path("data/Data_Model_IoTMLCQ_2024.xlsx")
MEASUREMENT_DATA_PATH = Path("data/shrimp_measurement.csv")
OUTPUT_DIR = Path("outputs")

POPULATION_SIZE = 120
GENERATIONS = 200
CHROMOSOME_LENGTH = 7
# Chromosome = one complete 7-day feeding schedule.
# Gene = one daily feed amount in grams within that schedule.
MIN_FEED = 50.0
MAX_FEED = 180.0
CROSSOVER_RATE = 0.85
MUTATION_RATE = 0.08
ELITISM_COUNT = 4
TOURNAMENT_SIZE = 3

FIXED_FEED_AMOUNT = 150.0
FEED_COST_PER_GRAM = 0.02775
RANDOM_SEED = 42


def _environment_score(data):
    """Convert water quality values into one feeding-suitability score."""
    # environment_score summarizes temperature, pH, and dissolved oxygen.
    # Higher values mean pond conditions are safer for stronger feeding.
    temperature_score = 1 - (np.abs(data["temperature"] - 28.0) / 5.0)
    ph_score = 1 - (np.abs(data["pH"] - 7.8) / 1.2)
    oxygen_score = (data["dissolved_oxygen"] - 4.0) / 4.0

    score = (
        0.40 * np.clip(temperature_score, 0, 1)
        + 0.30 * np.clip(ph_score, 0, 1)
        + 0.30 * np.clip(oxygen_score, 0, 1)
    )
    return np.clip(score, 0, 1)


def load_data(
    iot_path=IOT_DATA_PATH,
    measurement_path=MEASUREMENT_DATA_PATH,
    chromosome_length=CHROMOSOME_LENGTH,
):
    """Load IoT water data and shrimp measurements for the GA model.

    The IoT data provides daily temperature, pH, and dissolved oxygen values.
    Shrimp measurements provide the baseline ABW and target ABW used to score
    each candidate feeding schedule.
    """
    iot_data = pd.read_excel(iot_path)
    measurements = pd.read_csv(measurement_path)

    required_iot_columns = [
        "Datetime",
        "Temperature (°C)",
        "pH",
        "Dissolved Oxygen (mg/L)",
    ]
    required_measurement_columns = ["treatment", "average_body_weight"]

    missing_iot = [col for col in required_iot_columns if col not in iot_data.columns]
    missing_measurements = [
        col for col in required_measurement_columns if col not in measurements.columns
    ]
    if missing_iot:
        raise ValueError(f"Missing IoT columns: {missing_iot}")
    if missing_measurements:
        raise ValueError(f"Missing measurement columns: {missing_measurements}")

    # Convert hourly IoT sensor readings into daily pond-condition records.
    iot_data = iot_data[required_iot_columns].copy()
    iot_data["Datetime"] = pd.to_datetime(iot_data["Datetime"], errors="coerce")
    iot_data = iot_data.dropna(subset=["Datetime"])
    iot_data["date"] = iot_data["Datetime"].dt.date

    daily_environment = (
        iot_data.groupby("date", as_index=False)
        .agg(
            temperature=("Temperature (°C)", "mean"),
            pH=("pH", "mean"),
            dissolved_oxygen=("Dissolved Oxygen (mg/L)", "mean"),
        )
    )

    sample_indexes = np.linspace(
        0, len(daily_environment) - 1, chromosome_length, dtype=int
    )
    # Select one environmental record for each gene in the 7-day chromosome.
    environment_data = daily_environment.iloc[sample_indexes].copy()

    if len(environment_data) < chromosome_length:
        raise ValueError("Not enough daily environment records for the chromosome length.")

    for column in ["temperature", "pH", "dissolved_oxygen"]:
        environment_data[column] = environment_data[column].fillna(
            environment_data[column].median()
        )

    measurements["normalized_treatment"] = (
        measurements["treatment"].astype(str).str.strip().str.lower()
    )
    # No-treatment ABW is the starting point; with-treatment ABW is the target.
    with_treatment = measurements[
        measurements["normalized_treatment"].eq("with treatment")
    ]
    no_treatment = measurements[
        measurements["normalized_treatment"].eq("no treatment")
    ]

    if with_treatment.empty or no_treatment.empty:
        raise ValueError("Expected both 'With Treatment' and 'No Treatment' groups.")

    baseline_abw = float(no_treatment["average_body_weight"].mean())
    target_abw = float(with_treatment["average_body_weight"].mean())

    environment_data = environment_data.reset_index(drop=True)
    environment_data["day"] = np.arange(1, len(environment_data) + 1)
    environment_data["environment_score"] = _environment_score(environment_data)
    environment_data["baseline_abw"] = baseline_abw
    environment_data["target_abw"] = target_abw
    environment_data["fixed_feed"] = FIXED_FEED_AMOUNT
    # safe_feed_limit lowers allowable feeding when water conditions are weaker.
    # This helps the GA avoid overfeeding that may increase waste and cost.
    environment_data["safe_feed_limit"] = FIXED_FEED_AMOUNT * (
        0.65 + 0.50 * environment_data["environment_score"]
    )

    return environment_data[
        [
            "day",
            "date",
            "temperature",
            "pH",
            "dissolved_oxygen",
            "environment_score",
            "safe_feed_limit",
            "fixed_feed",
            "baseline_abw",
            "target_abw",
        ]
    ]


def initialize_population(
    population_size=POPULATION_SIZE,
    chromosome_length=CHROMOSOME_LENGTH,
    min_feed=MIN_FEED,
    max_feed=MAX_FEED,
):
    """Create the first generation of possible 7-day feeding schedules."""
    # Each row is one candidate solution; each column is one daily feed gene.
    return np.random.uniform(
        min_feed,
        max_feed,
        size=(population_size, chromosome_length),
    )


def simulate_schedule(chromosome, data):
    """Estimate shrimp ABW growth for one candidate feeding schedule.

    Daily growth depends on the feed amount and the environment score. The
    simulation rewards balanced feeding and penalizes feeding above the safe
    limit when water conditions are less suitable.
    """
    chromosome = np.asarray(chromosome, dtype=float)
    current_abw = float(data["baseline_abw"].iloc[0])
    growth_trend = [current_abw]

    for feed, (_, row) in zip(chromosome, data.iterrows()):
        environment_score = row["environment_score"]
        # Better water conditions allow a higher effective feeding target.
        optimal_feed = 95.0 + 45.0 * environment_score
        feed_ratio = feed / optimal_feed
        if feed_ratio <= 1:
            feed_response = feed_ratio
        else:
            # Overfeeding gives less growth benefit and may harm pond quality.
            feed_response = max(0.65, 1 - 0.55 * (feed_ratio - 1))
        overfeed_ratio = max(feed - row["safe_feed_limit"], 0) / MAX_FEED
        daily_gain = 0.38 * environment_score * feed_response
        daily_gain -= 0.10 * overfeed_ratio
        current_abw += max(daily_gain, -0.05)
        growth_trend.append(current_abw)

    return np.array(growth_trend)


def fitness_function(chromosome, data):
    """Score a feeding schedule by balancing farm goals.

    The fitness value rewards higher estimated ABW and reaching the growth
    target. It subtracts penalties for feed usage, feed cost, excess feeding,
    and feeding heavily during weaker water conditions.
    """
    chromosome = np.asarray(chromosome, dtype=float)
    growth_trend = simulate_schedule(chromosome, data)
    final_abw = growth_trend[-1]
    target_abw = float(data["target_abw"].iloc[0])

    total_feed = float(chromosome.sum())
    feed_cost = total_feed * FEED_COST_PER_GRAM
    excess_feed = np.maximum(chromosome - data["safe_feed_limit"].to_numpy(), 0).sum()
    low_water_penalty = (chromosome * (1 - data["environment_score"].to_numpy())).sum()
    target_gap = max(target_abw - final_abw, 0)
    target_bonus = 35.0 if final_abw >= target_abw else 0.0

    # Higher fitness means a better trade-off between growth, cost, and safety.
    fitness = (
        final_abw * 90.0
        + target_bonus
        - target_gap * 120.0
        - total_feed * 0.070
        - feed_cost * 1.8
        - excess_feed * 0.35
        - low_water_penalty * 0.12
    )
    return float(fitness)


def selection(population, fitness_scores, tournament_size=TOURNAMENT_SIZE):
    """Choose a strong parent schedule using tournament selection."""
    # Tournament selection keeps pressure toward better feeding plans while
    # still allowing different schedules to compete.
    contender_indexes = np.random.choice(
        len(population), size=tournament_size, replace=False
    )
    best_index = contender_indexes[np.argmax(fitness_scores[contender_indexes])]
    return population[best_index].copy()


def crossover(parent_one, parent_two, crossover_rate=CROSSOVER_RATE):
    """Combine two parent schedules to create new feeding schedules."""
    if random.random() > crossover_rate or len(parent_one) < 2:
        return parent_one.copy(), parent_two.copy()

    # A crossover point swaps part of the weekly feeding pattern between parents.
    point = random.randint(1, len(parent_one) - 1)
    child_one = np.concatenate([parent_one[:point], parent_two[point:]])
    child_two = np.concatenate([parent_two[:point], parent_one[point:]])
    return child_one, child_two


def mutation(
    chromosome,
    mutation_rate=MUTATION_RATE,
    min_feed=MIN_FEED,
    max_feed=MAX_FEED,
):
    """Randomly adjust daily feed genes to explore new feeding options."""
    mutated = chromosome.copy()
    for index in range(len(mutated)):
        if random.random() < mutation_rate:
            # Mutation helps the GA discover schedules not found by crossover.
            mutated[index] += np.random.normal(0, 12)
            mutated[index] = np.clip(mutated[index], min_feed, max_feed)
    return mutated


def run_ga(data, seed=RANDOM_SEED, verbose=False, progress_interval=20):
    """Evolve feeding schedules over generations to find an optimal plan.

    The GA repeatedly scores, selects, crosses over, and mutates candidate
    schedules until it finds a strong 7-day feeding plan for the given shrimp
    and water-condition data.
    """
    np.random.seed(seed)
    random.seed(seed)

    population = initialize_population(
        POPULATION_SIZE,
        len(data),
        MIN_FEED,
        MAX_FEED,
    )
    history = []
    best_overall_schedule = None
    best_overall_fitness = -np.inf

    for generation in range(GENERATIONS):
        # Evaluate every candidate schedule using the shrimp-feeding objective.
        fitness_scores = np.array(
            [fitness_function(chromosome, data) for chromosome in population]
        )
        best_index = int(np.argmax(fitness_scores))
        best_fitness = float(fitness_scores[best_index])
        average_fitness = float(fitness_scores.mean())

        if best_fitness > best_overall_fitness:
            best_overall_fitness = best_fitness
            best_overall_schedule = population[best_index].copy()

        generation_number = generation + 1
        history.append(
            {
                "generation": generation_number,
                "best_fitness": best_fitness,
                "average_fitness": average_fitness,
            }
        )

        should_print_progress = (
            verbose
            and (
                generation_number == 1
                or generation_number % progress_interval == 0
                or generation_number == GENERATIONS
            )
        )
        if should_print_progress:
            print(
                f"Generation {generation_number}/{GENERATIONS} | "
                f"Best Fitness: {best_fitness:.3f} | "
                f"Average Fitness: {average_fitness:.3f}"
            )

        if generation_number == GENERATIONS:
            break

        # Elitism preserves the best schedules so good solutions are not lost.
        elite_indexes = np.argsort(fitness_scores)[-ELITISM_COUNT:]
        new_population = [population[index].copy() for index in elite_indexes]

        while len(new_population) < POPULATION_SIZE:
            # New schedules are created from selected parents, then mutated.
            parent_one = selection(population, fitness_scores)
            parent_two = selection(population, fitness_scores)
            child_one, child_two = crossover(parent_one, parent_two)
            new_population.append(mutation(child_one))
            if len(new_population) < POPULATION_SIZE:
                new_population.append(mutation(child_two))

        population = np.array(new_population)

    return best_overall_schedule.copy(), pd.DataFrame(history)


def _schedule_metrics(schedule, data, label):
    """Summarize feed use, cost, ABW, and target status for one schedule."""
    growth_trend = simulate_schedule(schedule, data)
    total_feed = float(np.sum(schedule))
    final_abw = float(growth_trend[-1])
    target_abw = float(data["target_abw"].iloc[0])

    return {
        "schedule": label,
        "total_feed_used_g": total_feed,
        "estimated_final_abw_g": final_abw,
        "feed_cost": total_feed * FEED_COST_PER_GRAM,
        "fitness_score": fitness_function(schedule, data),
        "growth_target_status": (
            "Target Growth Achieved"
            if final_abw >= target_abw
            else "Continue Optimization"
        ),
    }


def compare_with_fixed_schedule(best_schedule, data):
    """Compare the GA schedule with a traditional fixed feeding baseline."""
    # The fixed baseline represents feeding the same amount every day.
    # This shows whether GA adjustment improves feed use and estimated growth.
    fixed_schedule = np.full(len(data), FIXED_FEED_AMOUNT)
    comparison = pd.DataFrame(
        [
            _schedule_metrics(best_schedule, data, "GA Optimized Feeding"),
            _schedule_metrics(fixed_schedule, data, "Fixed Feeding Schedule"),
        ]
    )
    return comparison


def run_weekly_optimization(weekly_data, seed=RANDOM_SEED):
    """Repeat the short-term 7-day GA model across weekly data windows.

    The 7-day schedule is not the entire shrimp grow-out period. It is a
    short-term feeding window that can be rerun each week as new pond water
    data and shrimp growth observations become available.
    """
    if isinstance(weekly_data, pd.DataFrame):
        weekly_data = [weekly_data]

    weekly_results = []
    weekly_schedules = []

    for week_number, week_data in enumerate(weekly_data, start=1):
        best_schedule, history = run_ga(week_data, seed=seed + week_number - 1)
        comparison = compare_with_fixed_schedule(best_schedule, week_data)
        ga_result = comparison[comparison["schedule"].eq("GA Optimized Feeding")]

        weekly_schedules.append(best_schedule)
        best_generation = int(history.loc[history["best_fitness"].idxmax(), "generation"])
        weekly_results.append(
            {
                "week": week_number,
                "best_generation": best_generation,
                "best_fitness": float(history["best_fitness"].max()),
                "ga_total_feed_g": float(ga_result["total_feed_used_g"].iloc[0]),
                "ga_estimated_final_abw_g": float(
                    ga_result["estimated_final_abw_g"].iloc[0]
                ),
                "growth_target_status": ga_result["growth_target_status"].iloc[0],
            }
        )

    return weekly_schedules, pd.DataFrame(weekly_results)


def plot_results(best_schedule, history, data, output_dir=OUTPUT_DIR):
    """Create charts for GA learning, feeding decisions, and ABW trends."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fixed_schedule = np.full(len(data), FIXED_FEED_AMOUNT)
    days = data["day"].to_numpy()
    ga_growth = simulate_schedule(best_schedule, data)
    fixed_growth = simulate_schedule(fixed_schedule, data)

    plt.figure(figsize=(8, 4))
    plt.plot(history["generation"], history["best_fitness"], label="Best fitness")
    plt.plot(history["generation"], history["average_fitness"], label="Average fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness score")
    plt.title("GA Fitness Score per Generation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fitness_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(days, best_schedule, marker="o", label="GA optimized")
    plt.plot(days, fixed_schedule, marker="s", label="Fixed schedule")
    plt.plot(days, data["safe_feed_limit"], linestyle="--", label="Safe feed limit")
    plt.xlabel("Day")
    plt.ylabel("Feed amount (g)")
    plt.title("7-Day GA Feeding Window vs Fixed Feeding Schedule")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "feeding_schedule_comparison.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(0, len(data) + 1), ga_growth, marker="o", label="GA optimized")
    plt.plot(np.arange(0, len(data) + 1), fixed_growth, marker="s", label="Fixed schedule")
    plt.axhline(
        float(data["target_abw"].iloc[0]),
        color="gray",
        linestyle="--",
        label="Target ABW",
    )
    plt.xlabel("Day")
    plt.ylabel("Estimated ABW (g)")
    plt.title("Estimated ABW Trend During 7-Day Optimization Window")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "estimated_growth_trend.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    # Run the full workflow: load data, optimize feeding, compare, and save outputs.
    data = load_data()
    print("Running SmartShrimp Genetic Algorithm optimizer...\n")
    print("Optimization window: 7 days (repeatable as new weekly data becomes available)\n")
    best_schedule, history = run_ga(data, verbose=True, progress_interval=20)
    comparison = compare_with_fixed_schedule(best_schedule, data)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    schedule_output = data[["day", "temperature", "pH", "dissolved_oxygen"]].copy()
    schedule_output["ga_feed_g"] = np.round(best_schedule, 2)
    schedule_output["fixed_feed_g"] = FIXED_FEED_AMOUNT
    schedule_output["safe_feed_limit_g"] = np.round(data["safe_feed_limit"], 2)
    schedule_output.to_csv(OUTPUT_DIR / "seven_day_feeding_schedule.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "final_comparison_summary.csv", index=False)
    history.to_csv(OUTPUT_DIR / "ga_history.csv", index=False)

    best_history_row = history.loc[history["best_fitness"].idxmax()]
    comparison_by_schedule = comparison.set_index("schedule")
    ga_result = comparison_by_schedule.loc["GA Optimized Feeding"]
    fixed_result = comparison_by_schedule.loc["Fixed Feeding Schedule"]
    feed_saved = fixed_result["total_feed_used_g"] - ga_result["total_feed_used_g"]
    cost_saved = fixed_result["feed_cost"] - ga_result["feed_cost"]
    abw_improvement = (
        ga_result["estimated_final_abw_g"] - fixed_result["estimated_final_abw_g"]
    )

    print("\nFinal GA Summary")
    print(f"Best generation: {int(best_history_row['generation'])}")
    print(f"Best fitness score: {best_history_row['best_fitness']:.3f}")
    print(f"Final GA schedule (g/day): {np.round(best_schedule, 2)}")
    print(f"Total feed saved vs fixed feeding: {feed_saved:.2f} g")
    print(f"Feed cost saved vs fixed feeding: {cost_saved:.2f} pesos")
    print(f"ABW improvement vs fixed feeding: {abw_improvement:.3f} g")
    print(f"Growth target status: {ga_result['growth_target_status']}")

    print("\nGA vs fixed feeding comparison:")
    print(comparison.to_string(index=False))

    plot_results(best_schedule, history, data)
