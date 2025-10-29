import matplotlib.pyplot as plt
import random

# --- Simulation Parameters ---
ROAD_LENGTH = 100  # Number of cells on the road
V_MAX = 5  # Maximum velocity (cells per step)
PROB_SLOW = 0.3  # Probability of random slowdown

WARMUP_STEPS = 200  # Steps to run before measuring to let system stabilize
MEASURE_STEPS = 500  # Steps to average flow over
DENSITY_STEPS = 100  # Number of density points to calculate (from 0.01 to 1.0)
N_RUNS = 100  # <<< NEW: Number of runs for ensemble average


class NaSchModel:
    """
    Implements the Nagel-Schreckenberg (NaSch) traffic model.

    Manages car positions and velocities and steps the simulation.
    """

    def __init__(self, length, v_max, p_slow, density):
        self.length = length
        self.v_max = v_max
        self.p_slow = p_slow
        self.density = density
        self.num_cars = int(self.length * self.density)

        # Store cars as a list of dictionaries
        self.cars = []  # List of {'pos': int, 'vel': int}
        self.init_cars()

    def init_cars(self):
        """Places cars randomly on the road with 0 velocity."""
        self.cars = []
        # Get a random sample of unique positions
        occupied_positions = random.sample(range(self.length), self.num_cars)
        for pos in occupied_positions:
            self.cars.append({'pos': pos, 'vel': 0})
        # Sort cars by position to make gap calculation easy
        self.cars.sort(key=lambda car: car['pos'])

    def step(self):
        """
        Performs a single step of the simulation following NaSch rules.
        Returns the total flow (sum of all velocities / road length) in this step.
        """
        if not self.cars:
            return 0

        # We'll store new velocities to avoid affecting calculations mid-step
        new_velocities = {}  # {car_index: new_vel}

        # 1. Calculate new velocities for all cars (Rules 1, 2, 3)
        for i in range(self.num_cars):
            car = self.cars[i]

            # Find gap to next car (periodic boundary)
            next_car_index = (i + 1) % self.num_cars
            next_car_pos = self.cars[next_car_index]['pos']

            if next_car_pos > car['pos']:
                gap = next_car_pos - car['pos'] - 1  # -1 for car length
            else:
                # Wrap-around case (e.g., car at pos 98, next at pos 5)
                gap = (self.length - car['pos']) + next_car_pos - 1

            # --- NaSch Rules ---
            # Rule 1: Acceleration
            new_vel = min(car['vel'] + 1, self.v_max)

            # Rule 2: Braking (Collision Avoidance)
            new_vel = min(new_vel, gap)

            # Rule 3: Randomization
            if random.random() < self.p_slow and new_vel > 0:
                new_vel = new_vel - 1

            new_velocities[i] = new_vel

        # 2. Update all cars' velocities and positions (Rule 4)
        total_velocity = 0
        for i in range(self.num_cars):
            car = self.cars[i]
            car['vel'] = new_velocities[i]
            car['pos'] = (car['pos'] + car['vel']) % self.length
            total_velocity += car['vel']

        # Re-sort cars by their new positions for the next step
        self.cars.sort(key=lambda car: car['pos'])

        # Flow (q) = (total velocity) / (road length)
        return total_velocity / self.length


def run_experiment():
    """
    Runs the NaSch simulation across a range of densities
    and plots the fundamental diagram.
    """
    print("Running NaSch model simulation...")
    print(f"Road Length: {ROAD_LENGTH}, V_max: {V_MAX}, P_slow: {PROB_SLOW}, N_Runs: {N_RUNS}")

    densities = []
    flows = []

    # Generate densities from 0.01 to 1.0
    for i in range(1, DENSITY_STEPS + 1):
        density = i / DENSITY_STEPS
        densities.append(density)

        run_flows = []
        # --- NEW: Ensemble Averaging Loop ---
        for run in range(N_RUNS):
            # Each run needs a new model with a new random start
            model = NaSchModel(ROAD_LENGTH, V_MAX, PROB_SLOW, density)

            # 1. Warm-up period (let simulation stabilize)
            for _ in range(WARMUP_STEPS):
                model.step()

            # 2. Measurement period (collect flow data)
            measured_flows = []
            for _ in range(MEASURE_STEPS):
                measured_flows.append(model.step())

            # Calculate average flow for this single run
            avg_flow_this_run = sum(measured_flows) / len(measured_flows)
            run_flows.append(avg_flow_this_run)

        # --- End of Ensemble Loop ---

        # Calculate the ensemble average flow for this density
        ensemble_avg_flow = sum(run_flows) / len(run_flows)
        flows.append(ensemble_avg_flow)

        # Print progress
        if i % 10 == 0:
            print(f"Calculated: Density={density:.2f}, Avg Ensemble Flow={ensemble_avg_flow:.3f}")

    print("Simulation complete. Plotting results...")

    # --- Plotting ---
    plt.figure(figsize=(10, 6))

    # Use a solid line ('-') for the curve
    plt.plot(densities, flows, 'b-', linewidth=2, label='Simulated Data (Ensemble Avg.)')

    plt.title(f'Fundamental Diagram (NaSch Model)\n$v_{{max}}={V_MAX}$, $p={PROB_SLOW}$, $N_{{runs}}={N_RUNS}$')
    plt.xlabel('Density (k)')
    plt.ylabel('Flow (q)')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(bottom=0)

    # Find and annotate the peak flow
    max_flow = max(flows)
    max_density = densities[flows.index(max_flow)]

    # Add a vertical dashed line at critical density
    plt.axvline(max_density, color='red', linestyle='--', linewidth=1, label=f'Critical Density ({max_density:.2f})')
    # Add a horizontal dashed line at peak flow
    plt.axhline(max_flow, color='red', linestyle='--', linewidth=1, label=f'Peak Flow ({max_flow:.3f})')

    plt.annotate(
        f'Peak Flow: {max_flow:.3f}\nat Density: {max_density:.2f} (Critical Density)',
        xy=(max_density, max_flow),
        xytext=(max_density + 0.1, max_flow * 0.8),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="b", lw=0.5, alpha=0.8)
    )
    plt.legend()

    plt.savefig("NaSch_fundamental_diagram.pdf", dpi=300)

    plt.show()


if __name__ == "__main__":
    run_experiment()

