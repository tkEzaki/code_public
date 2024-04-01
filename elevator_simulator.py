import numpy as np
import heapq
import time
import pandas as pd
import multiprocessing


class Family:
    def __init__(self, floor, lambda_rate):
        self.floor = floor
        self.lambda_rate = lambda_rate

    def generate_request(self, dt=1):
        return np.random.poisson(self.lambda_rate * dt)  # Generate the number of requests using a Poisson process

# Elevator simulation class


class ElevatorSimulation:
    def __init__(self, families, t_attach, t_detach, t_open, t_close, v_ascend, v_descend):
        self.families = families
        self.t_attach = t_attach
        self.t_detach = t_detach
        self.t_open = t_open  # Time to open the doors
        self.t_close = t_close  # Time to close the doors
        self.v_ascend = v_ascend  # Ascending speed
        self.v_descend = v_descend  # Descending speed
        self.pending_requests = []  # Requests waiting to be loaded
        self.loaded_requests = []  # Loaded requests
        self.wait_times = []  # Waiting time for requests
        self.current_time = 0
        self.last_loading_time = 0
        self.current_floor = 1
        self.total_distance = 0
        self.door_operations = 0
        self.idle_time = 0
        self.delivery_times = []
        self.num_travels = 0

    def floor_distance(self, floor1, floor2):
        return abs(floor1 - floor2)

    def run_simulation(self, simulation_time, dt=1):
        while self.current_time < simulation_time:
            # Elevator operation
            if self.pending_requests:  # If there are requests waiting to be loaded
                self.num_travels += 1
                self.load_requests()
                self.last_loading_time = self.current_time
                self.process_requests()
                last_delivery_round = int(self.last_loading_time / dt)
                latest_delivery_round = int(self.current_time / dt)

                for round in range(latest_delivery_round - last_delivery_round):
                    for family in self.families:
                        request_num = family.generate_request(dt)
                        for _ in range(request_num):
                            heapq.heappush(
                                self.pending_requests, (family.floor, (last_delivery_round + round + 1) * dt))
            else:  # If there are no requests waiting to be loaded
                self.current_time += 1
                self.idle_time += 1

                if int(self.current_time) % dt == 0:
                    for family in self.families:
                        request_num = family.generate_request(dt)
                        for _ in range(request_num):
                            heapq.heappush(self.pending_requests,
                                           (family.floor, self.current_time))

        self.calculate_statistics()

    def load_requests(self):
        while self.pending_requests:
            floor, request_time = heapq.heappop(self.pending_requests)
            heapq.heappush(self.loaded_requests, (floor, request_time))
            self.current_time += self.t_attach

    def process_requests(self):
        delivered_floors = set()
        while self.loaded_requests:
            floor, request_time = heapq.heappop(self.loaded_requests)

            if floor not in delivered_floors:
                self.move_to_floor(floor)
                self.stop_and_open_doors()
                self.deliver_items()
                self.wait_times.append(self.current_time - request_time)
                delivered_floors.add(floor)
            else:  # If there's already a request for that floor
                self.deliver_items()
                self.wait_times.append(self.current_time - request_time)

        # Return to the first floor
        self.move_to_floor(0)

    def move_to_floor(self, floor):
        distance = 3 * self.floor_distance(self.current_floor, floor)
        self.total_distance += distance
        move_time = distance / self.v_ascend if self.current_floor < floor else distance / self.v_descend
        self.current_time += self.t_close
        self.current_time += move_time
        self.current_floor = floor

    def stop_and_open_doors(self):
        self.current_time += self.t_open
        self.door_operations += 1

    def deliver_items(self):
        self.current_time += self.t_detach

    def calculate_statistics(self):
        total_wait_time = sum(self.wait_times)
        average_wait_time = total_wait_time / len(self.wait_times) if self.wait_times else 0
        total_time = self.current_time - self.idle_time
        utilization_rate = total_time / self.current_time * 100 if self.current_time > 0 else 0
        delivery_num = len(self.wait_times)
        return {
            "total_distance": self.total_distance,
            "door_operations": self.door_operations,
            "utilization_rate": utilization_rate,
            "average_wait_time": average_wait_time,
            "total_wait_time": total_wait_time,
            "delivery_num": delivery_num,
            "num_travels": self.num_travels
        }


def simulate_system(F, M, request_rate, total_time,
                    v_ascend, v_descend, t_detach, t_attach, dt=1):
    # Simulation parameters
    families = [Family(floor=f + 1, lambda_rate=request_rate)
                for f in range(F) for _ in range(M)]

    t_open, t_close = 10, 10

    # Execute the simulation
    simulation = ElevatorSimulation(
        families, t_attach, t_detach, t_open, t_close, v_ascend, v_descend)
    simulation.run_simulation(total_time, dt)  # Simulate for 1 hour
    statistics = simulation.calculate_statistics()
    return statistics


def run_simulation(params):
    request_rate, F, M, v_ascend, v_descend, t_detach, t_attach, dt = params

    result = simulate_system(
        F=F, M=M, request_rate=request_rate, total_time=1000000,
        v_ascend=v_ascend, v_descend=v_descend, t_detach=t_detach, t_attach=t_attach, dt=dt
    )
    return {
        'request_rate': request_rate,
        'F': F,
        'M': M,
        'v_ascend': v_ascend,
        'v_descend': v_descend,
        't_detach': t_detach,
        't_attach': t_attach,
        'dt': dt,
        "total_distance": result["total_distance"],
        "door_operations": result["door_operations"],
        "utilization_rate": result["utilization_rate"],
        "average_wait_time": result["average_wait_time"],
        "total_wait_time": result["total_wait_time"],
        "delivery_num": result["delivery_num"],
        "num_travels": result["num_travels"]
    }


def formal_experiment_elevator(dt: int = 1, core_num: int = 42):
    # Create a list of parameters for the simulation
    request_rates = [1 / 3600 * (1 + i) / 100 for i in range(100)]
    F_values = [50]
    M_values = [10]
    v_ascend_values = [1.5]
    v_descend_values = [1.5]
    t_detach_values = [0]
    t_attach_values = [0]
    dt = dt

    param_combinations = []
    for request_rate in request_rates:
        for F in F_values:
            for M in M_values:
                for v_ascend in v_ascend_values:
                    for v_descend in v_descend_values:
                        for t_detach in t_detach_values:
                            for t_attach in t_attach_values:
                                param_combinations.append(
                                    (request_rate, F, M, v_ascend, v_descend, t_detach, t_attach, dt)
                                )

    # Measure simulation time
    time_start = time.time()
    # Save results to a DataFrame
    results_list = []

    with multiprocessing.Pool(core_num) as pool:
        results = pool.map(run_simulation, param_combinations)

    for result in results:
        results_list.append(result)

    # Convert list of results to a DataFrame
    df = pd.DataFrame(results_list)

    # Save results to a CSV file
    df.to_csv(f'results\\simulation_results_elevator_{dt}.csv', index=False)

    time_end = time.time()
    print(f'Computation time: {time_end - time_start} seconds')
    print("Results have been saved to a CSV file.")
