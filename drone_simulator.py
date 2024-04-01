import numpy as np
import queue
import time
import pandas as pd
import multiprocessing
import itertools


class Family:
    def __init__(self, floor, lambda_rate):
        self.floor = floor
        self.lambda_rate = lambda_rate

    def generate_request(self):
        return np.random.poisson(self.lambda_rate)  # Generate the number of requests with a Poisson process


class Drone:
    def __init__(self, service_time, charge_time, v_ascend=1, v_descend=1, t_detach=1.5, t_attach=1.5):
        self.service_time_full = service_time  # Service time
        self.charge_time = charge_time  # Charging time
        self.v_ascend = v_ascend  # Ascending speed
        self.v_descend = v_descend  # Descending speed
        self.t_detach = t_detach  # Detach time
        self.t_attach = t_attach  # Attach time
        self.next_available_time = 0  # Next available time
        self.service_time = service_time  # Remaining service time
        self.idle_time = 0  # Idle time
        self.num_recharge = 0  # Number of recharges

    def process_request(self, request, height=3):
        process_time = max(self.next_available_time, request['time'])
        self.idle_time += process_time - self.next_available_time
        process_duration = request['floor'] * height * (
            1 / self.v_ascend + 1 / self.v_descend) + self.t_detach + self.t_attach
        finish_time = process_time + process_duration
        self.service_time -= process_duration

        if self.service_time < 0:
            self.next_available_time = finish_time + self.charge_time
            self.service_time = self.service_time_full
            self.num_recharge += 1
        else:
            self.next_available_time = finish_time

        wait_time = finish_time - request['time']
        return wait_time


def simulate_system(F, M, D, request_rate, total_time, service_time, charge_time,
                    v_ascend=3, v_descend=3, t_detach=1.5, t_attach=1.5, call_loss=True, dt=1):
    "compute average waiting time and average queue length"
    drones = [Drone(service_time=service_time,
                    charge_time=charge_time,
                    v_ascend=v_ascend,
                    v_descend=v_descend,
                    t_detach=t_detach,
                    t_attach=t_attach)
              for _ in range(D)]
    request_queue = queue.Queue()

    current_time = 0
    wait_times = []
    endtime_list = []
    queue_length = [0]
    request_num = 0
    last_request_time = 0
    while current_time < total_time:
        if last_request_time < total_time + 120 and current_time % dt == 0:
            n_requests = np.random.poisson(request_rate * F * M * dt)
            request_num += n_requests
            requested_Floors = np.random.randint(1, F + 1, n_requests)
            for floor in requested_Floors:
                request = {'time': current_time, 'floor': floor}
                request_queue.put(request)

            while not request_queue.empty():
                request = request_queue.get()
                available_drones = sorted(
                    drones, key=lambda x: x.next_available_time)
                drone = available_drones[0]
                wait_time = drone.process_request(request, height=3)
                wait_times.append(wait_time)
                endtime_list.append(current_time + wait_time)
                last_request_time = current_time + wait_time

        current_time += 1

    drone_idle_time = np.mean([drone.idle_time for drone in drones])
    drone_recharge_num = np.mean([drone.num_recharge for drone in drones])
    return np.mean(wait_times), np.mean(queue_length), request_num, drone_idle_time, drone_recharge_num


def run_simulation(params):
    request_rate, F, M, D, service_time, charge_time, v_ascend, v_descend, t_detach, t_attach, dt = params

    waiting_time, queue_length, request_num, drone_idle_time, drone_recharge_num = simulate_system(
        F=F, M=M, D=D, request_rate=request_rate, total_time=1000000, service_time=service_time, charge_time=charge_time,
        v_ascend=v_ascend, v_descend=v_descend, t_detach=t_detach, t_attach=t_attach, dt=dt
    )
    return {
        'request_rate': request_rate,
        'F': F,
        'M': M,
        'D': D,
        'service_time': service_time,
        'charge_time': charge_time,
        'v_ascend': v_ascend,
        'v_descend': v_descend,
        't_detach': t_detach,
        't_attach': t_attach,
        'dt': dt,
        'waiting_time': waiting_time,
        'queue_length': queue_length,
        'request_num': request_num,
        'drone_idle_time': drone_idle_time,
        'drone_recharge_num': drone_recharge_num
    }


def formal_experiment_drone(dt: int = 1, core_num: int = 42):
    request_rates = [1 / 3600 * (1 + i) / 100 for i in range(100)]

    F_values = [50]
    M_values = [10]
    D_values = [(1 + i) for i in range(10)]
    service_time_values = [18 * 60]
    charge_time_values = [60]
    v_ascend_values = [3]
    v_descend_values = [2]
    t_detach_values = [10]
    t_attach_values = [10]
    dt = dt

    param_combinations = []
    for request_rate in request_rates:
        for F in F_values:
            for M in M_values:
                for D in D_values:
                    for service_time in service_time_values:
                        for charge_time in charge_time_values:
                            for v_ascend in v_ascend_values:
                                for v_descend in v_descend_values:
                                    for t_detach in t_detach_values:
                                        for t_attach in t_attach_values:
                                            param_combinations.append(
                                                (request_rate, F, M, D, service_time, charge_time,
                                                 v_ascend, v_descend, t_detach, t_attach, dt)
                                            )

    # Measure simulation time
    time_start = time.time()
    # Save results to DataFrame
    results_list = []

    with multiprocessing.Pool(core_num) as pool:
        results = pool.map(run_simulation, param_combinations)

    for result in results:
        results_list.append(result)

    # Convert list of results to DataFrame
    df = pd.DataFrame(results_list)

    # Save results to a CSV file
    df.to_csv(f'results\\simulation_results_drone_{dt}.csv', index=False)

    time_end = time.time()
    print(f'Computation time: {time_end - time_start} seconds')
    print("Results have been saved to a CSV file.")
