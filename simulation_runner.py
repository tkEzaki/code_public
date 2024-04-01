from drone_simulator import formal_experiment_drone
from elevator_simulator import formal_experiment_elevator
from multiprocessing import freeze_support


if __name__ == '__main__':
    freeze_support()

    formal_experiment_drone()
    formal_experiment_elevator()

    # simulations for appendix
    formal_experiment_drone(dt=600)
    formal_experiment_elevator(dt=600)
