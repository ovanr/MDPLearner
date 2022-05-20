import stormpy
import stormpy.core
import stormpy.simulator
from typing import List, Any, cast
from sys import argv, stderr

if len(argv) == 1:
    raise Exception("Enter path of prism file")

def oops(err):
    print(err, file=stderr)
    exit(1)
    
class Simulator:
    def run_simulator(model):
        simulator = stormpy.simulator.create_simulator(model, seed=42)
        if not simulator:
            oops("simulator not loaded")
    
        final_outcomes = dict()
    
        for _ in range(1000):
            while not simulator.is_done():
                observation, reward, labels = simulator.step()
            print(labels)
            if observation not in final_outcomes:
                final_outcomes[observation] = 1
            else:
                final_outcomes[observation] += 1
            simulator.restart()
    
        options = stormpy.BuilderOptions([])
        options.set_build_state_valuations()
        options.set_build_all_reward_models()
        model = stormpy.build_sparse_model_with_options(prism_program, options)
        simulator = stormpy.simulator.create_simulator(model, seed=42)
        simulator.set_observation_mode(stormpy.simulator.SimulatorObservationMode.PROGRAM_LEVEL)
        final_outcomes = dict()
        print(simulator.get_reward_names())
        for n in range(1000):
            while not simulator.is_done():
                observation, reward, labels = simulator.step()
            if observation not in final_outcomes:
                final_outcomes[observation] = 1
            else:
                final_outcomes[observation] += 1
            simulator.restart()
        print(", ".join([f"{str(k)}: {v}" for k,v in final_outcomes.items()]))