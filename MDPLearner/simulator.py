import stormpy
import stormpy.core
import stormpy.simulator
import logging

from stormpy.simulator import SparseSimulator

from sys import stderr
from typing import List, cast

from MDPLearner.model import Model, Scheduler, State, Action

Observation = List[tuple[State,Action,State]]

def oops(err):
    print(err, file=stderr)
    exit(1)
    
class Simulator:
    def __init__(self, model: Model):
        self.model = model
        self._simulator = stormpy.simulator.create_simulator(model.storm_model, seed=42)
        if not self._simulator:
            oops("simulator not loaded")

        self._cur_state, _, _ = self.simulator.restart()

    @property
    def simulator(self) -> SparseSimulator:
        if not self._simulator:
            oops("simulator not loaded")
        return cast(SparseSimulator, self._simulator)

    @property
    def logger(self):
        return logging.getLogger("Simulator")

    @property
    def cur_state(self) -> State:
        return self._cur_state

    def reset(self):
        self._cur_state, _, _ = self.simulator.restart()

    @property
    def available_actions(self):
        return self.simulator.available_actions()

    def take_action(self, action) -> State:
        self._cur_state, _, _ = self.simulator.step(action)
        return self._cur_state
        
    @property
    def is_in_sink_state(self) -> bool:
        if self.simulator:
            return self.simulator.is_done()
        else:
            return False

    def run_simulator_with_scheduler(self, scheduler: Scheduler) -> List[Observation]:
        observations: List[Observation] = []
        runs = self.model.num_states() * 10
        self.logger.info(f"Running with scheduler: {scheduler}")
    
        for _ in range(0, runs):
            self.reset()
            observation: Observation = []
            is_in_sink = False

            for _ in range(0, self.model.num_states() + 10):
                prev_state = self.cur_state
                action = scheduler[prev_state]
                next_state = self.take_action(action)
                observation.append((prev_state, action, next_state))
                self.logger.info(f"From state {prev_state} we took action {action} and went to {next_state}")

                if is_in_sink:
                    break
                if self.is_in_sink_state:
                    is_in_sink = True

            observations.append(observation)

        return observations

    def mk_observations(self) -> List[Observation]:
        obs: List[Observation] = []

        for scheduler in self.model.mk_schedulers():
            obs.extend(self.run_simulator_with_scheduler(scheduler))

        return obs
            

