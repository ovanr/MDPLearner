
from typing import List, Dict
from copy import deepcopy
from math import floor
import logging

from MDPLearner.simulator import Observation
from MDPLearner.model import Model, Matrix, Action, State

K = int
Occurances = Dict[tuple[State,Action,State], K]
Alphas = Dict[State, Dict[Action, Dict[State, float]]]

class DirichletLearner:
    def __init__(self, model: Model, observations: List[Observation]):
        self.model = model
        self.observations = observations

    @property
    def logger(self):
        return logging.getLogger("Dirichlet")

    def _count_occurances(self, observations: list[Observation]) -> Occurances:
        occurances = {}
        for obs in observations:
            for tup in obs:
                if tup not in occurances.keys():
                    occurances[tup] = 1
                else:
                    occurances[tup] += 1

        return occurances
            
    def mk_new_transition_matrix(self) -> Matrix:
        new_matrix = deepcopy(self.model.transition_matrix)
        for state in new_matrix.keys():
            for action in new_matrix[state].keys():
                num_next_states = len(new_matrix[state][action].keys())
                for next_state in new_matrix[state][action].keys():
                    new_matrix[state][action][next_state] = 1 / num_next_states

        return new_matrix

    @staticmethod
    def _mk_init_alphas(shape: Matrix, init_value: float) -> Alphas:
        alphas = deepcopy(shape)
        for state in alphas.keys():
            for action in alphas[state].keys():
                for next_state in alphas[state][action].keys():
                    alphas[state][action][next_state] = init_value
        return alphas
                    
    def run_learner_incremental(self, num_batches: int, init_value: float) -> List[Matrix]:
        matrices = []
        alphas: Alphas = DirichletLearner._mk_init_alphas(self.model.transition_matrix, init_value)
        batch_size: int = floor(len(self.observations) / num_batches)

        for i in range(0, num_batches):
            batch = self._count_occurances(self.observations[i:(i+1)*batch_size])
            self.logger.info(f"Batch size: {(batch)}")
            matrix = self.mk_new_transition_matrix()

            for state in self.model.states:
                transitions = self.model[state]
                for action in transitions.keys():
                    m = len(transitions[action].keys())
                    ks = 0
                    for next_state in transitions[action].keys():
                        k = batch.get((state,action,next_state), 0)
                        alphas[state][action][next_state] += k
                        ks += alphas[state][action][next_state] 

                    for next_state in transitions[action].keys():
                        alpha = alphas[state][action][next_state]
                        matrix[state][action][next_state] = ((alpha - 1) / (ks - m))
            
            matrices.append(matrix)

        return matrices

    def run_learner_single_run(self, init_value: float):
        #   for every action a in the node:
        #       store Ai = 1 variables where i <= number of outgoing nodes
        #       Ai += Ki
        #       p(n, a, n_i) = Ai - 1 / (Sum of all Ai) - m
        matrix = self.mk_new_transition_matrix()
        occurances = self._count_occurances(self.observations)
        for state in self.model.states:
            transitions = self.model[state]
            for action in transitions.keys():
                m = len(transitions[action].keys())
                ks = 0
                for next_state in transitions[action].keys():
                    k = occurances[(state,action,next_state)]
                    ks += k + init_value
                for next_state in transitions[action].keys():
                    k = occurances[(state,action,next_state)]
                    matrix[state][action][next_state] = ((k + init_value - 1) / (ks - m))

        return matrix
