
from typing import List, Dict
from copy import deepcopy
from MDPLearner.simulator import Observation
from MDPLearner.model import Model, Matrix, Action, State

Occurances = Dict[tuple[State,Action,State],int]

class DirichletLearner:
    def __init__(self, model: Model, observations: List[Observation]):
        self.model = model
        self.observations = observations

    def _count_occurances(self) -> Occurances:
        occurances = {}
        for obs in self.observations:
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

    def run_learner(self):
        #   for every action a in the node:
        #       store Ai = 1 variables where i <= number of outgoing nodes
        #       Ai += Ki
        #       p(n, a, n_i) = Ai - 1 / (Sum of all Ai) - m
        matrix = self.mk_new_transition_matrix()
        occurances = self._count_occurances()
        for state in self.model.states:
            transitions = self.model[state]
            for action in transitions.keys():
                num_next_states = len(transitions[action].keys())
                ks = 0
                for next_state in transitions[action].keys():
                    k = occurances[(state,action,next_state)]
                    ks += 1 + k     
                for next_state in transitions[action].keys():
                    k = occurances[(state,action,next_state)]
                    matrix[state][action][next_state] = (k / ks - num_next_states)

        return matrix
