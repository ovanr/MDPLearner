from typing import Dict
from copy import deepcopy

from MDPLearner.model import Model, Probability, State, Action
from MDPLearner.simulator import Observation

ComputedProbs = Dict[tuple[State, Action, State], Probability]


class FrequentistLearner:
    def __init__(self, model: Model, observations: Observation, laplace_smoothing: float = 0.0,
                 use_coupled_list: bool = True):
        self.model = model
        self.matrix = model.transition_matrix
        self.observations = observations
        self.possible_observations = self.build_all_possible_observations()
        self.delta = laplace_smoothing
        self.use_coupled_list = use_coupled_list

    def build_all_possible_observations(self):
        possible_obs: Observation = []
        for state in self.matrix.keys():
            for action in self.matrix[state].keys():
                for next_state in self.matrix[state][action].keys():
                    possible_obs.append((state, action, next_state))
        return possible_obs

    def run_learner(self) -> ComputedProbs:
        """
        For every observation in observations, count how many times a transition was taken.
        :return: Dict with computed probabilities for every state, action, state tuple.
        """
        computedprobs: ComputedProbs = {}
        final = deepcopy(self.matrix)
        N: Dict[tuple[State, Action], int] = {}
        m: Dict[tuple[State, Action], int] = {}

        # init counts
        for tup in self.possible_observations:
            computedprobs[tup] = 0
            N[tuple[tup[0], tup[1]]] = 0
            if tuple[tup[0], tup[1]] in m:
                m[tuple[tup[0], tup[1]]] += 1
            else:
                m[tuple[tup[0], tup[1]]] = 1

        # count all transitions
        for sch in self.observations:
            for obs in sch:
                computedprobs[obs] += 1
                N[tuple[obs[0], obs[1]]] += 1

        # for every entry of the dict, divide by N transitions from a [state, action] (tuple[0] and tuple[1])
        for key in computedprobs.keys():
            final[key[0]][key[1]][key[2]] = (computedprobs[key] + self.delta) / (
                        N[tuple[key[0], key[1]]] + m[tuple[key[0], key[1]]] * self.delta)

        if self.use_coupled_list:
            final = self.model.update_probs_coupled(final)

        return final
