from typing import Dict, List
from copy import deepcopy

from MDPLearner.model import Model, Probability, State, Action, Matrix, CoupledList
from MDPLearner.simulator import Observation

ComputedProbs = Dict[tuple[State, Action, State], Probability]


def computedProbsToDict(flatProbs):
    matrix = {}
    for (s, _, _) in flatProbs:
        next_pairs = filter(lambda key: s == key[0], flatProbs.keys())
        action_dict = {}
        for (_, a, s_next) in next_pairs:
            prob = flatProbs[(s, a, s_next)]
            if a in action_dict:
                action_dict[a][s_next] = prob
            else:
                action_dict[a] = {s_next: prob}

        matrix[s] = action_dict
    return matrix


class FrequentistLearner:
    def __init__(self, model: Model, observations: List[Observation], laplace_smoothing: float = 0.0,
                 use_coupled_list: bool = True):
        self.model = model
        self.matrix = model.transition_matrix
        self.observations = observations
        self.possible_observations = self.build_all_possible_observations()
        self.delta = laplace_smoothing
        self.use_coupled_list = use_coupled_list

    def build_all_possible_observations(self) -> Observation:
        possible_obs: Observation = []
        for state in self.matrix.keys():
            for action in self.matrix[state].keys():
                for next_state in self.matrix[state][action].keys():
                    possible_obs.append((state, action, next_state))
        return possible_obs

    def run_learner(self) -> Matrix:
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
            N[(tup[0], tup[1])] = 0
            if (tup[0], tup[1]) in m:
                m[(tup[0], tup[1])] += 1
            else:
                m[(tup[0], tup[1])] = 1

        # count all transitions
        for sch in self.observations:
            for obs in sch:
                computedprobs[obs] += 1
                N[(obs[0], obs[1])] += 1

        # for every entry of the dict, divide by N transitions from a [state, action]
        for key in computedprobs.keys():
            final[key[0]][key[1]][key[2]] = (computedprobs[key] + self.delta) / \
                                            (N[(key[0], key[1])] + m[(key[0], key[1])] * self.delta)

        if self.use_coupled_list:
            final = self.model.update_probs_coupled(final)

        return final
