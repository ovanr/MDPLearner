from typing import cast, List, Any, Dict
from copy import deepcopy
from math import ceil

import stormpy

State = int
Action = int
Probability = float
Matrix = Dict[State, Dict[Action, Dict[State, Probability]]]
PublicMatrix = Dict[State, Dict[Action, State]]
Scheduler = Dict[State, Action]


def freeze(x):
    return eval(x.__repr__())


def make_min_schedulers(matrix: Matrix) -> List[Scheduler]:
    """
        Makes the minimum number of schedulers
        such that every action is taken in at least one
        scheduler.
        Note that it may be the case that one action can still not be taken
        due to the combination of actions in a scheduler causing a later state
        not to be reached.
    """
    if len(matrix) == 0:
        return [{}]

    state = list(matrix.keys())[0]
    actions = list(matrix.pop(state).keys())

    schedulers = make_min_schedulers(matrix)

    if len(actions) < len(schedulers):
        actions = (actions * ceil(len(schedulers) / len(actions)))[:len(schedulers)]
    elif len(actions) > len(schedulers):
        schedulers = freeze(schedulers * ceil(len(actions) / len(schedulers)))[:len(actions)]

    for (i, sch) in enumerate(schedulers):
        sch[state] = actions[i]

    return schedulers


def make_all_schedulers(matrix: Matrix) -> List[Scheduler]:
    """
        Makes all the possible schedulers
    """
    if len(matrix) == 0:
        return [{}]

    state = list(matrix.keys())[0]
    actions = list(matrix.pop(state).keys())

    schedulers = make_all_schedulers(matrix)

    new_schedulers = []
    for a in actions:
        for sch in schedulers:
            new_sch = deepcopy(sch)
            new_sch[state] = a
            new_schedulers.append(new_sch)

    return new_schedulers


class Model:
    def __init__(self, model_path: str):
        self.prism_program = stormpy.parse_prism_program(model_path)  # type: ignore
        self.model = stormpy.build_model(self.prism_program)
        self.transition_matrix, _ = self._mk_transition_matrix()

    @property
    def storm_model(self):
        return self.model

    """
        Given a formula like 'Pmin=? ["s2"]'
        it returns a list of probabilities 
        for every state to the goal condition
    """

    def run_model(self, formula: str):
        properties = cast(List[Any], stormpy.parse_properties(formula, self.prism_program))
        result = stormpy.model_checking(self.model, properties[0])

        assert result.result_for_all_states

        return result.get_values()

    @property
    def initial_states(self) -> List[State]:
        return self.model.initial_states

    def num_states(self) -> int:
        return self.model.nr_states

    def _mk_transition_matrix(self) -> Matrix:
        matrix = {}
        public_matrix = {}
        for state in self.model.states:
            matrix[state.id] = {}
            public_matrix[state.id] = {}
            for action in state.actions:
                matrix[state.id][action.id] = {}
                public_matrix[state.id][action.id] = {}
                for transition in action.transitions:
                    matrix[state.id][action.id][transition.column] = transition.value()
                    public_matrix[state.id][action.id][transition.column] = None
        return matrix, public_matrix

    def mk_schedulers(self) -> List[Scheduler]:
        matrix, _ = self._mk_transition_matrix()
        return make_all_schedulers(matrix)

    def print_model(self):
        print("Number of states: {}".format(self.model.nr_states))
        print("Number of transitions: {}".format(self.model.nr_transitions))
        print("Labels in the model: {}".format(sorted(self.model.labeling.get_labels())))

        for state in self.model.states:
            initial = False
            if state.id in self.model.initial_states:
                initial = True

            for action in state.actions:
                for transition in action.transitions:
                    print(f"From{' initial' if initial else ''} state {state}, "
                          f"with probability {transition.value()}, "
                          f"go to state {transition.column}")
