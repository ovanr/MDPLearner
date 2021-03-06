from typing import cast, List, Any, Dict
from copy import deepcopy
from math import ceil

import stormpy
from string import ascii_lowercase

State = int
Action = int
Probability = float
Matrix = Dict[State, Dict[Action, Dict[State, Probability]]]
PublicMatrix = Dict[State, Dict[Action, State]]
Scheduler = Dict[State, Action]
CoupledList = List[List[tuple[State, Action, State]]]


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


def flatten_matrix(matrix: Matrix) -> Dict:
    final = {}
    for state in matrix.keys():
        for action in matrix[state].keys():
            for next_state in matrix[state][action].keys():
                final[(state, action, next_state)] = matrix[state][action][next_state]
    return final


class Model:
    def __init__(self, model_path: str):
        self.prism_program = stormpy.parse_prism_program(model_path)  # type: ignore
        self.model = stormpy.build_model(self.prism_program)
        self.transition_matrix: Matrix = self._mk_transition_matrix()
        self.coupled_probs: CoupledList = self.mk_coupled_list()

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

    def __getitem__(self, key: State) -> Dict[Action, Dict[State, Probability]]:
        return self.transition_matrix[key]

    @property
    def states(self) -> List[State]:
        return list(self.transition_matrix.keys())

    def num_states(self) -> int:
        return self.model.nr_states

    def _mk_transition_matrix(self) -> Matrix:
        matrix = {}
        for state in self.model.states:
            matrix[state.id] = {}
            for action in state.actions:
                matrix[state.id][action.id] = {}
                for transition in action.transitions:
                    matrix[state.id][action.id][transition.column] = transition.value()
        return matrix

    def mk_schedulers(self) -> List[Scheduler]:
        matrix = self._mk_transition_matrix()
        return make_all_schedulers(matrix)

    def mk_coupled_list(self) -> CoupledList:
        matrix = flatten_matrix(self.transition_matrix)
        l = []
        seen = []
        for key in matrix.keys():
            temp = [key]
            seen.append(key)
            for key2 in matrix.keys():
                if (key2 not in seen) and (key != key2) and (matrix[key] == matrix[key2]):
                    temp.append(key2)
                    seen.append(key2)
            if len(temp) != 1:
                l.append(temp)
        return l

    def update_probs_coupled(self, matrix: Matrix) -> Matrix:
        if not self.coupled_probs[0]:  # if list is empty
            return matrix

        # mean
        for elem in self.coupled_probs:
            temp = 0
            for trans in elem:
                temp += matrix[trans[0]][trans[1]][trans[2]]
            temp /= len(elem)
            for trans in elem:
                matrix[trans[0]][trans[1]][trans[2]] = temp

        # normalise
        for state in matrix.keys():
            for action in matrix[state].keys():
                temp = 0
                for new_state in matrix[state][action].keys():
                    temp += matrix[state][action][new_state]
                for new_state in matrix[state][action].keys():
                    matrix[state][action][new_state] /= temp

        return matrix

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
                    print(f"From{' initial' if initial else ''} state {state}, action {action} "
                          f"with probability {transition.value()}, "
                          f"go to state {transition.column}")

    def print_matrix(self, matrix: Matrix):
        print("Number of states: {}".format(len(matrix.keys())))
        print("Number of transitions: {}".format(self.model.nr_transitions))

        for state in matrix.keys():
            for action in matrix[state].keys():
                for next_state in matrix[state][action].keys():
                    print(f"From state {state}, action {action} "
                          f"with probability {matrix[state][action][next_state]}, "
                          f"go to state {next_state}")

    def gen_prism_model(self, matrix: Matrix, out_file: str):
        lines = []
        lines.append("mdp")
        lines.append("module main")
        lines.append(f"   s : [0..{self.num_states() - 1}] init {' '.join(map(str, self.initial_states))};")

        for state in matrix.keys():
            for action in matrix[state].keys():
                next_transitions = map(lambda pair: f"{pair[1]}:(s'={pair[0]})", matrix[state][action].items())
                lines.append(f"   [{ascii_lowercase[action]}] s={state} -> {' '.join(next_transitions)};")

        lines.append("endmodule")

        file = open(out_file, "w+")
        file.write("\n".join(lines))
        file.close()
         
