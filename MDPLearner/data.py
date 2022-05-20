


class Model:
    def __init__(self, model_path: str):
        self.model

    def print_model(self):
        print("Number of states: {}".format(model.nr_states))
        print("Number of transitions: {}".format(model.nr_transitions))
        print("Labels in the model: {}".format(sorted(model.labeling.get_labels())))
        for state in model.states:
            initial = False
            if state.id in model.initial_states:
                initial = True
        
            for action in state.actions:
                for transition in action.transitions:
                    print(f"From{ ' initial' if initial else '' } state {state}, "
                          f"with probability {transition.value()}, "
                          f"go to state {transition.column}")
        
    prism_program = stormpy.parse_prism_program(argv[1]) #type: ignore
    formula_str = 'Pmin=? ["s2"]'
    properties = cast(List[Any], stormpy.parse_properties(formula_str, prism_program))
    model = stormpy.build_model(prism_program)
    result = stormpy.model_checking(model, properties[0])
    assert result.result_for_all_states
    print("Result Vector: ", result.get_values())
    
    print_model(model)

