
from sys import argv

from MDPLearner.logging import set_up_logging, INFO, WARNING
from MDPLearner.frequentist import FrequentistLearner
from MDPLearner.dirichlet import DirichletLearner
from MDPLearner.simulator import Simulator
from MDPLearner.model import Model

logging_mode = WARNING

if len(argv) == 3 and argv[1] == "-v":
    logging_mode = INFO
    path = argv[2]
elif len(argv) == 2:
    path = argv[1]
else:
    raise Exception(f"Usage: {argv[0]} [-v] FILE")

set_up_logging(logging_mode)

model = Model(path)
simulator = Simulator(model)

observations = simulator.mk_observations()

model.print_model()

freq_learner = FrequentistLearner(model, observations)
freq_result = freq_learner.run_learner()
model.print_matrix(freq_result)

learner = DirichletLearner(model, observations)
new_matrix = learner.run_learner()
model.print_matrix(new_matrix)
