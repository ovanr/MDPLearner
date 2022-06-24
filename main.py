
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
public_matrix = simulator.get_public_matrix()

freq_learner = FrequentistLearner(public_matrix, observations)
print(freq_learner.run_learner())

learner = DirichletLearner(model, observations)
new_matrix = learner.run_learner()
model.print_model()
model.print_matrix(new_matrix)
