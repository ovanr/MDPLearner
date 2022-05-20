
from sys import argv

from MDPLearner.logging import set_up_logging, INFO, WARNING
from MDPLearner.frequentist import FrequentistLearner
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

simulator.mk_observations()
