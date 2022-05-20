
from sys import argv

from MDPLearner.logging import set_up_logging, DEBUG
from MDPLearner.frequentist import FrequentistLearner
from MDPLearner.simulator import Simulator
from MDPLearner.model import Model


if len(argv) <= 1:
    raise Exception("Enter path of prism file")

set_up_logging(DEBUG)

model = Model(argv[1])
simulator = Simulator(model)

simulator.mk_observations()
