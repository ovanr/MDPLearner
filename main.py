
from math import floor
import argparse

from MDPLearner.logging import set_up_logging, INFO, WARNING
from MDPLearner.frequentist import FrequentistLearner
from MDPLearner.dirichlet import DirichletLearner
from MDPLearner.simulator import Simulator
from MDPLearner.model import Model

parser = argparse.ArgumentParser(description='Model Learner', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('input', metavar='FILE', type=str, help='input model to learn')
parser.add_argument('-N', metavar='int', type=int, default=1000, help='number of simulator runs')
parser.add_argument('--batches', metavar='int', type=int, default=10, help='number of batches in dirichlet learner')
parser.add_argument('--laplace-smoothing', metavar='float', type=float, default=0.1, help='frequentist learner laplace smoothing value')
parser.add_argument('--use-coupled', default=False, action='store_true', help='assume transitions with same probability are coupled')
parser.add_argument('--init-alpha', metavar='float', type=float, default=0.0, help='dirichlet learner initial alpha parameter value')
parser.add_argument( '--verbose', default=False, action="store_true", help='be verbose')
args = parser.parse_args()

if args.verbose:
    set_up_logging(INFO)
else:
    set_up_logging(WARNING)

model = Model(args.input)
print("--------- Actual Learned Model -----------")
model.print_model()
model.gen_prism_model(model.transition_matrix, "out/model.prism")
simulator = Simulator(model, args.N)

observations = simulator.mk_observations()

print()
print("--------- Frequentist Learned Model -----------")
freq_learner = FrequentistLearner(model, observations, laplace_smoothing = args.laplace_smoothing, use_coupled_list=args.use_coupled)
frequentist_matrix = freq_learner.run_learner()
model.print_matrix(frequentist_matrix)
model.gen_prism_model(frequentist_matrix, "out/frequentist_model.prism")

print()
print("--------- Dirichlet Learned Model -----------")
learner = DirichletLearner(model, observations)
dirichlet_matrices = learner.run_learner_incremental(num_batches=args.batches, init_value=args.init_alpha)
batch_size = floor(len(observations) / args.batches)

for (i,m) in enumerate(dirichlet_matrices):
    print(f"Matrix {i+1}:")
    model.print_matrix(m)
    model.gen_prism_model(m, f"out/dirichlet_model_{(i+1)*batch_size}.prism")
