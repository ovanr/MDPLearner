
## MDP-Learner

## Overview

This project implements an MDP learner using the frequentist and the bayesian approach.
When given an input model, the model will try both approaches and the resulting models will be produced in the `out/` folder.
Different options are available to configure the learning process.

## Requirements

This project depends on *stormpy*

### Usage

```
usage: main.py [-h] [-N int] [--batches int] [--laplace-smoothing float] [--use-coupled bool] [--init-alpha float] [--verbose] FILE

Model Learner

positional arguments:
  FILE                  input model to learn

options:
  -h, --help                  show this help message and exit
  -N int                      number of simulator runs (default: 1000)
  --batches int               number of batches in dirichlet learner (default: 10)
  --laplace-smoothing float   frequentist learner laplace smoothing value (default: 0.1)
  --use-coupled bool          assume transitions with same probability are coupled (default: False)
  --init-alpha float          dirichlet learner initial alpha parameter value (default: 0.0)
  --verbose                   be verbose (default: False)
```

### Example Usage

Learn the Knuth Die model using frequentist & bayesian approaches.
Run 2000 simulation runs for every scheduler, and the bayesian approach should learn all at once.
```
python main.py -N 2000 --batches 1 models/coin_flip.prism 
```

Learn the even-odd model using frequentist & bayesian approaches.
Run 2000 simulation runs for every scheduler, perform the bayesian learning in 10 incremental steps,
use 0.3 for laplace smoothing in frequentist,
use 0.2 for bayesian initial alpha parameter values, and be verbose.
```
python main.py -N 2000 --batches 10 --laplace-smoothing 0.3 --init-alpha 0.2 --verbose models/even_odd.prism 
```

Learn the even-odd model using frequentist & bayesian approaches.
Run 2000 simulation runs for every scheduler, and assume transitions with same probability are coupled.
```
python main.py -N 2000 --batches 1 --use-coupled models/even_odd.prism 
```
