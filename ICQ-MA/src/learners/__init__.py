from .q_learner import QLearner
from .offpg_learner import OffPGLearner
from .bc_learner import BCLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["offpg_learner"] = OffPGLearner
REGISTRY["bc_learner"] = BCLearner
