from .q_learner import QLearner
from .offpg_learner import OffPGLearner
from .bc_learner import BCLearner
from .cpl_learner import CPLLearner
from .macpl_learner import MACPLLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["offpg_learner"] = OffPGLearner
REGISTRY["bc_learner"] = BCLearner
REGISTRY["cpl_learner"] = CPLLearner
REGISTRY["macpl_learner"] = MACPLLearner