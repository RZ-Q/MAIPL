from .q_learner import QLearner
from .offpg_learner import OffPGLearner
from .bc_learner import BCLearner
from .cpl_learner import CPLLearner
from .maicpl_learner import MAICPLLearner
from .hacpl_learner import HACPLLearner
from .dppo_learner import DPPOLearner
from .madpo_learner import MADPOLearner
from .maidpo_learner import MAIDPOLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["offpg_learner"] = OffPGLearner
REGISTRY["bc_learner"] = BCLearner
REGISTRY["cpl_learner"] = CPLLearner
REGISTRY["maicpl_learner"] = MAICPLLearner
REGISTRY["hacpl_learner"] = HACPLLearner
REGISTRY["dppo_learner"] = DPPOLearner
REGISTRY["madpo_learner"] = MADPOLearner
REGISTRY["maidpo_learner"] = MAIDPOLearner