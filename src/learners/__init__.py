from .q_learner import QLearner
from .q_learner_differ import QLearner_differ
from .pref_qmix_learner import Pref_QMIX_learner
from .q_learner_globalRM import QLearnerGlobalRM
from .dmaq_qatten_learner import DMAQ_qattenLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["qdiffer_learner"] = QLearner_differ
REGISTRY["pref_qmix_learner"] = Pref_QMIX_learner
REGISTRY["q_learner_w_globalRM"] = QLearnerGlobalRM
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner