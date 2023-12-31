from .q_learner import QLearner
from .q_learner_differ import QLearner_differ
from .pref_qmix_learner import Pref_QMIX_learner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["qdiffer_learner"] = QLearner_differ
REGISTRY["pref_qmix_learner"] = Pref_QMIX_learner