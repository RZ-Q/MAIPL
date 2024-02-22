from .q_learner import QLearner
from .q_learner_differ import QLearner_differ
from .pref_qmix_learner import Pref_QMIX_learner
from .q_learner_globalRM import QLearnerGlobalRM
from .dmaq_qatten_learner import DMAQ_qattenLearner
from .max_q_learner import MAXQLearner
from .dmaq_qatten_learner_globalRM import DMAQ_qattenLearnerGlobalRM
from .max_q_learner_globalRM import MAXQLearnerGlobalRM
from .pref_qmix_learner_new import Pref_QMIX_learner_new
from .q_learner_new import QLearnerNew

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["q_learner_new"] = QLearnerNew
REGISTRY["qdiffer_learner"] = QLearner_differ
REGISTRY["pref_qmix_learner"] = Pref_QMIX_learner
REGISTRY["q_learner_w_globalRM"] = QLearnerGlobalRM
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner
REGISTRY["max_q_learner"] = MAXQLearner
REGISTRY["dmaq_qatten_learner_w_globalRM"] = DMAQ_qattenLearnerGlobalRM
REGISTRY["max_q_learner_w_globalRM"] = MAXQLearnerGlobalRM
REGISTRY["pref_qmix_learner_new"] = Pref_QMIX_learner_new