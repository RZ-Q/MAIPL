from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .q_learner_differ import QLearner_differ
from .max_q_learner_ddpg import DDPGQLearner
from .maddpg_learner import MADDPGLearner
from .pref_qmix_learner_wo_RM import Pref_QLearner
from .qdiffer_globalRM_learner import Qdiffer_globalRM_learner
from .pref_qmix_learner import Pref_QMIX_learner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["qdiffer_learner"] = QLearner_differ
REGISTRY["ddpg"] = DDPGQLearner
REGISTRY["maddpg"] = MADDPGLearner
REGISTRY["pref_q_learner_wo_RM"] = Pref_QLearner
REGISTRY["qdiffer_globalRM_learner"] = Qdiffer_globalRM_learner
REGISTRY["pref_qmix_learner"] = Pref_QMIX_learner