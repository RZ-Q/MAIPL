REGISTRY = {}

from .rnn_agent import RNNAgent
from .central_rnn_agent import CentralRNNAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["central_rnn"] = CentralRNNAgent