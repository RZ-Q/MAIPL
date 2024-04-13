REGISTRY = {}

from .rnn_agent import RNNAgent
from .grnn_agent import GRNNAgent
from .rnn_ns_agent import RNNNSAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["grnn"] = GRNNAgent
REGISTRY["rnn_ns"] = RNNNSAgent