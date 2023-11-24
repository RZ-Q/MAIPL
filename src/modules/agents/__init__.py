REGISTRY = {}

from .rnn_agent import RNNAgent
from .central_rnn_agent import CentralRNNAgent
from .mlp_agent import MLPAgent
from .mlp_ns_agent import MLPNSAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["central_rnn"] = CentralRNNAgent
REGISTRY["mlp"] = MLPAgent
REGISTRY["mlp_ns"] = MLPNSAgent