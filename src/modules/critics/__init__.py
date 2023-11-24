from .coma import COMACritic
from .maddpg_ns import MADDPGCriticNS
REGISTRY = {}

REGISTRY["coma_critic"] = COMACritic
REGISTRY["maddpg_critic_ns"] = MADDPGCriticNS