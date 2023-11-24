REGISTRY = {}

from .basic_controller import BasicMAC
from .central_basic_controller import CentralBasicMAC
from .maddpg_controller import MADDPGMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["basic_central_mac"] = CentralBasicMAC
REGISTRY["maddpg_mac"] = MADDPGMAC