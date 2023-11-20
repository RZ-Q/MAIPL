REGISTRY = {}

from .basic_controller import BasicMAC
from .central_basic_controller import CentralBasicMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["basic_central_mac"] = CentralBasicMAC