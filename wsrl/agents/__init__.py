from .anchor import AnchorAgent
from .calql import CalQLAgent
from .cql import CQLAgent
from .iql import IQLAgent
from .sac import SACAgent

agents = {
    "anchor": AnchorAgent,
    "wsrl": CQLAgent,
    "porl": CQLAgent,
    "iql": IQLAgent,
    "cql": CQLAgent,
    "calql": CalQLAgent,
    "sac": SACAgent,
}
