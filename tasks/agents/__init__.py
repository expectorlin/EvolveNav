from .base_agent import MetaAgent

# import the agent class here
from .r2r import R2RAgent, R2RAugAgent
from .cvdn import CVDNAgent



def load_agent(name, *args, **kwargs):
    cls = MetaAgent.registry[name]
    return cls(*args, **kwargs)