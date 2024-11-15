'''
temperal requirement:
(make an abstract class later?)
1. rewards functions should be `Callable` objects,
    with only one mandatory arg: `seq`:str
2. return log_rewards instead of rewards
'''

from .simple import simple_pattern,simple_pattern_v1
from .adcp_offline import offline_query
from typing import Dict,Callable

#TODO inspect callables
# import inspect
# inspect.signature(simple_pattern_v1)

REWARD_REGISTRY:Dict[str,Callable[[str],float]]={}
for k,v in tuple(locals().items()):
    if callable(v):
        REWARD_REGISTRY[k]=v
        

