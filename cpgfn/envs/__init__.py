'''
temperal requirement:

`envs` could import other submodules,
while all other submodules should not have `envs` dependencies.

'''
from .addpep import AdditivePepEnv,ADCPCycEnv
from gfn.env import Env

# ENV_REGISTRY={}
# for k,v in tuple(locals().items()):
#     if isinstance(v, type) and issubclass(v,Env) and v is not Env:
#         ENV_REGISTRY[k]=v