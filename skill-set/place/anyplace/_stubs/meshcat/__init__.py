# stub: viz only. We pass a NoOp mc_vis; Visualizer must be importable.
class _N:
    def __init__(self,*a,**k): pass
    def __getattr__(self,n): return _N()
    def __call__(self,*a,**k): return _N()
    def __getitem__(self,k): return _N()
    def __setitem__(self,k,v): pass
class Visualizer(_N): pass
def __getattr__(name): return _N()
