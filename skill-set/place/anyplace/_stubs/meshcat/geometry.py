class _N:
    def __init__(self,*a,**k): pass
    def __getattr__(self,n): return _N()
    def __call__(self,*a,**k): return _N()
def __getattr__(name): return _N()
