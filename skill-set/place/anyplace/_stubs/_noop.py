class NoOp:
    def __init__(self, *a, **k): pass
    def __getattr__(self, n): return NoOp()
    def __call__(self, *a, **k): return NoOp()
    def __getitem__(self, k): return NoOp()
    def __setitem__(self, k, v): pass
    def __iter__(self): return iter(())
def noop(*a, **k): return None
