# stub: AnyPlace only uses airobot for logging (log_warn/log_debug/...).
def _noop(*a, **k): return None
def __getattr__(name): return _noop
