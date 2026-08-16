def mesh_to_voxels(*a, **k):
    raise NotImplementedError("mesh_to_sdf stub: not needed on inference path")
def __getattr__(name):
    def _f(*a, **k):
        raise NotImplementedError("mesh_to_sdf stub: "+name)
    return _f
