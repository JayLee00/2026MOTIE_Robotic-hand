import os, os.path as osp


def get_anyplace_src() -> str:
    return os.environ['ANYPLACE_SOURCE_DIR']

def get_anyplace_config() -> str:
    return osp.join(get_anyplace_src(), 'config')

def get_anyplace_data() -> str:
    return os.environ['ANYPLACE_DATA_DIR']

def get_anyplace_eval_data() -> str:
    return osp.join(get_anyplace_src(), 'eval_data')

def get_anyplace_model_weights() -> str:
    return osp.join(get_anyplace_src(), 'model_weights')

def get_train_config_dir() -> str:
    return osp.join(get_anyplace_config(), 'train_cfgs')

def get_eval_config_dir() -> str:
    return osp.join(get_anyplace_config(), 'full_eval_cfgs')
