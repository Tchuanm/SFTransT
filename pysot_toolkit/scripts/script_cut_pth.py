import torch
import os
import sys
from pathlib import Path
import importlib
import inspect


# import ltr.admin.settings as ws_settings


def torch_load_legacy(path):
    """Load network with legacy environment."""

    # Setup legacy env (for older networks)
    _setup_legacy_env()

    # Load network
    checkpoint_dict = torch.load(path, map_location='cpu')

    # Cleanup legacy
    _cleanup_legacy_env()

    return checkpoint_dict


def _setup_legacy_env():
    importlib.import_module('ltr')
    sys.modules['dlframework'] = sys.modules['ltr']
    sys.modules['dlframework.common'] = sys.modules['ltr']
    importlib.import_module('ltr.admin')
    sys.modules['dlframework.common.utils'] = sys.modules['ltr.admin']
    for m in ('model_constructor', 'stats', 'settings', 'local'):
        importlib.import_module('ltr.admin.' + m)
        sys.modules['dlframework.common.utils.' + m] = sys.modules['ltr.admin.' + m]


def _cleanup_legacy_env():
    del_modules = []
    for m in sys.modules.keys():
        if m.startswith('dlframework'):
            del_modules.append(m)
    for m in del_modules:
        del sys.modules[m]


if __name__ == '__main__':

    pth_path= os.path.dirname(__file__) + '/../../ltr/sftranst/sftranst_train/sftranst_ep80.pth.tar'
    pth_dict = torch_load_legacy(pth_path)  ## load pth

    keys_old_pth = []  ##
    for k in pth_dict:
        print("before_cut:%s" % (str(k)))
        keys_old_pth.append(k)

    keys_we_need = ['model']  #
    for key in keys_old_pth:
        if key not in keys_we_need:
            del pth_dict[key]  ##

    for k in pth_dict:
        print("after_cut:%s" % (str(k)))

    torch.save(pth_dict, pth_path)
