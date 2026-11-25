import json
import os

import jax
import orbax.checkpoint as ocp
from omegaconf import OmegaConf


def restore_params(path, item_name='params'):
    """
    Restores one item of the latest checkpoint in `path` onto the default device.
    """

    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    args = ocp.args.Composite(**{item_name: ocp.args.StandardRestore(fallback_sharding=sharding)})

    with ocp.CheckpointManager(os.path.abspath(path), item_names=[item_name]) as manager:
        return getattr(manager.restore(None, args=args), item_name)


def create_checkpoint_manager(path, keep_all=False):
    options = ocp.CheckpointManagerOptions(max_to_keep=None if keep_all else 1, create=True)
    return ocp.CheckpointManager(os.path.abspath(path), options=options)


def save_checkpoint(manager, step, **items):
    manager.save(step, args=ocp.args.Composite(**{name: ocp.args.PyTreeSave(item) for name, item in items.items()}))


def load_config(path, name='config.yaml'):
    return OmegaConf.load(os.path.join(path, name))


class Logger:
    """
    Appends metrics as JSON lines to `<path>/log.jsonl`.
    """

    def __init__(self, path):
        self.file = open(os.path.join(path, 'log.jsonl'), 'a')

    def __call__(self, step, metrics):
        metrics = {k: float(v) for k, v in metrics.items()}
        self.file.write(json.dumps({'step': step, **metrics}) + '\n')
        self.file.flush()
        print(step, ' '.join(f'{k}={v:.4g}' for k, v in metrics.items()), flush=True)

    def close(self):
        self.file.close()
