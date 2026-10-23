from __future__ import annotations

from torch.utils.data import DataLoader
from ml_collections import config_dict


def _module(config):
    name = str(config.data.dataset)
    if name == "LM1B":
        from . import lm1b
        return lm1b, lm1b.LM1BDataset
    if name == "OpenWebText":
        from . import openwebtext
        return openwebtext, openwebtext.OpenWebTextDataset
    if name == "tinygsm":
        from . import tinygsm
        return tinygsm, tinygsm.TinyGSMDataset
    raise NotImplementedError(f"Unknown dataset '{name}' (LM1B, OpenWebText, tinygsm)")


def get_loader(config: config_dict.ConfigDict, *, split: str, batch_size=None, shuffle=None, drop_last=None) -> DataLoader:
    """One split, in fixed order unless it is the train split (used by evaluation)."""
    _, dataset_cls = _module(config)
    num_workers = int(getattr(config.data, "num_workers", 4))
    return DataLoader(
        dataset_cls(config, split=split),
        batch_size=int(batch_size or config.train.batch_size),
        shuffle=(split == "train") if shuffle is None else shuffle,
        drop_last=(split != "test") if drop_last is None else drop_last,
        num_workers=num_workers,
        pin_memory=bool(getattr(config.data, "pin_memory", True)),
        persistent_workers=num_workers > 0,
        prefetch_factor=int(getattr(config.data, "prefetch_factor", 2)) if num_workers > 0 else None,
    )


def get_dataloaders(config: config_dict.ConfigDict, *, batch_size=None, seed: int = 42):
    """(train, val, test-or-None) loaders for training."""
    module, _ = _module(config)
    return module.get_dataloaders(config, batch_size=batch_size, seed=seed)
