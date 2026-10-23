from ml_collections import config_dict

from .sdt import SequenceVDTContinuousModel


def create_model(cfg: config_dict.ConfigDict):
    """The CoBit Diffusion Transformer (cfg.model.name == "sdt")."""
    if str(cfg.model.name).lower() != "sdt":
        raise ValueError(f"Unknown model name: '{cfg.model.name}'")
    return SequenceVDTContinuousModel(cfg)
