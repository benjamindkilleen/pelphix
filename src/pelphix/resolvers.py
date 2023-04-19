from typing import Optional
import logging
from omegaconf import OmegaConf

try:
    from detectron2 import model_zoo
except ImportError:
    model_zoo = None

log = logging.getLogger(__name__)


def merge_resolver(*configs):
    return OmegaConf.merge(*configs)


def model_zoo_resolver(name):
    if model_zoo is None:
        log.error("detectron2 is not installed")
        return None
    return model_zoo.get_config_file(name)


def checkpoint_resolver(name) -> Optional[str]:
    if model_zoo is None:
        log.error("detectron2 is not installed")
        return None
    try:
        return model_zoo.get_checkpoint_url(name)
    except RuntimeError:
        return None


OmegaConf.register_new_resolver("merge", merge_resolver)
OmegaConf.register_new_resolver("model_zoo", model_zoo_resolver)
OmegaConf.register_new_resolver("checkpoint", checkpoint_resolver)
