from jetnet.utils import jet_features

from fgsim.config import conf
from fgsim.datasets.jetnet.utils import to_stacked_mask


def postprocess(batch):
    if len({"kpd", "fgd"} & set(conf.training.val.metrics)):
        from fgsim.datasets.jetnet.utils import to_efp

        batch = to_efp(batch)
    for k, v in jet_features(to_stacked_mask(batch).cpu().numpy()[..., :3]).items():
        batch[k] = v
    return batch
