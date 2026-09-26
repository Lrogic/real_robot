from tabletop_sim.embodiments.base import EmbodimentCfg
from tabletop_sim.embodiments.menagerie_wxai import MENAGERIE_WXAI
from tabletop_sim.embodiments.trossen_wxai import TROSSEN_WXAI_BASE

EMBODIMENTS: dict[str, EmbodimentCfg] = {
  emb.name: emb for emb in (MENAGERIE_WXAI, TROSSEN_WXAI_BASE)
}


def get_embodiment(name: str) -> EmbodimentCfg:
  if name not in EMBODIMENTS:
    raise KeyError(f"Unknown embodiment '{name}'. Available: {sorted(EMBODIMENTS)}")
  return EMBODIMENTS[name]
