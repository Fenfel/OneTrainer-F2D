"""LayerOffloadConductor.materialize() must be idempotent: the trainer keeps the conductor
active between steps and samplers call materialize_only() before every sample. A second
materialize on an active conductor must not re-place layers or re-clone offloaded weights
(that clone storm was the +16..23 GiB sampling peak)."""
import torch
from torch import nn

from modules.util.LayerOffloadConductor import LayerOffloadConductor
from modules.util.config.TrainConfig import TrainModelPartConfig, TrainConfig


def _make_conductor():
    config = TrainConfig.default_values()
    config.train_device = "cpu"
    config.temp_device = "cpu"
    config.async_offloading = False
    part = TrainModelPartConfig.default_values()
    part.offload_fraction = 1.0
    part.activation_offloading = False
    model = nn.Sequential(*[nn.Linear(128, 128) for _ in range(8)])
    conductor = LayerOffloadConductor(model, config, part)
    for layer in model:
        conductor.add_layer(layer)
    return conductor


class TestConductorMaterializeIdempotent:
    def test_second_materialize_is_noop(self):
        conductor = _make_conductor()
        try:
            conductor.materialize()

            # storage addresses, not tensor object identity: .data returns a fresh Python
            # object per access; a re-placed weight lives in a freshly allocated cache buffer
            ptrs_before = [p.data.data_ptr() for p in conductor._LayerOffloadConductor__module.parameters()]
            conductor.materialize()  # must not re-clone/re-place
            ptrs_after = [p.data.data_ptr() for p in conductor._LayerOffloadConductor__module.parameters()]

            assert ptrs_before == ptrs_after, (
                "materialize() on an active conductor re-placed the weight tensors"
            )
        finally:
            conductor.release()

    def test_materialize_after_evict_runs_again(self):
        conductor = _make_conductor()
        try:
            conductor.materialize()
            conductor.evict()
            conductor.materialize()  # evict cleared the active flag -> full pass must run
            assert conductor._LayerOffloadConductor__is_active
        finally:
            conductor.release()
