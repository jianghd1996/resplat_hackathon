"""CPU microprobes for hidden state changes in the current FastGS trainer.

Use the fastgs environment. No dataset, render or training checkpoint is changed.
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baseline/fastgs"))
from lpipsPyTorch.modules.lpips import LPIPS
from scene.gaussian_model import GaussianModel


def main():
    torch.manual_seed(0)
    before = torch.get_rng_state().clone()
    model = LPIPS(net_type="vgg")
    after = torch.get_rng_state().clone()
    actual = torch.multinomial(torch.ones(1000), 100, replacement=False)
    torch.set_rng_state(before)
    control = torch.multinomial(torch.ones(1000), 100, replacement=False)
    rng = {"lpips_initialization_changes_cpu_rng": not torch.equal(before, after),
           "subsequent_prune_sample_equal": torch.equal(actual, control),
           "sample_overlap_fraction": len(set(actual.tolist()) & set(control.tolist()))/100}
    del model

    g = GaussianModel(3)
    parameter = torch.nn.Parameter(torch.tensor([-1.0, -2.0]))
    g.optimizer = torch.optim.Adam([{"params": [parameter], "name": "opacity"}], lr=.01)
    parameter.square().sum().backward()
    g.optimizer.step()
    old_value = parameter.detach().clone()
    old_moment = g.optimizer.state[parameter]["exp_avg"].clone()
    # All values are below the 0.8 clamp. Match the actual implementation.
    clamped = torch.logit(torch.minimum(parameter.sigmoid(), torch.full_like(parameter, .8)))
    replacement = g.replace_tensor_to_optimizer(clamped, "opacity")["opacity"]
    state = g.optimizer.state[replacement]
    opacity = {"unchanged_values": torch.allclose(old_value, replacement),
               "old_moment_abs_sum": old_moment.abs().sum().item(),
               "new_moment_abs_sum": state["exp_avg"].abs().sum().item(),
               "new_second_moment_abs_sum": state["exp_avg_sq"].abs().sum().item()}
    result = {"rng": rng, "opacity_optimizer_state": opacity,
              "long_axis_split_analytic_transverse_variance_ratio": 1/(.8*2)**2,
              "densification_events_7000": len(range(600, 7001, 100)),
              "score_renders_7000": len(range(600, 7001, 100))*10*2,
              "note": "event/render counts from current loop; counts are not measured time"}
    out = ROOT / "results/rethink_v0/training_mechanics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
