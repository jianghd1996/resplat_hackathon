"""Module B continuation-training backends."""

BACKENDS = {"fastgs": "fastgs.py", "short_budget": "short_budget.py",
            "short_schedule": "short_schedule.py", "regularized": "regularized.py",
            "no_reset": "no_reset.py", "topk_growth": "topk_growth.py",
            "coarse_to_fine": "coarse_to_fine.py", "long_axis_split": "long_axis_split.py",
            "perceptual_loss": "perceptual_loss.py",
            "terminal_guard": "terminal_guard.py",
            "topology_freeze": "topology_freeze.py",
            "opacity_state": "opacity_state.py",
            "topology_opacity": "topology_opacity.py",
            "surface_split": "surface_split.py",
            "surface_topology": "surface_topology.py"}
