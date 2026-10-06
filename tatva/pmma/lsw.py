"""Self-contained LSW cases using the shared PMMA solver (no RSF calibration)."""

from dataclasses import asdict
from pathlib import Path
import tomllib

from .dynamics import RunConfig
from .model import BlockSpec, FrictionReference, LoadingReference, Material, PMMAModelInput


def load_lsw_case(path):
    with Path(path).open("rb") as stream:
        data = tomllib.load(stream)
    cfg = RunConfig(**data["solver"])
    if cfg.friction_law != "slip-weakening":
        raise ValueError("The LSW runner requires friction_law='slip-weakening'.")
    material = data["material"]
    materials = {
        name: Material(name, material["density"], material["young_modulus"], material["poisson_ratio"])
        for name in ("moving-block", "stationary-block")
    }
    blocks = [
        BlockSpec(name + "-block", tuple(data["geometry"][name]["origin"]),
                  tuple(data["geometry"][name]["dimensions"]), tag)
        for name, tag in (("moving", 1), ("stationary", 2))
    ]
    case = PMMAModelInput(
        *blocks, materials, FrictionReference(**data["friction"]),
        LoadingReference(0.02, 0.02, 16.0, 0.3, 0.99, 0,
                         "stationary-block-back", "moving-block-front"),
    )
    return case, cfg, data["output"], data["case"]


def mesh_estimate(case, cfg, output, model):
    nodes = model["moving"].n_nodes + model["stationary"].n_nodes
    elements = sum(len(model[name].mesh.elements) for name in ("moving", "stationary"))
    frames = output["frames_per_phase"] + output["shear_frames_per_phase"]
    frames += int(output.get("include_initial_frame", True))
    node_values = 2 * (1 + output.get("store_bulk_velocity", True))
    element_values = 4 * (1 + output.get("store_bulk_strain", True))
    return {
        "nodes": nodes, "dofs": nodes * 2, "elements": elements,
        "element_type": cfg.element_type, "dt_s": float(model["dt"]),
        "normal_steps": model["pressure_steps"], "shear_steps": model["shear_steps"],
        "frames": frames,
        "bulk_uncompressed_tb": frames * 4 * (nodes * node_values + elements * element_values) / 1e12,
        "normal_mode": cfg.normal_loading_mode,
        "normal_stress_mpa": model["normal_stress"],
        "friction": asdict(case.friction),
    }
