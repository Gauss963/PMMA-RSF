import os
import shutil
from pathlib import Path

try:                      # Python 3.11+
    import tomllib
except ModuleNotFoundError:               # pragma: no cover
    import tomli as tomllib


CASES_DIRECTORY = Path(__file__).resolve().parent.parent / "cases"


def delete_pycache() -> None:
    current_directory = os.getcwd()
    pycache_directory = os.path.join(current_directory, '__pycache__')

    if os.path.exists(pycache_directory):
        shutil.rmtree(pycache_directory)

    return None


def latest_case_toml(directory=CASES_DIRECTORY) -> Path:
    """Return the most recently modified case TOML in `directory`."""
    directory = Path(directory)
    cases = sorted(directory.glob("*.toml"), key=lambda p: p.stat().st_mtime)
    if not cases:
        raise FileNotFoundError(f"No .toml case files under {directory}")
    return cases[-1]


def read_case_toml(path) -> dict:
    """Parse one PMMA-RSF case file and return it verbatim."""
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def read_case_parameters(path=None, zone="middle") -> dict:
    """Return the SI parameters of a case TOML, ready for CohesiveModel.

    The case files carry the legacy simulation units (mm, s, MPa,
    tonne/mm^3); everything below is converted to SI. `path` defaults to the
    newest case in `cases/`, and `zone` selects which rate-and-state block the
    a/b/dc triple is taken from.

    The cohesive strength follows the peak friction, tau_c = mu_s * sigma_n,
    which is what the retired .dat files stored as sigma_c * beta.
    """
    if path is None:
        path = latest_case_toml()
    path = Path(path)
    case = read_case_toml(path)

    material = case["material"]
    rsf = case["rsf"]
    rsf_zone = rsf[zone]

    sigma_n = material_stress = case["loading"]["normal_stress_reference"] * 1e6
    mu_s = rsf["initial_friction"]

    return {
        "path": path,
        "name": case.get("case", {}).get("name", path.stem),
        "zone": zone,
        # Bulk elasticity.
        "E": material["young_modulus"] * 1e6,          # MPa   -> Pa
        "nu": material["poisson_ratio"],               # dimensionless
        "rho": material["density"] * 1e12,             # t/mm^3 -> kg/m^3
        # Interface strength.
        "sigma_n": sigma_n,                            # MPa   -> Pa
        "mu_s": mu_s,                                  # dimensionless
        "tau_c": mu_s * material_stress,               # Pa
        # Rate and state.
        "a": rsf_zone["a"],                            # dimensionless
        "b": rsf_zone["b"],                            # dimensionless
        "dc": rsf_zone["dc"] * 1e-3,                   # mm    -> m
        "V_init": rsf["initial_steady_velocity"] * 1e-3,   # mm/s -> m/s
        "V_dyn": rsf["dynamic_calibration_velocity"] * 1e-3,
        "mesh_size": case["numerics"]["mesh_size"] * 1e-3,  # mm -> m
    }
