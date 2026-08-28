import numpy as np
import matplotlib.pyplot as plt

import FolderActions
import CohesiveModel

def main():
    
    mm = 1e-3

    # Newest cases/*.toml; pass a path to read_case_parameters to pin one.
    case = FolderActions.read_case_parameters()
    E = case["E"]                                                      # Pa
    nu = case["nu"]                                                    # Poisson's ratio
    rho = case["rho"]                                                  # kg/m³

    # The case files carry rate-and-state parameters rather than a fracture
    # energy, so Gamma is reconstructed from the weakening step. case["tau_c"]
    # holds the peak strength if X_c is ever computed instead of prescribed.
    Gamma = CohesiveModel.rsf_fracture_energy(                         # J/m²
        case["sigma_n"], case["b"], case["dc"], case["V_dyn"], case["V_init"]
    )

    C_s = CohesiveModel.get_Cs(E, nu, rho)                             # Shear wave speed (m/s)
    C_d = CohesiveModel.get_Cd(E, nu, rho)                             # Longitudinal wave speed (m/s)
    X_c = 5 * mm                                                       # Cohesive zone size (m)

    C_f = 0.9 * C_s                                                    # Rupture speed (m/s)       [To be fit with experiment data]

    print(f"Case: {case['name']}  [{case['path'].name}, {case['zone']} zone]")
    print(f"Fracture energy (Gamma): {Gamma} J/m^2")
    print(f"Young's modulus (E): {E/1e9} GPa")
    print(f"Poisson's ratio (nu): {nu}")
    print(f"Density (rho): {rho} kg/m³")
    print(f"Rupture speed (C_f): {C_f:.1f} m/s")
    print(f"Shear wave speed (C_s): {C_s:.1f} m/s")
    print(f"Longitudinal wave speed (C_d): {C_d:.1f} m/s")
    print(f"Cohesive zone size (X_c): {X_c*1e3:.3g} mm")




    y_values = [1e-8, 0.1e-3, 0.5e-3, 1.0e-3, 2.0e-3, 5e-3, 10e-3, 15e-3]

    range = 300 # mm
    x = np.linspace(-range * 1e-3, range * 1e-3, 8192)

    # shift_scale = 1.0 # with sigma_c = 0.113
    shift_scale = 0.05 # with sigma_c = 0.513

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    for i, y in enumerate(y_values):
        delta_sigma_xx = CohesiveModel.delta_sigma_xx(x, y, X_c, C_f, C_s, C_d, nu, Gamma, E)
        axes[0].plot(x * 1000, delta_sigma_xx / 1e6 + i * shift_scale, '-.', label=f'y = {y * 1e3:.1f} mm')

    axes[0].set_xlabel('Rupture tip position x (mm)')
    axes[0].set_ylabel('Shear stress fluctuation $\\Delta \\sigma_{xx}$ (MPa)')
    axes[0].set_title('Normal stress fluctuation (xx)')
    axes[0].axvline(0, color='k', linestyle='--', linewidth=1)
    axes[0].legend(loc = "lower right")
    axes[0].grid()

    for i, y in enumerate(y_values):
        delta_sigma_xy = CohesiveModel.delta_sigma_xy(x, y, X_c, C_f, C_s, C_d, nu, Gamma, E)
        axes[1].plot(x * 1000, delta_sigma_xy / 1e6 + i * shift_scale, '-.', label=f'y = {y * 1e3:.1f} mm')

    axes[1].set_xlabel('Rupture tip position x (mm)')
    axes[1].set_ylabel('Shear stress fluctuation $\\Delta \\sigma_{xy}$ (MPa)')
    axes[1].set_title('Shear stress fluctuation (xy)')
    axes[1].axvline(0, color='k', linestyle='--', linewidth=1)
    axes[1].legend(loc = "lower right")
    axes[1].grid()

    plt.suptitle(f'Stress fluctuations along the fault | E = {E/1e9:.2f}GPa | ν = {nu} | $C_f$ = {C_f:.0f}m/s, $C_s$ = {C_s:.0f}m/s, $C_d$ = {C_d:.0f}m/s', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./CZM-Plot/example_xx_xy.png', dpi=900)
    plt.savefig('./CZM-Plot/example_xx_xy.pdf', dpi=900)
    plt.show()

    return 0

if __name__ == "__main__":
    main()
    FolderActions.delete_pycache()