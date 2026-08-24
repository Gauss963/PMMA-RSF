import CohesiveModel
import FolderActions

# Example usage:
tau_c = 2.00     # MPa
G_IIc = 0.20     # N/mm
beta  = 0.0001     # chosen for Akantu

sigma_c, G_c, beta_out, gamma_c = CohesiveModel.convert_modeII_to_akantu(tau_c, G_IIc, beta)

print("Converted parameters for Akantu:")
print(f"  sigma_c  = {sigma_c:.0f} MPa")
print(f"  G_c      = {G_c:.4f} N/mm  (Akantu fracture energy)")
print(f"  beta     = {beta_out}")
print(f"  gamma_c  = {gamma_c:.6f} mm (for verification only)")

FolderActions.delete_pycache()