import CohesiveModel
import FolderActions

mat_file = "../Materials/material-mm-MPa.dat"
materials = FolderActions.read_materials(mat_file)

# print(materials)

Gamma = materials["interface"]["parameters"]["G_c"]      # Fracture energy (J/m^2)
E = materials["moving-block"]["parameters"]["E"]         # Young's modulus (MPa)
nu = materials["moving-block"]["parameters"]["nu"]       # Poisson's ratio
rho = materials["moving-block"]["parameters"]["rho"]     # Density (tonne/mm^3)


Vp = 2727
Vs = 1666

E, nu = CohesiveModel.compute_E_nu_from_VpVsRho(Vp, Vs, rho * 1e12)  # Convert density to kg/m^3

print(f"Fracture energy (Gamma): {Gamma} J/m²")
print(f"Young's modulus (E): {E/1e9:.3f} GPa")
print(f"Poisson's ratio (nu): {nu:.2f}")
print(f"Density (rho): {rho * 1e12} kg/m³")


FolderActions.delete_pycache()