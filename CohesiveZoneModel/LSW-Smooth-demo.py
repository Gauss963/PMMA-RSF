import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def gelu(x):
    return x * norm.cdf(x)

def LSW_mu(delta: float) -> float:

    mu_s: float = 0.8
    mu_k: float = 0.6
    D_c: float = 5.5

    delta_eff = np.minimum(delta, D_c)

    return delta_eff * (mu_k - mu_s) / D_c + mu_s


def LSW_mu_smooth(delta: float) -> float:

    mu_s: float = 0.8
    mu_k: float = 0.6
    D_c: float = 5.5
    eps: float = 0.01 * D_c

    x = (delta - D_c) / eps
    delta_eff = delta - eps * gelu(x)
    return mu_s + (mu_k - mu_s) * delta_eff / D_c


x = np.linspace(0, 10, 8192)
y = LSW_mu(x)

y_smooth = LSW_mu_smooth(x)

plt.plot(x, y, label="LSW")
plt.plot(x, y_smooth, label="LSW (Smooth)", linestyle="--")

plt.xlabel("Slip (mm)")
plt.ylabel("Friction Coefficient")
plt.title("Linear Slip-Weakening Friction Law")
plt.grid()
plt.legend()
plt.show()