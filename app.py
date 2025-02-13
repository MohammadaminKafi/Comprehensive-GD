import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ---------- Helper Functions ----------
def poly_from_coeffs(active_coeffs):
    """Return polynomial coefficients (as a NumPy array) from a list of coefficients (in descending order)."""
    return np.array(active_coeffs)

def poly_from_roots(active_roots):
    """Return polynomial coefficients with leading coefficient 1 from a list of roots."""
    return np.poly(active_roots)

def derivative_coeffs(poly_coeffs):
    """Compute the derivative coefficients for a polynomial given its coefficients (highest power first)."""
    n = len(poly_coeffs) - 1
    if n <= 0:
        return np.array([0])
    return np.array([poly_coeffs[i] * (n - i) for i in range(n)])

def gradient_descent_steps(x0, alpha, poly_coef, deriv_coef, steps=5):
    """Return an array of x-values representing the gradient descent steps."""
    pts = [x0]
    for _ in range(steps):
        grad = np.polyval(deriv_coef, pts[-1])
        pts.append(pts[-1] - alpha * grad)
    return np.array(pts)

def filter_real(roots, tol=1e-7):
    """Return only the real roots (within a tolerance) from an array of roots."""
    return np.array([r.real for r in roots if np.abs(r.imag) < tol])

# ---------- Default Settings ----------
# We are now using Roots mode by default.
default_degree = 4
# For Roots mode, we need a list of 6 roots; the first 4 will be active.
default_roots = [-30, 2, 2, 22, 0, 0]

# (These defaults for coefficients won't be used in default mode but are kept for completeness.)
default_coefs = [0, 0, 1, -14, -220, 1016, -1056]

default_alpha = 0.001
default_x0 = 5.5
default_steps = 5

# ---------- Streamlit Sidebar Widgets ----------
st.sidebar.title("Polynomial Settings")

# Select mode: Coefficients or Roots.
# Default is Roots (index=1 in the tuple).
mode = st.sidebar.radio("Select Mode", ("Coeffs", "Roots"), index=1)

# Degree slider (determines how many coefficients/roots are active)
degree = st.sidebar.slider("Degree", min_value=1, max_value=6, value=default_degree, step=1)

if mode == "Coeffs":
    st.sidebar.subheader("Coefficient Settings")
    # For a degree d polynomial, we need d+1 coefficients.
    active_coef_defaults = default_coefs[-(degree+1):]
    active_coefs = []
    for i, default_val in enumerate(active_coef_defaults):
        exponent = degree - i
        coef = st.sidebar.slider(f"Coefficient for x^{exponent}",
                                 min_value=-10.0,
                                 max_value=10.0,
                                 value=float(default_val),
                                 step=0.1)
        active_coefs.append(coef)
    poly_coef = poly_from_coeffs(active_coefs)
else:
    st.sidebar.subheader("Root Settings")
    # For a degree d polynomial, we have d roots.
    active_roots = []
    for i in range(degree):
        root_val = st.sidebar.slider(f"Root {i+1}",
                                     min_value=-30.0,
                                     max_value=30.0,
                                     value=float(default_roots[i]),
                                     step=0.1)
        active_roots.append(root_val)
    poly_coef = poly_from_roots(active_roots)

# ---------- Gradient Descent Settings ----------
st.sidebar.title("Gradient Descent Settings")
alpha = st.sidebar.slider("Learning Rate (α)",
                          min_value=0.0001,
                          max_value=0.01,
                          value=default_alpha,
                          step=0.0001,
                          format="%.5f")
x0 = st.sidebar.slider("Starting x",
                       min_value=-30.0,
                       max_value=30.0,
                       value=float(default_x0),
                       step=0.1)
steps = st.sidebar.slider("Number of Steps",
                          min_value=1,
                          max_value=20,
                          value=default_steps,
                          step=1)

# ---------- Compute Derived Quantities ----------
deriv_coef = derivative_coeffs(poly_coef)
x_vals = np.linspace(-30, 30, 800)
y_vals = np.polyval(poly_coef, x_vals)
yprime_vals = np.polyval(deriv_coef, x_vals)
gd_points = gradient_descent_steps(x0, alpha, poly_coef, deriv_coef, steps=steps)
gd_y = np.polyval(poly_coef, gd_points)
try:
    roots_all = np.roots(poly_coef)
    real_roots = filter_real(roots_all)
except Exception:
    real_roots = np.array([])

# Create a polynomial object for the expanded form
poly_expanded = np.poly1d(poly_coef)

# ---------- Display the Expanded Polynomial ----------
st.write("### Expanded Polynomial")
st.code(poly_expanded, language="python")

# ---------- Create the Plot (Bigger Figure) ----------
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x_vals, y_vals, color="blue", lw=2, label="f(x)")
ax.plot(x_vals, yprime_vals, "--", color="green", lw=2, label="f'(x)")
if real_roots.size > 0:
    ax.scatter(real_roots, np.zeros_like(real_roots), color="red", s=60, zorder=5, label="Real Roots")
ax.plot(gd_points, gd_y, marker="o", color="orange", lw=2, label="Gradient Descent")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("Interactive Polynomial, Its Derivative, and Gradient Descent")
ax.legend()
ax.grid(True)

# ---------- Display the Plot in Streamlit ----------
st.pyplot(fig)
