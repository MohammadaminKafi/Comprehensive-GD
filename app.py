import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Set page configuration to wide layout.
st.set_page_config(layout="wide", page_title="Interactive Polynomial Explorer")

# ---------- Custom CSS for Wider Sliders in Sidebar ----------
st.markdown(
    """
    <style>
    /* Increase slider width in the sidebar */
    div[data-baseweb="slider"] {
        width: 400px;
    }
    /* Increase the width of text input boxes in the sidebar */
    div[data-baseweb="input"] > div {
        width: 100px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Helper Functions ----------

def poly_from_coeffs(active_coeffs):
    """Return polynomial coefficients (as a NumPy array) from a list (in descending order)."""
    return np.array(active_coeffs)

def poly_from_roots(active_roots):
    """Return polynomial coefficients with leading coefficient 1 from a list of roots."""
    return np.poly(active_roots)

def derivative_coeffs(poly_coeffs):
    """Compute derivative coefficients for a polynomial given its coefficients (highest power first)."""
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
    """Return only the real roots from an array of roots."""
    return np.array([r.real for r in roots if np.abs(r.imag) < tol])

def poly_to_latex(coeffs):
    """
    Convert a list of polynomial coefficients (in descending order) into a LaTeX string.
    For example, [1, 4, -688, 2672, -2640] becomes:
    f(x) = x^4 + 4x^3 - 688x^2 + 2672x - 2640
    """
    terms = []
    degree = len(coeffs) - 1
    for i, a in enumerate(coeffs):
        exp = degree - i
        if np.isclose(a, 0):
            continue
        # Determine sign.
        if a < 0:
            sign_str = " - " if terms else "-"
        else:
            sign_str = " + " if terms else ""
        abs_a = abs(a)
        if exp == 0:
            term_str = f"{abs_a:g}"
        elif exp == 1:
            term_str = "x" if np.isclose(abs_a, 1) else f"{abs_a:g}x"
        else:
            term_str = f"x^{{{exp}}}" if np.isclose(abs_a, 1) else f"{abs_a:g}x^{{{exp}}}"
        terms.append(sign_str + term_str)
    if not terms:
        return "0"
    return "".join(terms)

# ---------- Default Settings ----------
# Underlying polynomial: f(x) = (x+30) * (x-2)^2 * (x-22)
# Expanded: x^4 + 4x^3 - 688x^2 + 2672x - 2640.
default_degree = 4
default_coefs = [0, 0, 1, 4, -688, 2672, -2640]   # When degree=4, active coefficients are the last 5.
default_roots = [-30, 2, 2, 22, 0, 0]              # In "Roots" mode, first 4 are active.
default_alpha = 0.001
default_x0 = 5.5
default_steps = 5

# ---------- Sidebar Widgets ----------

st.sidebar.title("Polynomial Settings")

# Mode selection: "Coeffs" or "Roots" (default to Roots).
mode = st.sidebar.radio("Select Mode", ("Coeffs", "Roots"), index=1)

# Degree slider.
degree = st.sidebar.slider("Degree", min_value=1, max_value=6, value=default_degree, step=1)

# --- Coefficient / Root Input Section ---
if mode == "Coeffs":
    st.sidebar.subheader("Coefficient Settings")
    active_coef_defaults = default_coefs[-(degree+1):]
    active_coefs = []
    for i, default_val in enumerate(active_coef_defaults):
        exp = degree - i
        # Create slider and text input one after the other.
        slider_val = st.sidebar.slider(
            f"Coefficient for x^{exp}",
            min_value=-2000.0,
            max_value=2000.0,
            value=float(default_val),
            step=0.1,
            key=f"coef_slider_{degree}_{i}"
        )
        text_val = st.sidebar.text_input(
            f"Enter value for Coefficient x^{exp}",
            value=str(slider_val),
            key=f"coef_text_{degree}_{i}"
        )
        try:
            final_val = float(text_val)
        except ValueError:
            final_val = slider_val
        active_coefs.append(final_val)
        st.sidebar.markdown("<br>", unsafe_allow_html=True)
    poly_coef = poly_from_coeffs(active_coefs)
else:
    st.sidebar.subheader("Root Settings")
    active_roots = []
    for i in range(degree):
        slider_val = st.sidebar.slider(
            f"Root {i+1}",
            min_value=-100.0,
            max_value=100.0,
            value=float(default_roots[i]),
            step=0.1,
            key=f"root_slider_{degree}_{i}"
        )
        text_val = st.sidebar.text_input(
            f"Enter value for Root {i+1}",
            value=str(slider_val),
            key=f"root_text_{degree}_{i}"
        )
        try:
            final_val = float(text_val)
        except ValueError:
            final_val = slider_val
        active_roots.append(final_val)
        st.sidebar.markdown("<br>", unsafe_allow_html=True)
    poly_coef = poly_from_roots(active_roots)

st.sidebar.title("Gradient Descent Settings")
alpha = st.sidebar.slider(
    "Learning Rate (α)",
    min_value=0.0001,
    max_value=0.01,
    value=default_alpha,
    step=0.0001,
    format="%.5f"
)
x0 = st.sidebar.slider("Starting x", min_value=-30.0, max_value=30.0, value=float(default_x0), step=0.1)
steps = st.sidebar.slider("Number of Steps", min_value=1, max_value=20, value=default_steps, step=1)

# Zoom controls.
plot_x_range = st.sidebar.slider("Plot Range (x-axis)", min_value=10, max_value=2000, value=60, step=5)
plot_y_range = st.sidebar.slider("Plot Range (y-axis)", min_value=10, max_value=5000000, value=1000000, step=10)

# ---------- Compute Derived Quantities ----------
deriv_coef = derivative_coeffs(poly_coef)
x_vals = np.linspace(-plot_x_range, plot_x_range, 800)
y_vals = np.polyval(poly_coef, x_vals)
yprime_vals = np.polyval(deriv_coef, x_vals)
gd_points = gradient_descent_steps(x0, alpha, poly_coef, deriv_coef, steps=steps)
gd_y = np.polyval(poly_coef, gd_points)
try:
    roots_all = np.roots(poly_coef)
    real_roots = filter_real(roots_all)
except Exception:
    real_roots = np.array([])

# Create LaTeX string for expanded polynomial.
latex_poly = r"f(x) = " + poly_to_latex(poly_coef.tolist())

# ---------- Display Expanded Polynomial ----------
st.write("### Expanded Polynomial")
st.latex(latex_poly)

# ---------- Create the Plot (Larger Figure) ----------
fig, ax = plt.subplots(figsize=(14, 9))
ax.plot(x_vals, y_vals, color="blue", lw=2, label="f(x)")
ax.plot(x_vals, yprime_vals, "--", color="green", lw=2, label="f'(x)")
if real_roots.size > 0:
    ax.scatter(real_roots, np.zeros_like(real_roots), color="red", s=80, zorder=5, label="Real Roots")
ax.plot(gd_points, gd_y, marker="o", color="orange", lw=2, label="Gradient Descent")
ax.set_xlabel("x", fontsize=14)
ax.set_ylabel("f(x)", fontsize=14)
ax.set_title("Interactive Polynomial, Its Derivative, and Gradient Descent", fontsize=16)
ax.legend(fontsize=12)
ax.grid(True)
ax.set_xlim(-plot_x_range, plot_x_range)
ax.set_ylim(-plot_y_range, plot_y_range)
st.pyplot(fig)
