# Interactive Polynomial Explorer with Gradient Descent

This Streamlit app allows users to explore polynomial functions, their derivatives, and the gradient descent algorithm interactively. The application is designed to help students understand how gradient descent works by allowing them to modify polynomial equations and visualize the optimization process.

## Live Demo
The app is deployed and accessible at:  
🔗 [https://comprehensive-gd.streamlit.app](https://comprehensive-gd.streamlit.app)

## Features
- **Polynomial Customization:**
  - Define polynomials by **coefficients** or **roots**.
  - Adjust the **degree** of the polynomial (up to degree 6).
  - Modify coefficients (–2000 to 2000) and roots (–100 to 100) using **sliders** and **manual input fields**.
  
- **Gradient Descent Visualization:**
  - Set the **learning rate (α)** with **5 decimal places** precision.
  - Choose a **starting x-value**.
  - Adjust the **number of descent steps** dynamically.
  
- **Graphical Interactivity:**
  - Displays the function \( f(x) \) and its derivative \( f'(x) \).
  - Highlights **real roots** in red.
  - Plots **gradient descent steps** with connected points.
  - **Zoom controls** for x-axis and y-axis scaling.

- **Enhanced UI:**
  - **Wide layout** for better usability.
  - **Longer sliders** for precision adjustments.
  - **LaTeX-rendered polynomial expansion** for clarity.

## Purpose
This project was developed to educate students about **gradient descent** by providing an interactive way to experiment with different functions and learning rates.

## Installation
To run this project locally:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MohammadaminKafi/Comprehensive-GD.git
   cd comprehensive-gd
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
