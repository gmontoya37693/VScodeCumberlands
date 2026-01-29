"""
Programming Exercise 1: Optimizing an approximation line
2026 Spring - Deep Learning (MSDS-534-M40) - Full Term
Professor: Dr. Intisar Rizwan I Haque
January 28, 2026
Germán Montoya

Objective:
Implement a gradient-based optimization algorithm to minimize the function:
f(x) = 0.5 * ||A * x - b||^2
where A, x, and b are vectors. This is a least squares optimization problem
commonly used in machine learning for finding optimal parameters.
"""

# MATHEMATICAL DERIVATION
"""
GRADIENT DERIVATION FOR f(x) = 0.5 * ||A * x - b||^2

To apply gradient descent, we need to find ∇f(x), the gradient of f with respect to x.

Step 1: Expand the squared norm
||A * x - b||^2 = (A * x - b)^T * (A * x - b)

Therefore:
f(x) = 0.5 * (A * x - b)^T * (A * x - b)

Step 2: Expand the matrix multiplication
f(x) = 0.5 * (x^T * A^T - b^T) * (A * x - b)
f(x) = 0.5 * (x^T * A^T * A * x - x^T * A^T * b - b^T * A * x + b^T * b)

Since x^T * A^T * b = b^T * A * x (both are scalars), we get:
f(x) = 0.5 * (x^T * A^T * A * x - 2 * b^T * A * x + b^T * b)

Step 3: Take the gradient with respect to x
Using matrix calculus rules:
- ∇(x^T * A^T * A * x) = 2 * A^T * A * x
- ∇(b^T * A * x) = A^T * b
- ∇(b^T * b) = 0 (constant)

Therefore:
∇f(x) = 0.5 * (2 * A^T * A * x - 2 * A^T * b)
∇f(x) = A^T * A * x - A^T * b
∇f(x) = A^T * (A * x - b)

This is gradient calculation: gradient = A^T * (A * x - b)
where (A * x - b) is the residual between prediction and target.
"""

# PSEUDOCODE FOR GRADIENT-BASED OPTIMIZATION
"""
ALGORITHM: Gradient Descent for Least Squares Optimization

Step 1: Initialize parameters
    - x ← random initial vector (or zeros)          # Starting point for optimization
    - α ← learning_rate (e.g., 0.01)                # Step size for each update
    - ε ← tolerance (e.g., 1e-6)                     # Convergence threshold
    - max_iter ← maximum iterations (e.g., 1000)    # Prevent infinite loops

Step 2: Define the problem
    - A ← coefficient matrix                         # Given matrix A
    - b ← target vector                              # Given vector b

Step 3: Main optimization loop
    FOR iteration = 1 to max_iter:
        
        Step 3a: Compute prediction
        y_pred ← A * x                               # Current prediction using x
        
        Step 3b: Compute residual
        residual ← y_pred - b                        # Difference between prediction and target
        
        Step 3c: Compute gradient
        gradient ← A^T * residual                    # Derivative of f(x) w.r.t. x
        
        Step 3d: Update parameters
        x ← x - α * gradient                         # Move in opposite direction of gradient
        
        Step 3e: Check convergence
        IF ||gradient|| < ε:                         # If gradient is small enough
            BREAK                                    # Stop optimization (converged)
        
Step 4: Return optimized x
    RETURN x                                         # Final optimized parameters

EXPLANATION OF EACH STEP:

Step 1: Initialize parameters
    - x: Starting guess for the optimal solution
    - α: Controls how big steps we take (too large = overshooting, too small = slow)
    - ε: When gradient becomes this small, we consider the solution "good enough"
    - max_iter: Safety mechanism to prevent infinite loops

Step 2: Define the problem
    - A and b define our least squares problem Ax ≈ b
    - We want to find x that makes Ax as close as possible to b

Step 3a: Compute prediction
    - Calculate what our current x predicts (Ax)
    - This is our model's current output

Step 3b: Compute residual
    - Find the error between prediction and actual target
    - This tells us how far off we are

Step 3c: Compute gradient
    - Calculate A^T * residual, which is ∇f(x) = A^T(Ax - b)
    - This tells us which direction increases the error most
    - We'll move in the opposite direction to decrease error

Step 3d: Update parameters
    - Move x in the direction that reduces the error
    - Learning rate α controls the step size

Step 3e: Check convergence
    - If gradient is very small, we're at/near the minimum
    - No point continuing if we're not improving significantly
"""

# BRIEF CALCULATION EXAMPLE - ACTUAL IMPLEMENTATION
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent_example():
    """
    Actual implementation of the gradient descent calculation example
    """
    print("=" * 60)
    print("GRADIENT DESCENT - NUMERICAL EXAMPLE")
    print("=" * 60)
    
    # Problem Setup: Fit line y = intercept + slope*x to data points
    print("\nProblem Setup:")
    print("Fit a line y = intercept + slope*x to data points")
    
    # Given Data
    A = np.array([[1, 1],     # Point 1: (x=1, design matrix row)
                  [1, 2],     # Point 2: (x=2, design matrix row)  
                  [1, 3]])    # Point 3: (x=3, design matrix row)
    
    b = np.array([2, 4, 5])   # Corresponding y-values: (1,2), (2,4), (3,5)
    
    print(f"\nMatrix A:\n{A}")
    print(f"Vector b: {b}")
    print(f"Goal: Find x = [intercept, slope] that minimizes ||Ax - b||^2")
    
    # Initialize parameters
    x = np.array([0.0, 0.0])  # Start with intercept=0, slope=0
    alpha = 0.1               # Learning rate
    epsilon = 1e-6            # Tolerance
    max_iter = 50             # Maximum iterations
    
    print(f"\nInitialization:")
    print(f"x⁰ = {x}")
    print(f"Learning rate α = {alpha}")
    print(f"Tolerance ε = {epsilon}")
    
    # Store values for plotting
    x_history = [x.copy()]
    cost_history = []
    
    # Cost function
    def compute_cost(A, x, b):
        prediction = A @ x
        residual = prediction - b
        return 0.5 * np.sum(residual**2)
    
    print("\n" + "="*60)
    print("GRADIENT DESCENT ITERATIONS")
    print("="*60)
    
    # Gradient descent iterations
    for iteration in range(max_iter):
        # Compute prediction
        prediction = A @ x
        
        # Compute residual
        residual = prediction - b
        
        # Compute gradient
        gradient = A.T @ residual
        
        # Compute cost
        cost = compute_cost(A, x, b)
        cost_history.append(cost)
        
        # Print iteration details
        if iteration < 5 or iteration % 10 == 0:
            print(f"\nIteration {iteration}:")
            print(f"  x = [{x[0]:.4f}, {x[1]:.4f}]")
            print(f"  prediction = {prediction}")
            print(f"  residual = {residual}")
            print(f"  gradient = {gradient}")
            print(f"  cost = {cost:.6f}")
        
        # Check convergence
        if np.linalg.norm(gradient) < epsilon:
            print(f"\nConverged at iteration {iteration}!")
            break
            
        # Update parameters
        x = x - alpha * gradient
        x_history.append(x.copy())
    
    print(f"\nFinal Results:")
    print(f"Optimal x* = [{x[0]:.4f}, {x[1]:.4f}]")
    print(f"This represents the line: y = {x[0]:.4f} + {x[1]:.4f}*x")
    print(f"Final cost: {cost_history[-1]:.8f}")
    print(f"Cost reduction: {cost_history[0]:.2f} → {cost_history[-1]:.8f}")
    
    return A, b, x_history, cost_history

def create_visualization():
    """
    Create comprehensive visualization of the gradient descent process
    """
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # Run the gradient descent
    A, b, x_history, cost_history = gradient_descent_example()
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # === LEFT PLOT: Line Fitting Evolution ===
    ax1.set_title("Gradient Descent: Optimizing an Approximation Line", fontsize=14, fontweight='bold')
    
    # Original data points
    x_data = np.array([1, 2, 3])  # x-coordinates from the design matrix
    y_data = b  # y-coordinates are the target values
    ax1.scatter(x_data, y_data, color='blue', s=100, zorder=5, label='Original Data Points')
    
    # Create x range for plotting lines
    x_line = np.linspace(0.5, 3.5, 100)
    
    # Plot evolution of lines
    colors = ['red', 'orange', 'gold', 'lightgreen', 'green']
    iterations_to_show = [0, 1, 5, 10, len(x_history)-1]
    
    for i, iter_idx in enumerate(iterations_to_show):
        if iter_idx < len(x_history):
            intercept, slope = x_history[iter_idx]
            y_line = intercept + slope * x_line
            
            if iter_idx == 0:
                ax1.plot(x_line, y_line, color=colors[i], linestyle='--', 
                        linewidth=2, label=f'Initial Guess (Iter {iter_idx})', alpha=0.8)
            elif iter_idx == len(x_history)-1:
                ax1.plot(x_line, y_line, color=colors[i], linestyle='-', 
                        linewidth=3, label=f'Final Solution (Iter {iter_idx})', alpha=0.9)
            else:
                ax1.plot(x_line, y_line, color=colors[i], linestyle=':', 
                        linewidth=1.5, label=f'Iteration {iter_idx}', alpha=0.7)
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.5, 3.5)
    ax1.set_ylim(-1, 6)
    
    # === RIGHT PLOT: Cost Function Convergence ===
    ax2.set_title("Cost Function Convergence", fontsize=14, fontweight='bold')
    ax2.plot(cost_history, 'b-', linewidth=2, marker='o', markersize=4)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Cost f(x) = 0.5 * ||Ax - b||²')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # Log scale to better show convergence
    
    # Add cost reduction annotation
    initial_cost = cost_history[0]
    final_cost = cost_history[-1]
    ax2.annotate(f'Initial: {initial_cost:.2f}', 
                xy=(0, initial_cost), xytext=(5, initial_cost*0.5),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
    ax2.annotate(f'Final: {final_cost:.6f}', 
                xy=(len(cost_history)-1, final_cost), 
                xytext=(len(cost_history)*0.6, final_cost*10),
                arrowprops=dict(arrowstyle='->', color='green', alpha=0.7))
    
    plt.tight_layout()
    plt.show()
    
    print("\nVisualization complete!")
    print("Left plot: Shows how the line progressively fits the data better")
    print("Right plot: Shows the cost function decreasing (converging to optimum)")

if __name__ == "__main__":
    # Run the actual calculation example with visualization
    create_visualization()

# LINE FITTING VISUALIZATION STRATEGY
"""
VISUALIZATION APPROACH: From Data Exploration to Optimal Fit

Step 1: Initial Data Scatter Plot
- Plot original data points (x=1,2,3 vs y=2,4,5) as blue dots
- Title: "Original Data Points to Fit"
- This shows the linear relationship we're trying to capture

Step 2: Initial Line (Before Optimization)
- Plot starting line using x⁰ = [0, 0]: y = 0 + 0*x (horizontal line at y=0)
- Color: Red dashed line
- Label: "Initial Guess (Iteration 0)"
- Shows poor fit to data

Step 3: Intermediate Lines (During Optimization)
- Plot lines from several iterations (e.g., iterations 1, 5, 10)
- Colors: Orange to yellow gradient
- Labels: "Iteration 1", "Iteration 5", etc.
- Shows progressive improvement

Step 4: Final Optimized Line
- Plot final converged line: y = 0.67 + 1.5x
- Color: Green solid line  
- Label: "Optimized Line (Final)"
- Shows excellent fit to data points

Step 5: Cost Function Evolution (subplot)
- Plot f(x) vs iteration number
- Shows rapid decrease from 22.5 to near 0
- Demonstrates convergence

Visual Impact:
- Single plot with all elements tells the complete story
- Clear progression from poor fit to optimal fit
- Quantitative validation through cost reduction
- Intuitive understanding of what gradient descent accomplished

This visualization directly demonstrates "optimizing an approximation line"!
"""
