import matplotlib.pyplot as plt
import numpy as np

# Data points
x_data = [1, 2, 3, 4]
y_data = [0, -1, -2, -3]

# Create x values for drawing the line
x_line = np.linspace(0, 5, 100)

# Perfect model: W=-1, b=1
W_perfect = -1
b_perfect = 1
y_perfect = W_perfect * x_line + b_perfect

# Current model: W=0.3, b=-0.3
W_current = 0.3
b_current = -0.3
y_current = W_current * x_line + b_current

# Calculate predictions for data points
y_pred_perfect = [W_perfect * x + b_perfect for x in x_data]
y_pred_current = [W_current * x + b_current for x in x_data]

# Calculate losses
loss_perfect = sum([(y_pred - y_actual)**2 for y_pred, y_actual in zip(y_pred_perfect, y_data)])
loss_current = sum([(y_pred - y_actual)**2 for y_pred, y_actual in zip(y_pred_current, y_data)])

# Create the plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Perfect model (W=-1, b=1)
ax1.plot(x_line, y_perfect, 'g-', linewidth=2, label=f'y = {W_perfect}*x + {b_perfect}')
ax1.scatter(x_data, y_data, color='red', s=100, zorder=5, label='Data points')
ax1.set_xlabel('x (input)', fontsize=12)
ax1.set_ylabel('y (output)', fontsize=12)
ax1.set_title(f'Perfect Model: Loss = {loss_perfect:.2f}', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_xlim(0, 5)
ax1.set_ylim(-4, 2)

# Add annotations showing the line passes through all points
for x, y in zip(x_data, y_data):
    ax1.plot([x, x], [y, y], 'go', markersize=8, zorder=6)
    ax1.annotate(f'({x}, {y})', xy=(x, y), xytext=(5, 5), 
                textcoords='offset points', fontsize=10)

# Plot 2: Current model (W=0.3, b=-0.3)
ax2.plot(x_line, y_current, 'b-', linewidth=2, label=f'y = {W_current}*x + {b_current}')
ax2.scatter(x_data, y_data, color='red', s=100, zorder=5, label='Data points')
ax2.set_xlabel('x (input)', fontsize=12)
ax2.set_ylabel('y (output)', fontsize=12)
ax2.set_title(f'Current Model: Loss = {loss_current:.2f}', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_xlim(0, 5)
ax2.set_ylim(-4, 2)

# Draw vertical lines showing the errors (distances from points to line)
for x, y_actual, y_pred in zip(x_data, y_data, y_pred_current):
    ax2.plot([x, x], [y_actual, y_pred], 'r--', linewidth=1.5, alpha=0.7)
    error = y_pred - y_actual
    ax2.annotate(f'error={error:.2f}', xy=(x, (y_actual + y_pred)/2), 
                xytext=(10, 0), textcoords='offset points', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

plt.tight_layout()
plt.savefig('loss_visualization.png', dpi=150, bbox_inches='tight')
print(f"Visualization saved as 'loss_visualization.png'")
print(f"\nPerfect model (W={W_perfect}, b={b_perfect}): Loss = {loss_perfect}")
print(f"Current model (W={W_current}, b={b_current}): Loss = {loss_current}")
plt.show()
