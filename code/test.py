import numpy as np
import matplotlib.pyplot as plt
import mpld3
from mpld3 import plugins

# Define the probability matrix and state names
P = np.array([[0.5, 0.5], [0.8, 0.2]])
states = ['H', 'T']

# Define the initial probability distribution
pi = np.array([0.6, 0.4])

# Define the labels for the nodes
labels = [f"{state}: {prob:.2f}" for state, prob in zip(states, pi)]

# Define the colors for the nodes
colors = ['blue', 'green']

# Define the positions of the nodes
pos = np.array([[1, 1], [2, 1]])

# Create the figure and axis object
fig, ax = plt.subplots(figsize=(8, 6))

# Draw the nodes
scatter = ax.scatter(pos[:, 0], pos[:, 1], s=500, c=colors, alpha=0.8)
labels = [ax.text(pos[i, 0], pos[i, 1]+0.1, label, ha='center', va='center', fontsize=16) for i, label in enumerate(labels)]

# Draw the edges
for i in range(len(states)):
    for j in range(len(states)):
        ax.arrow(pos[i, 0], pos[i, 1], pos[j, 0]-pos[i, 0], pos[j, 1]-pos[i, 1],
                 length_includes_head=True, width=0.02, head_width=0.1,
                 head_length=0.1, fc='gray', ec='gray', alpha=P[i, j])

# Set the axis properties and limits
ax.set_xlim([0, 3])
ax.set_ylim([0, 2])
ax.set_xticks([1, 2])
ax.set_xticklabels(states)
ax.set_yticks([])
ax.set_title("Coin flip probabilities", fontweight='bold', fontsize=20)

# Add the interactive plugin
plugins.connect(fig, plugins.MousePosition(fontsize=14))

# Show the interactive plot
mpld3.show()
