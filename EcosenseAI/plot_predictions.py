import matplotlib.pyplot as plt
import numpy as np

# Simulate smooth heart-rate-like decay with noise
def generate_component_data(length=30, start_val=100, end_val=70, noise_scale=1.5):
    x = np.linspace(0, 4 * np.pi, length)
    base_line = np.linspace(start_val, end_val, length)
    wave = 4 * np.sin(x)
    noise = np.random.normal(scale=noise_scale, size=length)
    return base_line + wave + noise

# Component definitions
components = {
    "Smoke Sensor Efficiency": "#00FF99",
    "Air Quality Efficiency": "#00BFFF",
    "Battery Health Efficiency": "#FF33CC",
    "Engine Health Efficiency": "#FFA500"
}

# Dark theme
plt.style.use('dark_background')

for title, color in components.items():
    y_data = generate_component_data()
    x_data = np.arange(len(y_data))
    latest_value = y_data[-1]

    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#111')
    ax.set_facecolor('#111')

    # Plot line with circles
    ax.plot(x_data, y_data, color=color, linewidth=2.5, marker='o', markersize=6,
            markerfacecolor='white', markeredgewidth=1.2)

    # Mark last point with a red X
    ax.plot(x_data[-1], latest_value, marker='x', color='red', markersize=10, mew=3,
            label=f"Current: {latest_value:.1f}%")

    ax.set_title(title, color='white', fontsize=14)
    ax.set_xlabel("Time", color='gray')
    ax.set_ylabel("Efficiency (%)", color='gray')
    ax.set_ylim(0, 120)
    ax.tick_params(colors='gray')
    ax.grid(True, linestyle='--', alpha=0.3)

    # Show current value as legend
    legend = ax.legend(loc="upper right", frameon=True, facecolor="#222", edgecolor="#444")
    for text in legend.get_texts():
        text.set_color('white')

    plt.show()
