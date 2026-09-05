import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons

# 1. Load and parse the dataset
# Replace 'your_file.csv' with the actual path to your CSV file
file_path = 'data/DATA_MEO_Test2.csv'
df = pd.read_csv(file_path)
df['utc_time'] = pd.to_datetime(df['utc_time'])
df = df.sort_values('utc_time')

# 2. Map display names to column properties
variables = {
    'X Error': ('x_error (m)', '#1f77b4', 'o', '-'),
    'Y Error': ('y_error (m)', '#ff7f0e', 's', '-'),
    'Z Error': ('z_error (m)', '#2ca02c', '^', '-'),
    'Sat Clock Error': ('satclockerror (m)', '#d62728', 'd', '--')
}

# 3. Initialize the visualization layout
# Adjusted subplots to make clean, dedicated room for the radio widget box
fig, ax = plt.subplots(figsize=(14, 7))
plt.subplots_adjust(left=0.25, bottom=0.15)  # Leave wide space on the left margin

# 4. Draw the baseline placeholder plot (defaulting to the first variable)
initial_label = 'X Error'
col, color, marker, style = variables[initial_label]
line, = ax.plot(
    df['utc_time'], df[col], 
    color=color, marker=marker, markersize=4, 
    linestyle=style, linewidth=1.5, label=f"{initial_label} (m)"
)

# 5. Format axes, labels, and baseline grids
ax.set_title('GPS Tracking System Error Metrics (Interactive View)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('UTC Timestamp', fontsize=11, labelpad=10)
ax.set_ylabel('Error Margin (meters)', fontsize=11, labelpad=10)
ax.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.5)
ax.grid(True, linestyle=':', alpha=0.6)
plt.xticks(rotation=45, ha='right')

# 6. Build the Interactive Radio Buttons widget frame inside the graph
# Positioning coordinate tracking box: [left, bottom, width, height]
radio_ax = plt.axes([0.03, 0.45, 0.15, 0.20], facecolor='#f8f9fa')
radio_buttons = RadioButtons(radio_ax, list(variables.keys()), active=0, activecolor='#2b2b2b')

# 7. Define the functional logic trigger for data switching
def update_plot(label):
    col_name, new_color, new_marker, new_style = variables[label]
    
    # Dynamically update structural line configurations
    line.set_ydata(df[col_name])
    line.set_color(new_color)
    line.set_marker(new_marker)
    line.set_linestyle(new_style)
    line.set_label(f"{label} (m)")
    
    # Recalculate spatial margins on the Y-axis to auto-fit data bounds cleanly
    ax.relim()
    ax.autoscale_view(scalex=False, scaley=True)
    
    # Redraw canvas instantly
    fig.canvas.draw_idle()

# Connect selection modifications to the core plotting algorithm execution loop
radio_buttons.on_clicked(update_plot)

# 8. Render the screen view
plt.show()
