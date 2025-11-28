import sqlite3
import os
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import math
import numpy as np
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
import textwrap
import datetime as dt

class ForecasterAnalysis:
    def __init__(self, db_path: str = 'database.db'):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_path = os.path.join(self.base_dir, db_path)

        # Update figure aesthetics for high-end consultancy
        plt.rcParams.update({
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        })

    # Database connection and data retrieval
    def fetch_forecast_data(self):
        """Fetches all forecast data from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query("SELECT * FROM ensembles", conn)
            df['date_time'] = pd.to_datetime(df['date_time'])
            df = df.loc[df['expected_value'] > 0.01]
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def plot_charts(self, num_charts: int = 12, save_path: str = 'forecaster_plots.png'):
        """
        Plots expected values for the top queries with a polished, client-ready style.
        Generates a grid of subplots with consistent theming, clear annotations, and exports a high-resolution image.
        """
        data = self.fetch_forecast_data()
        if data is None or data.empty:
            print("No data available for plotting.")
            return

        # Calc max entropy
        max_entropy = np.log2(101)
        # Colormap for entropy dots (0 → green, 1 → red)
        cmap_entropy = plt.get_cmap('RdYlGn_r')

        # Select top queries
        query_counts = data['query'].value_counts()
        selected_queries = query_counts.index[:num_charts]

        # --- Define a landscape‑oriented grid ---
        rows = 3
        cols = math.ceil(len(selected_queries) / rows)
        fig_width = cols * 5      # 5 inches per subplot 
        fig_height = rows * 3.2   # slightly taller
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(fig_width, fig_height),
            sharex=True,
            sharey=True,
        )
        fig.suptitle("Forecaster Situational Awareness Dashboard", fontsize=20, fontweight='bold', y=0.92)

        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.flatten()

        # Predefine background band ranges and colormap
        ranges = [(0, 5), (10, 20), (25, 35), (40, 50), (55, 75), (80, 90), (95, 100)]
        cmap = plt.get_cmap('Greens', len(ranges))
        bg_colors = cmap(np.arange(len(ranges)))

        for i, query in enumerate(selected_queries):
            ax = axes[i]
            df_q = data[data['query'] == query].set_index('date_time').sort_index()

            # Plot background bands
            for idx, (low, high) in enumerate(ranges):
                ax.axhspan(low, high, color=bg_colors[idx], alpha=0.05, zorder=0)

            # Plot expected values
            ax.plot(df_q.index, df_q['expected_value'], label='Forecast', linewidth=2.5, zorder=2)

            if 'entropy' in df_q.columns and len(df_q) > 1:
                # Normalised entropy in [0, 1] → colour map (green → low, red → high)
                ent_frac = (df_q['entropy'] / max_entropy).clip(0, 1).values

                # Build coloured segments along y = 0
                x_vals = mdates.date2num(df_q.index.to_pydatetime())
                y_zero = np.zeros_like(ent_frac)
                points = np.array([x_vals, y_zero]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                colours = cmap_entropy(ent_frac[:-1])

                lc = LineCollection(
                    segments,
                    colors=colours,
                    linewidth=10,
                    alpha=1,
                    zorder=1,
                )
                ax.add_collection(lc)

            # --- Annotate latest point with directional arrow ---
            if not df_q.empty:
                latest_val = df_q['expected_value'].iloc[-1]
                latest_time = df_q.index[-1]

                # Determine arrow direction using previous point (if exists) and entropy trend
                arrow = "→"  # default flat arrow
                if len(df_q) >= 2 and 'entropy' in df_q.columns:
                    prev_val = df_q['expected_value'].iloc[-2]
                    prev_ent = df_q['entropy'].iloc[-2]
                    latest_ent = df_q['entropy'].iloc[-1]

                    if latest_val > prev_val and latest_ent < prev_ent:
                        arrow = "↑"
                    elif latest_val < prev_val and latest_ent < prev_ent:
                        arrow = "↓"

                ax.scatter(latest_time, latest_val, color='red', s=35, zorder=5)
                ax.text(
                    latest_time,
                    latest_val + 5,
                    f"{arrow} {latest_val:.0f}%",
                    va='bottom',
                    ha='right',
                    fontsize=9,
                    fontweight='bold',
                    color='red',
                    zorder=6
                )

            # Format axes
            wrapped_title = textwrap.fill(query, width=35)
            ax.set_title(wrapped_title, fontweight='bold', fontsize=10, pad=8)
            # Show y‑axis label only for the first column
            if i % cols != 0:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)
            ax.set_ylim(0, 100)
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
            ax.grid(alpha=0.2)

            # Only bottom row shows x‑tick labels
            if i < len(selected_queries) - cols:
                ax.tick_params(labelbottom=False)

            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha("right")

        # Hide any unused axes
        for j in range(len(selected_queries), len(axes)):
            fig.delaxes(axes[j])

        fig.text(0.01, 0.47, "Expected Value (%)",
                 va='center', rotation='vertical',
                 fontsize=10, fontweight='bold')

        # Minimize whitespace around and between subplots
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.92], pad=0.3)
        # Add last updated timestamp in top right corner
        last_updated = dt.datetime.now().strftime('Last updated: %Y-%m-%d %H:%M')
        fig.text(0.98, 0.98, last_updated, ha='right', va='top', fontsize=8)

        # Save at high resolution
        output_path = os.path.join(self.base_dir, save_path)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Charts saved to {output_path}")