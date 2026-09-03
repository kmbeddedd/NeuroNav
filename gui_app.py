from __future__ import annotations
import os
import sys
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import stats

from src.gui_engine import (
    load_dataset_file,
    compute_model_predictions,
    compare_and_evaluate,
    detect_series_type,
    TARGETS,
    TARGET_LABELS,
    TargetMetrics
)

# -----------------------------------------------------------------------------
# Aerospace & Mission-Control Design System (Palantir / Datadog Aesthetic)
# -----------------------------------------------------------------------------
AERO_THEME = {
    'bg_app': '#0B0E14',          # Deep Space Base Obsidian
    'bg_surface': '#141821',      # Elevated Panel / Card Surface
    'bg_surface_alt': '#181E2A',  # Secondary Surface / Alternating Row
    'bg_header': '#10141D',       # Persistent Top Navigation Bar
    'bg_input': '#10141D',        # Input Field & Dropdown Surface
    'border': '#252B38',          # Subtle 1px Technical Border
    'border_focus': '#0EA5E9',    # Active Border / Glow
    'fg_primary': '#E4E7EC',      # Crisp Off-White Body / Primary Text
    'fg_secondary': '#8B93A7',    # Muted Cool Gray Labels & Subtitles
    'fg_muted': '#4B5565',        # Deep Telemetry / Inactive Gray
    'accent': '#0EA5E9',          # Primary Aerospace Cyan
    'accent_hover': '#0284C7',    # Hover Accent State
    'accent_glow': '#38BDF8',     # Cyan Luminous Accent
    'accent_dark': '#0C4A6E',     # Deep Navy Accent Fill
    'success': '#22C55E',         # Emerald Pass
    'success_bg': '#052E16',      # Dark Emerald Pill Fill
    'error': '#EF4444',           # Coral Reject
    'error_bg': '#450A0A',        # Dark Coral Pill Fill
    'warning': '#F59E0B',         # Telemetry Warning Amber
    'table_header': '#1E2533',    # Elevated Sticky Header Row
    'table_even': '#141821',      # Zebra Striping Even
    'table_odd': '#181E2A',       # Zebra Striping Odd
    'table_select': '#0C4A6E',    # Table Row Active Select
}

FONT_UI = "Segoe UI"
FONT_MONO = "Consolas"

FONT_TITLE = (FONT_UI, 13, "bold")
FONT_HEADING = (FONT_UI, 11, "bold")
FONT_SUBHEADING = (FONT_UI, 10, "bold")
FONT_BODY = (FONT_UI, 10)
FONT_BODY_BOLD = (FONT_UI, 10, "bold")
FONT_SMALL = (FONT_UI, 9)
FONT_BADGE = (FONT_UI, 9, "bold")

FONT_TABLE_HEAD = (FONT_UI, 10, "bold")
FONT_TABLE_ROW = (FONT_MONO, 10)
FONT_TABLE_ROW_BOLD = (FONT_MONO, 10, "bold")


class AeroCard(tk.Canvas):
    """Elevated aerospace card container with 10px rounded corners and subtle 1px border."""

    def __init__(self, parent, bg_color=AERO_THEME['bg_surface'], border_color=AERO_THEME['border'],
                 border_width=1, radius=10, inner_pad=14, **kwargs):
        parent_bg = parent.cget('bg') if hasattr(parent, 'cget') and parent.cget('bg') else AERO_THEME['bg_app']
        super().__init__(parent, bg=parent_bg, highlightthickness=0, **kwargs)
        self.bg_color = bg_color
        self.border_color = border_color
        self.border_width = border_width
        self.radius = radius
        self.inner_pad = inner_pad
        self.inner_frame = tk.Frame(self, bg=bg_color)
        self.window_id = self.create_window(self.inner_pad, self.inner_pad, window=self.inner_frame, anchor='nw')
        self.bind("<Configure>", self._on_resize)

    def _on_resize(self, event):
        self.delete("card_shape")
        w, h = event.width, event.height
        if w < 10 or h < 10:
            return
        r = self.radius
        bw = self.border_width
        d = 2 * r
        x1, y1 = bw, bw
        x2, y2 = w - bw, h - bw

        if x2 > x1 + d and y2 > y1 + d:
            # 4 Arcs
            self.create_arc(x1, y1, x1 + d, y1 + d, start=90, extent=90, fill=self.bg_color, outline="", tags="card_shape")
            self.create_arc(x2 - d, y1, x2, y1 + d, start=0, extent=90, fill=self.bg_color, outline="", tags="card_shape")
            self.create_arc(x2 - d, y2 - d, x2, y2, start=270, extent=90, fill=self.bg_color, outline="", tags="card_shape")
            self.create_arc(x1, y2 - d, x1 + d, y2, start=180, extent=90, fill=self.bg_color, outline="", tags="card_shape")
            # Inner rects
            self.create_rectangle(x1 + r, y1, x2 - r, y2, fill=self.bg_color, outline="", tags="card_shape")
            self.create_rectangle(x1, y1 + r, x2, y2 - r, fill=self.bg_color, outline="", tags="card_shape")

            # 1px Technical Border
            if self.border_color and bw > 0:
                self.create_arc(x1, y1, x1 + d, y1 + d, start=90, extent=90, style="arc", outline=self.border_color, width=bw, tags="card_shape")
                self.create_arc(x2 - d, y1, x2, y1 + d, start=0, extent=90, style="arc", outline=self.border_color, width=bw, tags="card_shape")
                self.create_arc(x2 - d, y2 - d, x2, y2, start=270, extent=90, style="arc", outline=self.border_color, width=bw, tags="card_shape")
                self.create_arc(x1, y2 - d, x1 + d, y2, start=180, extent=90, style="arc", outline=self.border_color, width=bw, tags="card_shape")
                self.create_line(x1 + r, y1, x2 - r, y1, fill=self.border_color, width=bw, tags="card_shape")
                self.create_line(x2, y1 + r, x2, y2 - r, fill=self.border_color, width=bw, tags="card_shape")
                self.create_line(x1 + r, y2, x2 - r, y2, fill=self.border_color, width=bw, tags="card_shape")
                self.create_line(x1, y1 + r, x1, y2 - r, fill=self.border_color, width=bw, tags="card_shape")

        inner_w = max(1, w - 2 * self.inner_pad)
        inner_h = max(1, h - 2 * self.inner_pad)
        self.coords(self.window_id, self.inner_pad, self.inner_pad)
        self.itemconfigure(self.window_id, width=inner_w, height=inner_h)


class AeroButton(tk.Canvas):
    """High-contrast aerospace button with rounded edges, hover transitions, and status states."""

    def __init__(self, parent, text="", command=None, variant="primary", radius=8,
                 font=FONT_BODY_BOLD, height=36, width=None, **kwargs):
        parent_bg = parent.cget('bg') if hasattr(parent, 'cget') and parent.cget('bg') else AERO_THEME['bg_surface']
        super().__init__(parent, bg=parent_bg, highlightthickness=0, cursor="hand2", height=height, **kwargs)
        if width:
            self.configure(width=width)

        self.text = text
        self.command = command
        self.variant = variant
        self.radius = radius
        self.btn_font = font
        self.btn_state = "normal"

        if variant == "primary":
            self.bg_color = AERO_THEME['accent']
            self.hover_bg = AERO_THEME['accent_hover']
            self.fg_color = '#ffffff'
            self.border_color = AERO_THEME['accent_glow']
        elif variant == "secondary":
            self.bg_color = AERO_THEME['bg_surface_alt']
            self.hover_bg = AERO_THEME['border']
            self.fg_color = AERO_THEME['fg_primary']
            self.border_color = AERO_THEME['border']
        elif variant == "ghost":
            self.bg_color = 'transparent'
            self.hover_bg = AERO_THEME['bg_surface_alt']
            self.fg_color = AERO_THEME['fg_secondary']
            self.border_color = None
        else:
            self.bg_color = AERO_THEME['bg_surface_alt']
            self.hover_bg = AERO_THEME['border']
            self.fg_color = AERO_THEME['fg_primary']
            self.border_color = AERO_THEME['border']

        self.current_bg = self.bg_color
        self.bind("<Configure>", self._draw)
        self.bind("<Button-1>", self._on_click)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    def _draw(self, event=None):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 10 or h < 10:
            return
        r = min(self.radius, h // 2, w // 2)
        d = 2 * r
        x1, y1 = 1, 1
        x2, y2 = w - 1, h - 1

        bg_draw = self.current_bg
        if bg_draw != 'transparent':
            self.create_arc(x1, y1, x1 + d, y1 + d, start=90, extent=90, fill=bg_draw, outline="")
            self.create_arc(x2 - d, y1, x2, y1 + d, start=0, extent=90, fill=bg_draw, outline="")
            self.create_arc(x2 - d, y2 - d, x2, y2, start=270, extent=90, fill=bg_draw, outline="")
            self.create_arc(x1, y2 - d, x1 + d, y2, start=180, extent=90, fill=bg_draw, outline="")
            self.create_rectangle(x1 + r, y1, x2 - r, y2, fill=bg_draw, outline="")
            self.create_rectangle(x1, y1 + r, x2, y2 - r, fill=bg_draw, outline="")

        if self.border_color and bg_draw != 'transparent':
            self.create_arc(x1, y1, x1 + d, y1 + d, start=90, extent=90, style="arc", outline=self.border_color, width=1)
            self.create_arc(x2 - d, y1, x2, y1 + d, start=0, extent=90, style="arc", outline=self.border_color, width=1)
            self.create_arc(x2 - d, y2 - d, x2, y2, start=270, extent=90, style="arc", outline=self.border_color, width=1)
            self.create_arc(x1, y2 - d, x1 + d, y2, start=180, extent=90, style="arc", outline=self.border_color, width=1)
            self.create_line(x1 + r, y1, x2 - r, y1, fill=self.border_color, width=1)
            self.create_line(x2, y1 + r, x2, y2 - r, fill=self.border_color, width=1)
            self.create_line(x1 + r, y2, x2 - r, y2, fill=self.border_color, width=1)
            self.create_line(x1, y1 + r, x1, y2 - r, fill=self.border_color, width=1)

        fg = self.fg_color if self.btn_state != "disabled" else AERO_THEME['fg_muted']
        self.create_text(w // 2, h // 2, text=self.text, fill=fg, font=self.btn_font)

    def _on_enter(self, e):
        if self.btn_state != "disabled":
            self.current_bg = self.hover_bg
            self._draw()

    def _on_leave(self, e):
        if self.btn_state != "disabled":
            self.current_bg = self.bg_color
            self._draw()

    def _on_click(self, e):
        if self.btn_state != "disabled" and self.command:
            self.command()

    def config_state(self, state: str):
        self.btn_state = state
        if state == "disabled":
            self.current_bg = AERO_THEME['bg_surface_alt']
            self.configure(cursor="arrow")
        else:
            self.current_bg = self.bg_color
            self.configure(cursor="hand2")
        self._draw()

    def set_text(self, text: str):
        self.text = text
        self._draw()


class AeroSegmentedControl(tk.Canvas):
    """Pill-based segmented control replacing generic radio buttons with an aerospace feel."""

    def __init__(self, parent, options: List[str], variable: tk.StringVar,
                 on_change: Optional[Callable] = None, height=36, radius=8, **kwargs):
        parent_bg = parent.cget('bg') if hasattr(parent, 'cget') and parent.cget('bg') else AERO_THEME['bg_surface']
        super().__init__(parent, bg=parent_bg, highlightthickness=0, cursor="hand2", height=height, **kwargs)
        self.options = options
        self.variable = variable
        self.on_change = on_change
        self.radius = radius
        self.segments_bounds: List[tuple] = []

        self.bind("<Configure>", self._draw)
        self.bind("<Button-1>", self._on_click)

    def _draw(self, event=None):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 10 or h < 10:
            return

        # Container Pill
        r = self.radius
        d = 2 * r
        x1, y1 = 1, 1
        x2, y2 = w - 1, h - 1

        self.create_arc(x1, y1, x1 + d, y1 + d, start=90, extent=90, fill=AERO_THEME['bg_input'], outline="")
        self.create_arc(x2 - d, y1, x2, y1 + d, start=0, extent=90, fill=AERO_THEME['bg_input'], outline="")
        self.create_arc(x2 - d, y2 - d, x2, y2, start=270, extent=90, fill=AERO_THEME['bg_input'], outline="")
        self.create_arc(x1, y2 - d, x1 + d, y2, start=180, extent=90, fill=AERO_THEME['bg_input'], outline="")
        self.create_rectangle(x1 + r, y1, x2 - r, y2, fill=AERO_THEME['bg_input'], outline="")
        self.create_rectangle(x1, y1 + r, x2, y2 - r, fill=AERO_THEME['bg_input'], outline="")

        # 1px border
        self.create_arc(x1, y1, x1 + d, y1 + d, start=90, extent=90, style="arc", outline=AERO_THEME['border'], width=1)
        self.create_arc(x2 - d, y1, x2, y1 + d, start=0, extent=90, style="arc", outline=AERO_THEME['border'], width=1)
        self.create_arc(x2 - d, y2 - d, x2, y2, start=270, extent=90, style="arc", outline=AERO_THEME['border'], width=1)
        self.create_arc(x1, y2 - d, x1 + d, y2, start=180, extent=90, style="arc", outline=AERO_THEME['border'], width=1)
        self.create_line(x1 + r, y1, x2 - r, y1, fill=AERO_THEME['border'], width=1)
        self.create_line(x2, y1 + r, x2, y2 - r, fill=AERO_THEME['border'], width=1)
        self.create_line(x1 + r, y2, x2 - r, y2, fill=AERO_THEME['border'], width=1)
        self.create_line(x1, y1 + r, x1, y2 - r, fill=AERO_THEME['border'], width=1)

        n = len(self.options)
        seg_w = (w - 6) / n
        self.segments_bounds = []

        current_val = self.variable.get()

        for idx, opt in enumerate(self.options):
            sx1 = 3 + idx * seg_w
            sx2 = sx1 + seg_w
            sy1 = 3
            sy2 = h - 3
            self.segments_bounds.append((sx1, sx2, opt))

            is_active = (opt == current_val)
            if is_active:
                # Active Pill with Cyan Glow
                pr = min(r - 2, (sy2 - sy1) // 2)
                pd_ = 2 * pr
                self.create_arc(sx1, sy1, sx1 + pd_, sy1 + pd_, start=90, extent=90, fill=AERO_THEME['accent_hover'], outline="")
                self.create_arc(sx2 - pd_, sy1, sx2, sy1 + pd_, start=0, extent=90, fill=AERO_THEME['accent_hover'], outline="")
                self.create_arc(sx2 - pd_, sy2 - pd_, sx2, sy2, start=270, extent=90, fill=AERO_THEME['accent_hover'], outline="")
                self.create_arc(sx1, sy2 - pd_, sx1 + pd_, sy2, start=180, extent=90, fill=AERO_THEME['accent_hover'], outline="")
                self.create_rectangle(sx1 + pr, sy1, sx2 - pr, sy2, fill=AERO_THEME['accent_hover'], outline="")
                self.create_rectangle(sx1, sy1 + pr, sx2, sy2 - pr, fill=AERO_THEME['accent_hover'], outline="")
                txt_color = '#ffffff'
                txt_font = FONT_BODY_BOLD
            else:
                txt_color = AERO_THEME['fg_secondary']
                txt_font = FONT_BODY

            self.create_text((sx1 + sx2) / 2, h / 2, text=opt, fill=txt_color, font=txt_font)

    def _on_click(self, event):
        x = event.x
        for sx1, sx2, opt in self.segments_bounds:
            if sx1 <= x <= sx2:
                self.variable.set(opt)
                self._draw()
                if self.on_change:
                    self.on_change()
                break


class NeuroNavApp(tk.Tk):
    """Aerospace-grade Data-Science Mission-Control Dashboard for Satellite Error Forecasting."""

    def __init__(self) -> None:
        super().__init__()
        self.title("NeuroNav — Satellite Orbit & Clock Error Forecasting [Mission Control]")
        self.geometry("1400x920")
        self.minsize(1180, 780)
        self.configure(bg=AERO_THEME['bg_app'])

        # State Variables
        self.current_page_idx = 1
        self.input_file_path: Optional[Path] = None
        self.input_df: Optional[pd.DataFrame] = None
        self.input_file_type = tk.StringVar(value="Tabular CSV File (*.csv)")
        
        self.selected_model = tk.StringVar(value="Harmonic Ridge (PS-08 Winner)")
        self.detected_orbit = tk.StringVar(value="Auto-Detect (GEO)")
        self.forecast_horizon_str = tk.StringVar(value="24 Hours (15-min cadence)")
        
        self.predictions_df: Optional[pd.DataFrame] = None
        self.ground_truth_path: Optional[Path] = None
        self.ground_truth_df: Optional[pd.DataFrame] = None
        
        self.eval_merged_df: Optional[pd.DataFrame] = None
        self.eval_metrics: Optional[List[TargetMetrics]] = None
        self.eval_summary: Optional[Dict[str, Any]] = None
        self.current_plot_type = tk.StringVar(value="Histogram + KDE Density")

        # TTK Themes & Zebra Styling
        self._setup_ttk_styles()

        # 1. Persistent Top Mission-Control Bar
        self._build_top_bar()

        # 2. Page Container Stack
        self.container = tk.Frame(self, bg=AERO_THEME['bg_app'])
        self.container.pack(fill='both', expand=True, padx=20, pady=(0, 16))
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.page1 = tk.Frame(self.container, bg=AERO_THEME['bg_app'])
        self.page2 = tk.Frame(self.container, bg=AERO_THEME['bg_app'])
        self.page3 = tk.Frame(self.container, bg=AERO_THEME['bg_app'])

        self.page1.grid(row=0, column=0, sticky='nsew')
        self.page2.grid(row=0, column=0, sticky='nsew')
        self.page3.grid(row=0, column=0, sticky='nsew')

        # Build individual page views
        self._build_page1()
        self._build_page2()
        self._build_page3()

        # Start on Ingestion Step
        self.show_page(1)

    def _setup_ttk_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use('clam')

        # Treeview Datagrid: Zebra striping, monospace tabular numbers, elevated header
        style.configure(
            'Treeview',
            background=AERO_THEME['table_even'],
            fieldbackground=AERO_THEME['table_even'],
            foreground=AERO_THEME['fg_primary'],
            rowheight=30,
            font=FONT_TABLE_ROW,
            borderwidth=0
        )
        style.configure(
            'Treeview.Heading',
            background=AERO_THEME['table_header'],
            foreground=AERO_THEME['fg_secondary'],
            relief='flat',
            font=FONT_TABLE_HEAD,
            padding=6
        )
        style.map(
            'Treeview',
            background=[('selected', AERO_THEME['table_select'])],
            foreground=[('selected', AERO_THEME['accent_glow'])]
        )

        # Subtle Aerospace Scrollbars
        style.configure('Vertical.TScrollbar', background=AERO_THEME['bg_surface_alt'], troughcolor=AERO_THEME['bg_app'], arrowcolor=AERO_THEME['accent'])
        style.configure('Horizontal.TScrollbar', background=AERO_THEME['bg_surface_alt'], troughcolor=AERO_THEME['bg_app'], arrowcolor=AERO_THEME['accent'])

        # Combobox
        style.configure(
            'TCombobox',
            fieldbackground=AERO_THEME['bg_input'],
            background=AERO_THEME['bg_surface_alt'],
            foreground=AERO_THEME['fg_primary'],
            selectbackground=AERO_THEME['accent_dark'],
            selectforeground='#ffffff',
            arrowcolor=AERO_THEME['accent_glow'],
            padding=5,
            font=FONT_BODY
        )

    # =========================================================================
    # PERSISTENT TOP NAVIGATION BAR (MISSION CONTROL)
    # =========================================================================
    def _build_top_bar(self) -> None:
        top_frame = tk.Frame(self, bg=AERO_THEME['bg_header'], height=64)
        top_frame.pack(fill='x', side='top', pady=(0, 16))

        # Bottom subtle border on top bar
        bot_line = tk.Frame(top_frame, bg=AERO_THEME['border'], height=1)
        bot_line.pack(fill='x', side='bottom')

        inner = tk.Frame(top_frame, bg=AERO_THEME['bg_header'])
        inner.pack(fill='both', expand=True, padx=20, pady=10)

        # Left: App Brand & Status Pill
        left_box = tk.Frame(inner, bg=AERO_THEME['bg_header'])
        left_box.pack(side='left', fill='y')

        # Live Pulse Dot
        beacon = tk.Label(left_box, text="●", fg=AERO_THEME['success'], bg=AERO_THEME['bg_header'], font=(FONT_UI, 12, "bold"))
        beacon.pack(side='left', padx=(0, 6))

        title_lbl = tk.Label(
            left_box,
            text="NEURONAV",
            font=FONT_TITLE,
            fg=AERO_THEME['accent_glow'],
            bg=AERO_THEME['bg_header']
        )
        title_lbl.pack(side='left', padx=(0, 8))

        sub_lbl = tk.Label(
            left_box,
            text="// SATELLITE ORBIT & CLOCK ERROR FORECASTING",
            font=FONT_SMALL,
            fg=AERO_THEME['fg_secondary'],
            bg=AERO_THEME['bg_header']
        )
        sub_lbl.pack(side='left', padx=(0, 16))

        # Status Pill
        self.header_status_pill = tk.Label(
            left_box,
            text="MODEL: HARMONIC RIDGE · ORBIT: GEO · 24H HORIZON",
            font=FONT_BADGE,
            fg=AERO_THEME['accent_glow'],
            bg=AERO_THEME['bg_surface_alt'],
            padx=12,
            pady=4,
            highlightbackground=AERO_THEME['border'],
            highlightthickness=1
        )
        self.header_status_pill.pack(side='left')

        # Right: 3-Step Progress Indicator
        right_box = tk.Frame(inner, bg=AERO_THEME['bg_header'])
        right_box.pack(side='right', fill='y')

        self.step1_btn = AeroButton(right_box, text="01 INGEST", command=lambda: self.show_page(1), variant="primary", height=32, width=105, radius=6)
        self.step1_btn.pack(side='left', padx=3)

        arrow1 = tk.Label(right_box, text="➔", fg=AERO_THEME['fg_muted'], bg=AERO_THEME['bg_header'], font=FONT_BODY)
        arrow1.pack(side='left', padx=4)

        self.step2_btn = AeroButton(right_box, text="02 PREDICT", command=lambda: self.show_page(2), variant="secondary", height=32, width=110, radius=6)
        self.step2_btn.pack(side='left', padx=3)

        arrow2 = tk.Label(right_box, text="➔", fg=AERO_THEME['fg_muted'], bg=AERO_THEME['bg_header'], font=FONT_BODY)
        arrow2.pack(side='left', padx=4)

        self.step3_btn = AeroButton(right_box, text="03 ANALYZE", command=lambda: self.show_page(3), variant="secondary", height=32, width=110, radius=6)
        self.step3_btn.pack(side='left', padx=3)

    def show_page(self, page_num: int) -> None:
        self.current_page_idx = page_num
        if page_num == 1:
            self.page1.tkraise()
            self.step1_btn.variant = "primary"
            self.step1_btn.bg_color = AERO_THEME['accent']
            self.step1_btn.border_color = AERO_THEME['accent_glow']
            self.step1_btn.fg_color = '#ffffff'
            self.step1_btn._draw()

            self.step2_btn.variant = "secondary"
            self.step2_btn.bg_color = AERO_THEME['bg_surface_alt']
            self.step2_btn.border_color = AERO_THEME['border']
            self.step2_btn.fg_color = AERO_THEME['fg_secondary']
            self.step2_btn._draw()

            self.step3_btn.variant = "secondary"
            self.step3_btn.bg_color = AERO_THEME['bg_surface_alt']
            self.step3_btn.border_color = AERO_THEME['border']
            self.step3_btn.fg_color = AERO_THEME['fg_secondary']
            self.step3_btn._draw()

        elif page_num == 2:
            self.page2.tkraise()
            self.step1_btn.variant = "secondary"
            self.step1_btn.bg_color = AERO_THEME['success_bg']
            self.step1_btn.border_color = AERO_THEME['success']
            self.step1_btn.fg_color = AERO_THEME['success']
            self.step1_btn.set_text("✔ 01 INGEST")

            self.step2_btn.variant = "primary"
            self.step2_btn.bg_color = AERO_THEME['accent']
            self.step2_btn.border_color = AERO_THEME['accent_glow']
            self.step2_btn.fg_color = '#ffffff'
            self.step2_btn._draw()

            self.step3_btn.variant = "secondary"
            self.step3_btn.bg_color = AERO_THEME['bg_surface_alt']
            self.step3_btn.border_color = AERO_THEME['border']
            self.step3_btn.fg_color = AERO_THEME['fg_secondary']
            self.step3_btn._draw()

        else:
            self.page3.tkraise()
            self.step1_btn.variant = "secondary"
            self.step1_btn.bg_color = AERO_THEME['success_bg']
            self.step1_btn.border_color = AERO_THEME['success']
            self.step1_btn.fg_color = AERO_THEME['success']
            self.step1_btn.set_text("✔ 01 INGEST")

            self.step2_btn.variant = "secondary"
            self.step2_btn.bg_color = AERO_THEME['success_bg']
            self.step2_btn.border_color = AERO_THEME['success']
            self.step2_btn.fg_color = AERO_THEME['success']
            self.step2_btn.set_text("✔ 02 PREDICT")

            self.step3_btn.variant = "primary"
            self.step3_btn.bg_color = AERO_THEME['accent']
            self.step3_btn.border_color = AERO_THEME['accent_glow']
            self.step3_btn.fg_color = '#ffffff'
            self.step3_btn._draw()

    # =========================================================================
    # PAGE 1: Ingestion Pipeline & Compact Model Configuration
    # =========================================================================
    def _build_page1(self) -> None:
        p1 = self.page1

        split = tk.Frame(p1, bg=AERO_THEME['bg_app'])
        split.pack(fill='both', expand=True)
        split.grid_columnconfigure(0, weight=7)
        split.grid_columnconfigure(1, weight=4)
        split.grid_rowconfigure(0, weight=1)

        # ----------------- Left: Ingestion Card (Wider ~65%) -----------------
        left_card = AeroCard(split, bg_color=AERO_THEME['bg_surface'], border_color=AERO_THEME['border'], radius=10, inner_pad=18)
        left_card.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
        left = left_card.inner_frame

        # Section Header
        sec_h1 = tk.Frame(left, bg=AERO_THEME['bg_surface'])
        sec_h1.pack(fill='x', pady=(0, 10))

        tk.Label(
            sec_h1,
            text="DATASET INGESTION PIPELINE",
            font=FONT_HEADING,
            fg=AERO_THEME['fg_primary'],
            bg=AERO_THEME['bg_surface']
        ).pack(side='left')

        tk.Label(
            sec_h1,
            text="IGS Precise / Broadcast / Tabular Telemetry",
            font=FONT_SMALL,
            fg=AERO_THEME['fg_secondary'],
            bg=AERO_THEME['bg_surface']
        ).pack(side='right')

        # Pill Segmented Selector
        seg_options = ["Tabular CSV File (*.csv)", "Precise SP3 / RNX File (*.sp3, *.rnx)"]
        self.seg_ctrl = AeroSegmentedControl(left, options=seg_options, variable=self.input_file_type, height=36, radius=8)
        self.seg_ctrl.pack(fill='x', pady=(0, 12))

        # Unified Input Group: [ Label | Path Entry | Browse... ] + Quick Sample Pills
        input_group = tk.Frame(left, bg=AERO_THEME['bg_surface'])
        input_group.pack(fill='x', pady=(0, 14))

        # Path Entry with seamless border
        entry_wrap = tk.Frame(input_group, bg=AERO_THEME['bg_input'], highlightbackground=AERO_THEME['border'], highlightthickness=1)
        entry_wrap.pack(side='left', fill='x', expand=True, padx=(0, 8))

        tk.Label(
            entry_wrap,
            text="📁 FILE:",
            font=FONT_BADGE,
            fg=AERO_THEME['accent_glow'],
            bg=AERO_THEME['bg_input'],
            padx=8
        ).pack(side='left')

        self.file_entry_var = tk.StringVar(value="")
        entry = tk.Entry(
            entry_wrap,
            textvariable=self.file_entry_var,
            font=FONT_BODY,
            bg=AERO_THEME['bg_input'],
            fg=AERO_THEME['fg_primary'],
            insertbackground=AERO_THEME['accent_glow'],
            relief='flat',
            bd=0
        )
        entry.pack(side='left', fill='x', expand=True, ipady=6, padx=(0, 6))

        browse_btn = AeroButton(
            input_group,
            text="Browse...",
            command=self._browse_input_file,
            variant="secondary",
            width=100,
            height=34,
            radius=6
        )
        browse_btn.pack(side='left', padx=(0, 8))

        # Quick Load Sample Data Pills
        geo_sample_btn = AeroButton(
            input_group,
            text="GEO Train",
            command=self._load_sample_geo_train,
            variant="secondary",
            width=90,
            height=34,
            radius=6
        )
        geo_sample_btn.pack(side='left', padx=(0, 4))

        meo_sample_btn = AeroButton(
            input_group,
            text="MEO-1 Train",
            command=self._load_sample_meo1_train,
            variant="secondary",
            width=95,
            height=34,
            radius=6
        )
        meo_sample_btn.pack(side='left')

        # Dataset Status Bar
        ds_bar = tk.Frame(left, bg=AERO_THEME['bg_surface'])
        ds_bar.pack(fill='x', pady=(2, 8))

        tk.Label(
            ds_bar,
            text="TELEMETRY DATA MATRIX (ALL RECORDS)",
            font=FONT_SUBHEADING,
            fg=AERO_THEME['fg_secondary'],
            bg=AERO_THEME['bg_surface']
        ).pack(side='left')

        self.full_ds_badge = tk.Label(
            ds_bar,
            text="● AWAITING DATASET INGESTION",
            font=FONT_BADGE,
            fg=AERO_THEME['fg_muted'],
            bg=AERO_THEME['bg_surface_alt'],
            padx=10,
            pady=3,
            highlightbackground=AERO_THEME['border'],
            highlightthickness=1
        )
        self.full_ds_badge.pack(side='right')

        # Full Dataset Treeview Container with Zebra Striping
        table_wrap = tk.Frame(left, bg=AERO_THEME['bg_surface'], highlightbackground=AERO_THEME['border'], highlightthickness=1)
        table_wrap.pack(fill='both', expand=True)

        cols = ('row_idx', 'utc_time', 'x_err', 'y_err', 'z_err', 'clk_err', 'sat_id')
        self.full_table = ttk.Treeview(table_wrap, columns=cols, show='headings', height=14)
        self.full_table.heading('row_idx', text='#')
        self.full_table.heading('utc_time', text='UTC TIMESTAMP')
        self.full_table.heading('x_err', text='X ERROR (M)')
        self.full_table.heading('y_err', text='Y ERROR (M)')
        self.full_table.heading('z_err', text='Z ERROR (M)')
        self.full_table.heading('clk_err', text='CLOCK ERROR (M)')
        self.full_table.heading('sat_id', text='PRN')

        self.full_table.column('row_idx', width=45, anchor='center')
        self.full_table.column('utc_time', width=160, anchor='w')
        self.full_table.column('x_err', width=110, anchor='e')
        self.full_table.column('y_err', width=110, anchor='e')
        self.full_table.column('z_err', width=110, anchor='e')
        self.full_table.column('clk_err', width=120, anchor='e')
        self.full_table.column('sat_id', width=80, anchor='center')

        # Configure Zebra Striping Tags
        self.full_table.tag_configure('even', background=AERO_THEME['table_even'], foreground=AERO_THEME['fg_primary'])
        self.full_table.tag_configure('odd', background=AERO_THEME['table_odd'], foreground=AERO_THEME['fg_primary'])

        v_scroll = ttk.Scrollbar(table_wrap, orient='vertical', command=self.full_table.yview)
        h_scroll = ttk.Scrollbar(table_wrap, orient='horizontal', command=self.full_table.xview)
        self.full_table.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        v_scroll.pack(side='right', fill='y')
        h_scroll.pack(side='bottom', fill='x')
        self.full_table.pack(side='left', fill='both', expand=True)

        # ----------------- Right: Model Config Panel (Compact ~35%) -----------------
        right_card = AeroCard(split, bg_color=AERO_THEME['bg_surface'], border_color=AERO_THEME['border'], radius=10, inner_pad=18)
        right_card.grid(row=0, column=1, sticky='nsew', padx=(10, 0))
        right = right_card.inner_frame

        # Section Header
        sec_h2 = tk.Frame(right, bg=AERO_THEME['bg_surface'])
        sec_h2.pack(fill='x', pady=(0, 10))

        tk.Label(
            sec_h2,
            text="MODEL INFERENCE CONFIGURATION",
            font=FONT_HEADING,
            fg=AERO_THEME['fg_primary'],
            bg=AERO_THEME['bg_surface']
        ).pack(anchor='w')

        # Model Selector
        m_box = tk.Frame(right, bg=AERO_THEME['bg_surface'])
        m_box.pack(fill='x', pady=(0, 10))

        tk.Label(m_box, text="FORECASTING MODEL:", font=FONT_BADGE, fg=AERO_THEME['fg_secondary'], bg=AERO_THEME['bg_surface']).pack(anchor='w', pady=(0, 3))
        model_choices = [
            "Harmonic Ridge (PS-08 Winner)",
            "BiLSTM-GRU (Deep Neural Net)",
            "Random Forest Regressor",
            "Transformer (Attention Network)",
            "Gaussian Process Regressor"
        ]
        self.model_combo = ttk.Combobox(m_box, textvariable=self.selected_model, values=model_choices, state='readonly', font=FONT_BODY)
        self.model_combo.pack(fill='x', ipady=3)
        self.model_combo.bind("<<ComboboxSelected>>", self._update_header_pill)

        # Orbit Profile Selector
        o_box = tk.Frame(right, bg=AERO_THEME['bg_surface'])
        o_box.pack(fill='x', pady=(0, 10))

        tk.Label(o_box, text="ORBIT PROFILE:", font=FONT_BADGE, fg=AERO_THEME['fg_secondary'], bg=AERO_THEME['bg_surface']).pack(anchor='w', pady=(0, 3))
        self.orbit_combo = ttk.Combobox(o_box, textvariable=self.detected_orbit, values=["Auto-Detect (GEO)", "GEO", "MEO-1", "MEO-2"], state='readonly', font=FONT_BODY)
        self.orbit_combo.pack(fill='x', ipady=3)
        self.orbit_combo.bind("<<ComboboxSelected>>", self._update_header_pill)

        # Horizon Selector
        h_box = tk.Frame(right, bg=AERO_THEME['bg_surface'])
        h_box.pack(fill='x', pady=(0, 12))

        tk.Label(h_box, text="FORECAST HORIZON:", font=FONT_BADGE, fg=AERO_THEME['fg_secondary'], bg=AERO_THEME['bg_surface']).pack(anchor='w', pady=(0, 3))
        self.horizon_combo = ttk.Combobox(h_box, textvariable=self.forecast_horizon_str, values=["24 Hours (15-min cadence)", "12 Hours (15-min cadence)", "48 Hours (15-min cadence)"], state='readonly', font=FONT_BODY)
        self.horizon_combo.pack(fill='x', ipady=3)

        # Nested Telemetry Specification Card
        spec_card = AeroCard(right, bg_color=AERO_THEME['bg_input'], border_color=AERO_THEME['border'], radius=8, inner_pad=12)
        spec_card.pack(fill='x', pady=(0, 16))
        spec = spec_card.inner_frame

        tk.Label(
            spec,
            text="MISSION PROTOCOL SPECIFICATIONS",
            font=FONT_BADGE,
            fg=AERO_THEME['accent_glow'],
            bg=AERO_THEME['bg_input']
        ).pack(anchor='w', pady=(0, 6))

        spec_text = (
            "• Targets: X, Y, Z coordinates & SatClock bias\n"
            "• Training: 7-day multi-satellite ephemeris\n"
            "• Objective: Maximizing Shapiro-Wilk W Score\n"
            "• Benchmark Ref: W = 0.9810, p = 0.5840\n"
            "• Loss Policy: Uncertainty-weighted Gaussian NLL"
        )
        tk.Label(
            spec,
            text=spec_text,
            font=FONT_SMALL,
            justify='left',
            fg=AERO_THEME['fg_secondary'],
            bg=AERO_THEME['bg_input']
        ).pack(anchor='w')

        # Primary Compute Action Button
        self.compute_btn = AeroButton(
            right,
            text="Compute ML Forecast Predictions ➔",
            command=self._start_computation,
            variant="primary",
            height=44,
            radius=8
        )
        self.compute_btn.pack(fill='x', pady=(0, 8))

        # Status Readout
        self.status_lbl = tk.Label(
            right,
            text="● ENGINE IDLE: Select dataset and launch forecast",
            font=FONT_SMALL,
            fg=AERO_THEME['fg_secondary'],
            bg=AERO_THEME['bg_surface']
        )
        self.status_lbl.pack(anchor='center')

    # =========================================================================
    # PAGE 2: Predictions Output & 8th Day Ground Truth Upload
    # =========================================================================
    def _build_page2(self) -> None:
        p2 = self.page2

        # Model Run Banner Card
        banner_card = AeroCard(p2, bg_color=AERO_THEME['bg_surface'], border_color=AERO_THEME['border'], radius=10, inner_pad=12)
        banner_card.pack(fill='x', pady=(0, 14))
        banner = banner_card.inner_frame

        left_b = tk.Frame(banner, bg=AERO_THEME['bg_surface'])
        left_b.pack(side='left', fill='y')

        tk.Label(
            left_b,
            text="● FORECAST RESULTS //",
            font=FONT_HEADING,
            fg=AERO_THEME['accent_glow'],
            bg=AERO_THEME['bg_surface']
        ).pack(side='left', padx=(0, 8))

        self.p2_banner = tk.Label(
            left_b,
            text="MODEL: HARMONIC RIDGE · ORBIT: GEO · 96 EPOCHS (24.0 HRS)",
            font=FONT_BODY,
            fg=AERO_THEME['fg_primary'],
            bg=AERO_THEME['bg_surface']
        )
        self.p2_banner.pack(side='left')

        export_btn = AeroButton(
            banner,
            text="Export Predictions (CSV) 💾",
            command=self._export_predictions_csv,
            variant="secondary",
            width=190,
            height=32,
            radius=6
        )
        export_btn.pack(side='right')

        # Predictions Data Table Card
        pred_card = AeroCard(p2, bg_color=AERO_THEME['bg_surface'], border_color=AERO_THEME['border'], radius=10, inner_pad=16)
        pred_card.pack(fill='both', expand=True, pady=(0, 14))
        pred = pred_card.inner_frame

        tk.Label(
            pred,
            text="PREDICTED SATELLITE RESIDUAL SERIES",
            font=FONT_HEADING,
            fg=AERO_THEME['fg_primary'],
            bg=AERO_THEME['bg_surface']
        ).pack(anchor='w', pady=(0, 10))

        # Predictions Table Container
        pred_table_wrap = tk.Frame(pred, bg=AERO_THEME['bg_surface'], highlightbackground=AERO_THEME['border'], highlightthickness=1)
        pred_table_wrap.pack(fill='both', expand=True)

        p_cols = ('row_idx', 'utc_time', 'pred_x', 'pred_y', 'pred_z', 'pred_clk')
        self.pred_table = ttk.Treeview(pred_table_wrap, columns=p_cols, show='headings', height=12)
        self.pred_table.heading('row_idx', text='#')
        self.pred_table.heading('utc_time', text='UTC FORECAST EPOCH')
        self.pred_table.heading('pred_x', text='PREDICTED X ERROR (M)')
        self.pred_table.heading('pred_y', text='PREDICTED Y ERROR (M)')
        self.pred_table.heading('pred_z', text='PREDICTED Z ERROR (M)')
        self.pred_table.heading('pred_clk', text='PREDICTED CLOCK BIAS (M)')

        self.pred_table.column('row_idx', width=45, anchor='center')
        self.pred_table.column('utc_time', width=190, anchor='w')
        self.pred_table.column('pred_x', width=160, anchor='e')
        self.pred_table.column('pred_y', width=160, anchor='e')
        self.pred_table.column('pred_z', width=160, anchor='e')
        self.pred_table.column('pred_clk', width=180, anchor='e')

        self.pred_table.tag_configure('even', background=AERO_THEME['table_even'], foreground=AERO_THEME['fg_primary'])
        self.pred_table.tag_configure('odd', background=AERO_THEME['table_odd'], foreground=AERO_THEME['fg_primary'])

        pv_scroll = ttk.Scrollbar(pred_table_wrap, orient='vertical', command=self.pred_table.yview)
        ph_scroll = ttk.Scrollbar(pred_table_wrap, orient='horizontal', command=self.pred_table.xview)
        self.pred_table.configure(yscrollcommand=pv_scroll.set, xscrollcommand=ph_scroll.set)

        pv_scroll.pack(side='right', fill='y')
        ph_scroll.pack(side='bottom', fill='x')
        self.pred_table.pack(side='left', fill='both', expand=True)

        # Bottom Card: Ground Truth Upload & Compare Action
        gt_card = AeroCard(p2, bg_color=AERO_THEME['bg_surface'], border_color=AERO_THEME['border'], radius=10, inner_pad=14)
        gt_card.pack(fill='x')
        gt = gt_card.inner_frame

        top_gt = tk.Frame(gt, bg=AERO_THEME['bg_surface'])
        top_gt.pack(fill='x', pady=(0, 10))

        tk.Label(
            top_gt,
            text="DAY-8 GROUND TRUTH TELEMETRY & STATISTICAL COMPARISON",
            font=FONT_HEADING,
            fg=AERO_THEME['fg_primary'],
            bg=AERO_THEME['bg_surface']
        ).pack(side='left')

        self.gt_badge = tk.Label(
            top_gt,
            text="● NO GROUND TRUTH LOADED",
            font=FONT_BADGE,
            fg=AERO_THEME['fg_muted'],
            bg=AERO_THEME['bg_surface_alt'],
            padx=10,
            pady=3,
            highlightbackground=AERO_THEME['border'],
            highlightthickness=1
        )
        self.gt_badge.pack(side='right')

        # Action Bar
        act_bar = tk.Frame(gt, bg=AERO_THEME['bg_surface'])
        act_bar.pack(fill='x')

        upload_btn = AeroButton(
            act_bar,
            text="Upload 8th Day Data (CSV/SP3/RNX) 📁",
            command=self._browse_8th_day_file,
            variant="secondary",
            width=280,
            height=40,
            radius=8
        )
        upload_btn.pack(side='left', padx=(0, 8))

        sample_test_btn = AeroButton(
            act_bar,
            text="Load Sample Day-8 Test",
            command=self._load_sample_geo_test,
            variant="secondary",
            width=180,
            height=40,
            radius=8
        )
        sample_test_btn.pack(side='left', padx=(0, 12))

        self.compare_btn = AeroButton(
            act_bar,
            text="Compare & View Error Distribution ➔",
            command=self._run_comparison_and_goto_page3,
            variant="primary",
            width=300,
            height=40,
            radius=8
        )
        self.compare_btn.pack(side='right')
        self.compare_btn.config_state('disabled')

    # =========================================================================
    # PAGE 3: Statistical Results Table & Error Distribution Visuals
    # =========================================================================
    def _build_page3(self) -> None:
        p3 = self.page3

        # Top Card: Shapiro-Wilk Results Table
        stat_card = AeroCard(p3, bg_color=AERO_THEME['bg_surface'], border_color=AERO_THEME['border'], radius=10, inner_pad=16)
        stat_card.pack(fill='x', pady=(0, 14))
        stat = stat_card.inner_frame

        tk.Label(
            stat,
            text="SHAPIRO-WILK NORMALITY & HYPOTHESIS TEST RESULTS (α = 0.05)",
            font=FONT_HEADING,
            fg=AERO_THEME['fg_primary'],
            bg=AERO_THEME['bg_surface']
        ).pack(anchor='w', pady=(0, 10))

        sh_table_wrap = tk.Frame(stat, bg=AERO_THEME['bg_surface'], highlightbackground=AERO_THEME['border'], highlightthickness=1)
        sh_table_wrap.pack(fill='x')

        sh_cols = ('target', 'w_stat', 'p_val', 'alpha', 'hypothesis', 'bias', 'std', 'mae', 'rmse')
        self.shapiro_table = ttk.Treeview(sh_table_wrap, columns=sh_cols, show='headings', height=5)
        self.shapiro_table.heading('target', text='TARGET COMPONENT')
        self.shapiro_table.heading('w_stat', text='SHAPIRO-WILK W')
        self.shapiro_table.heading('p_val', text='P-VALUE')
        self.shapiro_table.heading('alpha', text='α LEVEL')
        self.shapiro_table.heading('hypothesis', text='HYPOTHESIS DECISION (H0: NORMAL)')
        self.shapiro_table.heading('bias', text='|BIAS| (M)')
        self.shapiro_table.heading('std', text='STD DEV (M)')
        self.shapiro_table.heading('mae', text='MAE (M)')
        self.shapiro_table.heading('rmse', text='RMSE (M)')

        self.shapiro_table.column('target', width=130, anchor='w')
        self.shapiro_table.column('w_stat', width=115, anchor='center')
        self.shapiro_table.column('p_val', width=110, anchor='center')
        self.shapiro_table.column('alpha', width=80, anchor='center')
        self.shapiro_table.column('hypothesis', width=220, anchor='center')
        self.shapiro_table.column('bias', width=85, anchor='e')
        self.shapiro_table.column('std', width=90, anchor='e')
        self.shapiro_table.column('mae', width=85, anchor='e')
        self.shapiro_table.column('rmse', width=85, anchor='e')

        # Semantic Pass / Fail Badges
        self.shapiro_table.tag_configure('pass', foreground=AERO_THEME['success'], background=AERO_THEME['table_even'])
        self.shapiro_table.tag_configure('reject', foreground=AERO_THEME['error'], background=AERO_THEME['table_odd'])
        self.shapiro_table.tag_configure('summary', background=AERO_THEME['table_header'], foreground=AERO_THEME['accent_glow'], font=FONT_TABLE_ROW_BOLD)

        self.shapiro_table.pack(fill='x')

        # Bottom Card: Residual Distribution & Q-Q Plots
        plot_card = AeroCard(p3, bg_color=AERO_THEME['bg_surface'], border_color=AERO_THEME['border'], radius=10, inner_pad=14)
        plot_card.pack(fill='both', expand=True)
        pl = plot_card.inner_frame

        p_hdr = tk.Frame(pl, bg=AERO_THEME['bg_surface'])
        p_hdr.pack(fill='x', pady=(0, 8))

        tk.Label(
            p_hdr,
            text="RESIDUAL ERROR PROBABILITY VISUALS (X, Y, Z, CLOCK)",
            font=FONT_HEADING,
            fg=AERO_THEME['fg_primary'],
            bg=AERO_THEME['bg_surface']
        ).pack(side='left')

        # Segmented Control for Plot View
        plot_modes = ["Histogram + KDE Density", "Normal Q-Q Plots (Priority 3)"]
        self.plot_seg = AeroSegmentedControl(p_hdr, options=plot_modes, variable=self.current_plot_type, on_change=self._render_plot, height=32, radius=6)
        self.plot_seg.pack(side='right')

        # Embedded Matplotlib Canvas Frame
        plot_wrap = tk.Frame(pl, bg=AERO_THEME['bg_surface'], highlightbackground=AERO_THEME['border'], highlightthickness=1)
        plot_wrap.pack(fill='both', expand=True)

        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Segoe UI', 'DejaVu Sans', 'Arial']

        self.fig, self.axes = plt.subplots(2, 2, figsize=(8, 4), facecolor=AERO_THEME['bg_surface'])
        self.fig.tight_layout(pad=2.4)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_wrap)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        for ax in self.axes.flat:
            ax.set_facecolor(AERO_THEME['bg_app'])
            ax.tick_params(colors=AERO_THEME['fg_secondary'], labelsize=9)
            for s in ax.spines.values():
                s.set_color(AERO_THEME['border'])
                s.set_linewidth(1.0)
            ax.text(0.5, 0.5, "Awaiting Day-8 Comparison", color=AERO_THEME['fg_muted'], ha='center', va='center', transform=ax.transAxes, fontsize=11)
        self.canvas.draw()

    # =========================================================================
    # LOGIC & EVENT HANDLERS
    # =========================================================================
    def _update_header_pill(self, event=None) -> None:
        m = self.selected_model.get().split('(')[0].strip()
        o = self.detected_orbit.get()
        h = self.forecast_horizon_str.get().split('(')[0].strip()
        self.header_status_pill.config(text=f"MODEL: {m.upper()} · {o.upper()} · {h.upper()}")

    def _browse_input_file(self) -> None:
        ft = self.input_file_type.get()
        if "CSV" in ft:
            filetypes = [("CSV Files", "*.csv"), ("Text Files", "*.txt"), ("All Files", "*.*")]
        else:
            filetypes = [
                ("SP3 & RNX Files", "*.sp3;*.rnx;*.nav;*.gz;*.*n;*.*p"),
                ("SP3 Files", "*.sp3;*.sp3.gz;*.eph"),
                ("RINEX Files", "*.rnx;*.nav;*.*n;*.*p"),
                ("All Files", "*.*")
            ]

        chosen = filedialog.askopenfilename(title="Select Training Dataset", filetypes=filetypes)
        if chosen:
            self._load_file_data(Path(chosen))

    def _load_sample_geo_train(self) -> None:
        p = PROJECT_ROOT / 'Data_PS-08' / 'DATA_GEO_Train.csv'
        if p.exists():
            self._load_file_data(p)
        else:
            messagebox.showwarning("File Missing", f"Could not find sample at {p}")

    def _load_sample_meo1_train(self) -> None:
        p = PROJECT_ROOT / 'Data_PS-08' / 'DATA_MEO-1_Train.csv'
        if p.exists():
            self._load_file_data(p)
        else:
            messagebox.showwarning("File Missing", f"Could not find sample at {p}")

    def _load_file_data(self, path: Path) -> None:
        try:
            self.status_lbl.config(text=f"● INGESTING {path.name}...", fg=AERO_THEME['accent_glow'])
            self.update_idletasks()

            df = load_dataset_file(path)
            self.input_file_path = path
            self.input_df = df
            self.file_entry_var.set(str(path))

            detected = detect_series_type(df, path)
            self.detected_orbit.set(f"Auto-Detect ({detected})")
            self._update_header_pill()

            self.full_table.delete(*self.full_table.get_children())
            for idx, row in df.iterrows():
                t_str = row['utc_time'].strftime('%Y-%m-%d %H:%M') if pd.notnull(row['utc_time']) else ""
                x_str = f"{row['x_error_m']:.4f}" if 'x_error_m' in row and pd.notnull(row['x_error_m']) else "—"
                y_str = f"{row['y_error_m']:.4f}" if 'y_error_m' in row and pd.notnull(row['y_error_m']) else "—"
                z_str = f"{row['z_error_m']:.4f}" if 'z_error_m' in row and pd.notnull(row['z_error_m']) else "—"
                c_str = f"{row['clock_error_m']:.4f}" if 'clock_error_m' in row and pd.notnull(row['clock_error_m']) else "—"
                s_id = str(row['satellite_id']) if 'satellite_id' in row and pd.notnull(row['satellite_id']) else "—"
                
                tag = 'even' if idx % 2 == 0 else 'odd'
                self.full_table.insert('', 'end', values=(idx + 1, t_str, x_str, y_str, z_str, c_str, s_id), tags=(tag,))

            t_min = df['utc_time'].min().strftime('%Y-%m-%d')
            t_max = df['utc_time'].max().strftime('%Y-%m-%d')
            self.full_ds_badge.config(
                text=f"● {len(df):,} RECORDS LOADED | {t_min} → {t_max} | PROFILE: {detected}",
                fg=AERO_THEME['accent_glow']
            )
            self.status_lbl.config(
                text=f"● READY: Ingested {path.name} ({len(df):,} records). Launch inference below.",
                fg=AERO_THEME['success']
            )
        except Exception as exc:
            messagebox.showerror("Ingestion Error", f"Failed to load dataset: {exc}")
            self.status_lbl.config(text=f"▲ Error loading file: {exc}", fg=AERO_THEME['error'])

    def _start_computation(self) -> None:
        if self.input_df is None or self.input_df.empty:
            messagebox.showwarning("No Data", "Please select or load a training dataset first.")
            return

        self.compute_btn.config_state('disabled')
        self.compute_btn.set_text("Computing ML Predictions... ⏳")
        self.status_lbl.config(text="● RUNNING INFERENCE: Multi-channel residual forecast...", fg=AERO_THEME['accent_glow'])

        thread = threading.Thread(target=self._run_model_thread, daemon=True)
        thread.start()

    def _run_model_thread(self) -> None:
        try:
            model_name = self.selected_model.get()
            orbit_val = self.detected_orbit.get()
            series = 'GEO' if 'GEO' in orbit_val else ('MEO-2' if 'MEO-2' in orbit_val else 'MEO-1')

            preds = compute_model_predictions(
                input_df=self.input_df,
                model_name=model_name,
                target_series=series
            )
            self.predictions_df = preds
            self.after(0, self._on_computation_finished)
        except Exception as exc:
            self.after(0, lambda: self._on_computation_failed(str(exc)))

    def _on_computation_failed(self, err_msg: str) -> None:
        self.compute_btn.config_state('normal')
        self.compute_btn.set_text("Compute ML Forecast Predictions ➔")
        self.status_lbl.config(text=f"▲ Computation failed: {err_msg}", fg=AERO_THEME['error'])
        messagebox.showerror("Computation Error", f"Model prediction error: {err_msg}")

    def _on_computation_finished(self) -> None:
        self.compute_btn.config_state('normal')
        self.compute_btn.set_text("Compute ML Forecast Predictions ➔")
        self.status_lbl.config(text="● INFERENCE COMPLETE! Advancing to Predictions...", fg=AERO_THEME['success'])

        self._populate_predictions_page()
        self.show_page(2)

    def _populate_predictions_page(self) -> None:
        if self.predictions_df is None:
            return

        self.pred_table.delete(*self.pred_table.get_children())
        for idx, row in self.predictions_df.iterrows():
            t_str = row['utc_time'].strftime('%Y-%m-%d %H:%M') if pd.notnull(row['utc_time']) else ""
            px = f"{row.get('predicted_x_error_m', 0.0):.4f}"
            py = f"{row.get('predicted_y_error_m', 0.0):.4f}"
            pz = f"{row.get('predicted_z_error_m', 0.0):.4f}"
            pc = f"{row.get('predicted_clock_error_m', 0.0):.4f}"
            tag = 'even' if idx % 2 == 0 else 'odd'
            self.pred_table.insert('', 'end', values=(idx + 1, t_str, px, py, pz, pc), tags=(tag,))

        model_name = self.selected_model.get().split('(')[0].strip()
        orbit_val = self.detected_orbit.get()
        t_start = self.predictions_df['utc_time'].min().strftime('%Y-%m-%d %H:%M')
        t_end = self.predictions_df['utc_time'].max().strftime('%Y-%m-%d %H:%M')
        count = len(self.predictions_df)
        self.p2_banner.config(
            text=f"MODEL: {model_name.upper()} · PROFILE: {orbit_val.upper()} · {count} EPOCHS PREDICTED ({t_start} → {t_end})"
        )

    def _browse_8th_day_file(self) -> None:
        chosen = filedialog.askopenfilename(
            title="Select 8th Day Ground Truth Dataset",
            filetypes=[
                ("All Supported Files", "*.csv;*.sp3;*.rnx;*.nav;*.txt"),
                ("CSV Files", "*.csv"),
                ("SP3 / RNX Files", "*.sp3;*.rnx;*.nav"),
                ("All Files", "*.*")
            ]
        )
        if chosen:
            self._load_8th_day_file(Path(chosen))

    def _load_sample_geo_test(self) -> None:
        p = PROJECT_ROOT / 'Data_PS-08' / 'DATA_GEO_Test.csv'
        if p.exists():
            self._load_8th_day_file(p)
        else:
            messagebox.showwarning("File Missing", f"Could not find sample test at {p}")

    def _load_8th_day_file(self, path: Path) -> None:
        try:
            df = load_dataset_file(path)
            self.ground_truth_path = path
            self.ground_truth_df = df

            t_min = df['utc_time'].min().strftime('%m/%d %H:%M')
            t_max = df['utc_time'].max().strftime('%m/%d %H:%M')
            self.gt_badge.config(
                text=f"● {path.name} ({len(df):,} obs, {t_min} → {t_max})",
                fg=AERO_THEME['success']
            )
            self.compare_btn.config_state('normal')
        except Exception as exc:
            messagebox.showerror("Ground Truth Error", f"Failed to load 8th-day file: {exc}")

    def _run_comparison_and_goto_page3(self) -> None:
        if self.predictions_df is None or self.ground_truth_df is None:
            messagebox.showwarning("Missing Data", "Ensure both predictions and 8th-day ground truth are loaded.")
            return

        try:
            model_name = self.selected_model.get()
            orbit_val = self.detected_orbit.get()
            series = 'GEO' if 'GEO' in orbit_val else ('MEO-2' if 'MEO-2' in orbit_val else 'MEO-1')

            # Re-predict precisely on 8th-day timestamps
            aligned_preds = compute_model_predictions(
                input_df=self.input_df,
                model_name=model_name,
                target_series=series,
                custom_forecast_times=self.ground_truth_df['utc_time']
            )
            self.predictions_df = aligned_preds
            self._populate_predictions_page()

            merged, metrics, summary = compare_and_evaluate(aligned_preds, self.ground_truth_df, alpha=0.05)
            self.eval_merged_df = merged
            self.eval_metrics = metrics
            self.eval_summary = summary

            self._populate_shapiro_table(metrics, summary)
            self._render_plot()
            self.show_page(3)

        except Exception as exc:
            messagebox.showerror("Comparison Error", f"Evaluation failed: {exc}")

    def _populate_shapiro_table(self, metrics: List[TargetMetrics], summary: Dict[str, Any]) -> None:
        self.shapiro_table.delete(*self.shapiro_table.get_children())

        for m in metrics:
            tag = 'pass' if not m.reject_normality else 'reject'
            status_badge = "● PASS (Normal)" if not m.reject_normality else "▲ REJECT H0 (Non-Gaussian)"
            self.shapiro_table.insert('', 'end', values=(
                m.target_label,
                f"{m.shapiro_w:.4f}",
                f"{m.p_value:.4e}" if m.p_value < 0.001 else f"{m.p_value:.4f}",
                "0.05",
                status_badge,
                f"{m.mean_bias:.4f}",
                f"{m.std_dev:.4f}",
                f"{m.mae:.4f}",
                f"{m.rmse:.4f}"
            ), tags=(tag,))

        # Macro Average Total Row
        w_avg = summary['average_shapiro_w']
        p_avg = summary['average_p_value']
        mae_avg = summary['overall_mae']
        rmse_avg = summary['overall_rmse']
        pass_rate = f"{summary['total_tests'] - summary['rejected_count']}/{summary['total_tests']} PASSED"

        self.shapiro_table.insert('', 'end', values=(
            "★ MACRO AVERAGE",
            f"{w_avg:.4f}",
            f"{p_avg:.4f}",
            "0.05",
            pass_rate,
            "—",
            "—",
            f"{mae_avg:.4f}",
            f"{rmse_avg:.4f}"
        ), tags=('summary',))

    def _render_plot(self) -> None:
        if self.eval_merged_df is None or self.eval_metrics is None:
            return

        plot_mode = self.current_plot_type.get()
        self.fig.clf()
        axes = self.fig.subplots(2, 2)
        self.fig.patch.set_facecolor(AERO_THEME['bg_surface'])

        targets_to_plot = [t for t in TARGETS if f'residual_{t}' in self.eval_merged_df.columns]

        for i, ax in enumerate(axes.flat):
            if i >= len(targets_to_plot):
                ax.axis('off')
                continue

            target = targets_to_plot[i]
            res_col = f'residual_{target}'
            res_vals = self.eval_merged_df[res_col].dropna().to_numpy()
            label = TARGET_LABELS.get(target, target)

            # Dark Aerospace Subplot Styling
            ax.set_facecolor(AERO_THEME['bg_app'])
            ax.tick_params(colors=AERO_THEME['fg_secondary'], labelsize=9)
            for spine in ax.spines.values():
                spine.set_color(AERO_THEME['border'])
                spine.set_linewidth(1.0)
            ax.grid(alpha=0.35, color=AERO_THEME['border'], linestyle=':')

            m = next((item for item in self.eval_metrics if item.target == target), None)
            w_str = f"W = {m.shapiro_w:.4f}" if m else ""

            if "Histogram" in plot_mode:
                # Cyan Bars with Glowing Cyan Edge
                counts, bins, _ = ax.hist(
                    res_vals, bins=16, density=True,
                    color=AERO_THEME['accent_dark'], edgecolor=AERO_THEME['accent_glow'], linewidth=1.2, alpha=0.85
                )
                if len(res_vals) > 1 and np.std(res_vals) > 0:
                    x_axis = np.linspace(res_vals.min(), res_vals.max(), 120)
                    pdf = stats.norm.pdf(x_axis, np.mean(res_vals), np.std(res_vals))
                    ax.plot(x_axis, pdf, color=AERO_THEME['accent_glow'], linewidth=2.2, linestyle='-', label='Gaussian Fit')

                ax.set_title(f"{label}  [{w_str}]", color=AERO_THEME['fg_primary'], fontsize=11, fontweight='bold', fontfamily=FONT_UI)
                ax.set_xlabel("Residual (m)", color=AERO_THEME['fg_secondary'], fontsize=9, fontfamily=FONT_UI)

            else:
                # Q-Q Plot with Cyan Scatter Markers
                stats.probplot(res_vals, dist="norm", plot=ax)
                ax.get_lines()[0].set_marker('o')
                ax.get_lines()[0].set_markersize(4.5)
                ax.get_lines()[0].set_markerfacecolor(AERO_THEME['accent_glow'])
                ax.get_lines()[0].set_markeredgecolor(AERO_THEME['accent'])
                ax.get_lines()[1].set_color(AERO_THEME['fg_secondary'])
                ax.get_lines()[1].set_linewidth(1.8)
                ax.get_lines()[1].set_linestyle('--')

                ax.set_title(f"Q-Q: {label}  [{w_str}]", color=AERO_THEME['fg_primary'], fontsize=11, fontweight='bold', fontfamily=FONT_UI)
                ax.set_xlabel("Theoretical Normal Quantiles", color=AERO_THEME['fg_secondary'], fontsize=9, fontfamily=FONT_UI)
                ax.set_ylabel("Residual Quantiles", color=AERO_THEME['fg_secondary'], fontsize=9, fontfamily=FONT_UI)

        self.fig.tight_layout(pad=2.4)
        self.canvas.draw()

    def _export_predictions_csv(self) -> None:
        if self.predictions_df is None or self.predictions_df.empty:
            messagebox.showwarning("No Predictions", "No predictions available to export.")
            return

        dest = filedialog.asksaveasfilename(
            title="Export Predictions",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")]
        )
        if dest:
            try:
                if self.eval_merged_df is not None:
                    self.eval_merged_df.to_csv(dest, index=False)
                else:
                    self.predictions_df.to_csv(dest, index=False)
                messagebox.showinfo("Export Successful", f"Predictions exported to {dest}")
            except Exception as exc:
                messagebox.showerror("Export Failed", f"Could not write file: {exc}")


def main() -> None:
    app = NeuroNavApp()
    app.mainloop()


if __name__ == '__main__':
    main()
