from __future__ import annotations
import os
import sys
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy import stats

try:
    from ml_engine.ml_engine import (
        load_dataset_file,
        compute_model_predictions,
        compare_and_evaluate,
        detect_series_type,
        TARGETS,
        TARGET_LABELS,
        TargetMetrics
    )
except ImportError:
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
# Black & White (Monochrome) Design System & Times New Roman Typography
# -----------------------------------------------------------------------------
FONT_FAMILY = "Times New Roman"
FONT_TITLE = (FONT_FAMILY, 22, "bold")
FONT_HEADING = (FONT_FAMILY, 17, "bold")
FONT_SUBHEADING = (FONT_FAMILY, 15, "bold")
FONT_BODY = (FONT_FAMILY, 15)
FONT_BODY_BOLD = (FONT_FAMILY, 15, "bold")
FONT_SMALL = (FONT_FAMILY, 13)
FONT_TABLE_HEAD = (FONT_FAMILY, 14, "bold")
FONT_TABLE_ROW = (FONT_FAMILY, 13)

BW_THEME = {
    'bg_app': '#000000',          # Pure Black
    'bg_surface': '#111111',      # Dark Charcoal Card
    'bg_surface_alt': '#1c1c1c',  # Secondary Surface / Banner
    'bg_input': '#080808',        # Input field background
    'border': '#444444',          # Subtle Card Borders
    'border_light': '#ffffff',    # High-contrast White Border
    'fg_primary': '#ffffff',      # Pure White Text
    'fg_secondary': '#cccccc',    # Silver / Soft White Text
    'fg_muted': '#888888',        # Muted Gray
    'btn_bg': '#ffffff',          # White Action Buttons
    'btn_fg': '#000000',          # Black Text on White Buttons
    'btn_alt_bg': '#222222',      # Dark Button
    'btn_alt_fg': '#ffffff',      # White Text on Dark Button
    'highlight': '#ffffff'        # Selection Highlight
}

DEFAULT_SAMPLE_TRAIN = PROJECT_ROOT / 'Data_PS-08' / 'DATA_GEO_Train.csv'
DEFAULT_SAMPLE_TEST = PROJECT_ROOT / 'Data_PS-08' / 'DATA_GEO_Test.csv'


class NeuroNavApp(tk.Tk):
    """3-Page Black & White NeuroNav GUI in Times New Roman (15pt)."""

    def __init__(self) -> None:
        super().__init__()
        self.title("NeuroNav — Satellite Orbit & Clock Error Forecasting")
        self.geometry("1340x880")
        self.minsize(1120, 750)
        self.configure(bg=BW_THEME['bg_app'])

        # State Variables
        self.input_file_path: Optional[Path] = None
        self.input_df: Optional[pd.DataFrame] = None
        self.input_file_type = tk.StringVar(value="csv")
        
        self.selected_model = tk.StringVar(value="Harmonic Ridge (PS-08 Winner)")
        self.detected_orbit = tk.StringVar(value="Auto-Detect (GEO)")
        self.forecast_horizon_str = tk.StringVar(value="24 Hours (15-min cadence)")
        
        self.predictions_df: Optional[pd.DataFrame] = None
        self.ground_truth_path: Optional[Path] = None
        self.ground_truth_df: Optional[pd.DataFrame] = None
        
        self.eval_merged_df: Optional[pd.DataFrame] = None
        self.eval_metrics: Optional[List[TargetMetrics]] = None
        self.eval_summary: Optional[Dict[str, Any]] = None
        self.current_plot_type = tk.StringVar(value="distribution")

        # Configure TTK Styles for Black & White and Times New Roman
        self._setup_ttk_styles()

        # Multi-page Stack Container
        self.container = tk.Frame(self, bg=BW_THEME['bg_app'])
        self.container.pack(fill='both', expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        # 3 Distinct Pages
        self.page1 = tk.Frame(self.container, bg=BW_THEME['bg_app'])
        self.page2 = tk.Frame(self.container, bg=BW_THEME['bg_app'])
        self.page3 = tk.Frame(self.container, bg=BW_THEME['bg_app'])

        self.page1.grid(row=0, column=0, sticky='nsew')
        self.page2.grid(row=0, column=0, sticky='nsew')
        self.page3.grid(row=0, column=0, sticky='nsew')

        # Build All Pages
        self._build_page1()
        self._build_page2()
        self._build_page3()

        # Start on Page 1
        self.show_page(1)

    def _setup_ttk_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use('clam')

        # Treeview styling (Black & White, Times New Roman 13-14pt, taller rows)
        style.configure(
            'Treeview',
            background=BW_THEME['bg_surface'],
            fieldbackground=BW_THEME['bg_surface'],
            foreground=BW_THEME['fg_primary'],
            rowheight=32,
            font=FONT_TABLE_ROW
        )
        style.configure(
            'Treeview.Heading',
            background=BW_THEME['bg_surface_alt'],
            foreground=BW_THEME['fg_primary'],
            relief='flat',
            font=FONT_TABLE_HEAD
        )
        style.map(
            'Treeview',
            background=[('selected', '#333333')],
            foreground=[('selected', '#ffffff')]
        )

        # Scrollbars
        style.configure('Vertical.TScrollbar', background='#333333', troughcolor=BW_THEME['bg_app'])
        style.configure('Horizontal.TScrollbar', background='#333333', troughcolor=BW_THEME['bg_app'])

        # Combobox styling
        style.configure(
            'TCombobox',
            fieldbackground=BW_THEME['bg_input'],
            background=BW_THEME['bg_surface_alt'],
            foreground=BW_THEME['fg_primary'],
            selectbackground='#333333',
            selectforeground='#ffffff',
            arrowcolor='#ffffff',
            padding=6,
            font=FONT_BODY
        )

    def show_page(self, page_num: int) -> None:
        """Switch view between Page 1 (Ingest/Compute), Page 2 (Predictions), Page 3 (Error Distribution)."""
        if page_num == 1:
            self.page1.tkraise()
        elif page_num == 2:
            self.page2.tkraise()
        else:
            self.page3.tkraise()

    # =========================================================================
    # PAGE 1: File Ingestion & Compact Model Configuration
    # =========================================================================
    def _build_page1(self) -> None:
        p1 = self.page1

        # Header Title
        header = tk.Frame(p1, bg=BW_THEME['bg_app'])
        header.pack(fill='x', padx=28, pady=(20, 14))

        tk.Label(
            header,
            text="NeuroNav · Satellite Orbit & Clock Error Forecasting",
            font=FONT_TITLE,
            fg=BW_THEME['fg_primary'],
            bg=BW_THEME['bg_app']
        ).pack(anchor='w')

        tk.Label(
            header,
            text="Page 1: Ingest Training Dataset & Configure Forecasting Model",
            font=FONT_SUBHEADING,
            fg=BW_THEME['fg_secondary'],
            bg=BW_THEME['bg_app']
        ).pack(anchor='w', pady=(2, 0))

        # Main Split Frame: Left (Wider Ingestion Panel), Right (Compact Model Configuration)
        split = tk.Frame(p1, bg=BW_THEME['bg_app'])
        split.pack(fill='both', expand=True, padx=28, pady=(0, 20))
        split.grid_columnconfigure(0, weight=7)  # Ingestion panel gets 70% width
        split.grid_columnconfigure(1, weight=4)  # Compact configuration panel gets 30% width
        split.grid_rowconfigure(0, weight=1)

        # ----------------- Left: Large Ingestion Panel -----------------
        left_card = tk.Frame(split, bg=BW_THEME['bg_surface'], highlightbackground=BW_THEME['border'], highlightthickness=1)
        left_card.grid(row=0, column=0, sticky='nsew', padx=(0, 12))

        # Card Title
        top_left = tk.Frame(left_card, bg=BW_THEME['bg_surface'])
        top_left.pack(fill='x', padx=18, pady=(16, 8))

        tk.Label(
            top_left,
            text="1. Select Ingestion Format",
            font=FONT_HEADING,
            fg=BW_THEME['fg_primary'],
            bg=BW_THEME['bg_surface']
        ).pack(anchor='w')

        tk.Label(
            top_left,
            text="Accepts tabular CSV dataset or raw GNSS broadcast/precise orbit products (SP3 / RNX):",
            font=FONT_BODY,
            fg=BW_THEME['fg_secondary'],
            bg=BW_THEME['bg_surface']
        ).pack(anchor='w', pady=(2, 8))

        # Radio Selectors for File Format
        radio_box = tk.Frame(left_card, bg=BW_THEME['bg_surface_alt'], padx=12, pady=8, highlightbackground=BW_THEME['border'], highlightthickness=1)
        radio_box.pack(fill='x', padx=18, pady=(0, 10))

        r_csv = tk.Radiobutton(
            radio_box,
            text="Tabular CSV File (*.csv)",
            variable=self.input_file_type,
            value="csv",
            font=FONT_BODY_BOLD,
            bg=BW_THEME['bg_surface_alt'],
            fg=BW_THEME['fg_primary'],
            selectcolor=BW_THEME['bg_input'],
            activebackground=BW_THEME['bg_surface_alt'],
            activeforeground='#ffffff'
        )
        r_csv.pack(side='left', padx=(0, 24))

        r_sp3 = tk.Radiobutton(
            radio_box,
            text="Precise SP3 / RNX File (*.sp3, *.rnx, *.nav)",
            variable=self.input_file_type,
            value="sp3_rnx",
            font=FONT_BODY_BOLD,
            bg=BW_THEME['bg_surface_alt'],
            fg=BW_THEME['fg_primary'],
            selectcolor=BW_THEME['bg_input'],
            activebackground=BW_THEME['bg_surface_alt'],
            activeforeground='#ffffff'
        )
        r_sp3.pack(side='left')

        # File Selection Entry & Browse Button
        browse_box = tk.Frame(left_card, bg=BW_THEME['bg_surface'])
        browse_box.pack(fill='x', padx=18, pady=(0, 8))

        self.file_entry_var = tk.StringVar(value="")
        entry = tk.Entry(
            browse_box,
            textvariable=self.file_entry_var,
            font=FONT_BODY,
            bg=BW_THEME['bg_input'],
            fg=BW_THEME['fg_primary'],
            insertbackground='#ffffff',
            relief='flat',
            highlightbackground=BW_THEME['border'],
            highlightthickness=1
        )
        entry.pack(side='left', fill='x', expand=True, ipady=6, padx=(0, 10))

        browse_btn = tk.Button(
            browse_box,
            text="Browse...",
            font=FONT_BODY_BOLD,
            bg=BW_THEME['btn_bg'],
            fg=BW_THEME['btn_fg'],
            activebackground='#cccccc',
            relief='flat',
            cursor='hand2',
            command=self._browse_input_file
        )
        browse_btn.pack(side='right', ipadx=10, ipady=4)

        # Quick Load Sample Buttons
        sample_box = tk.Frame(left_card, bg=BW_THEME['bg_surface'])
        sample_box.pack(fill='x', padx=18, pady=(0, 12))

        tk.Label(
            sample_box,
            text="Quick Load Sample Data:",
            font=FONT_BODY_BOLD,
            fg=BW_THEME['fg_secondary'],
            bg=BW_THEME['bg_surface']
        ).pack(side='left', padx=(0, 10))

        btn_geo = tk.Button(
            sample_box,
            text="Load GEO Train",
            font=FONT_SMALL,
            bg=BW_THEME['btn_alt_bg'],
            fg=BW_THEME['btn_alt_fg'],
            relief='flat',
            cursor='hand2',
            command=lambda: self._load_sample_file(DEFAULT_SAMPLE_TRAIN)
        )
        btn_geo.pack(side='left', padx=4, ipadx=6)

        btn_meo = tk.Button(
            sample_box,
            text="Load MEO-1 Train",
            font=FONT_SMALL,
            bg=BW_THEME['btn_alt_bg'],
            fg=BW_THEME['btn_alt_fg'],
            relief='flat',
            cursor='hand2',
            command=lambda: self._load_sample_file(PROJECT_ROOT / 'Data_PS-08' / 'DATA_MEO_Train.csv')
        )
        btn_meo.pack(side='left', padx=4, ipadx=6)

        # Full Dataset Display Header (Showing Entire Dataset)
        ds_header = tk.Frame(left_card, bg=BW_THEME['bg_surface'])
        ds_header.pack(fill='x', padx=18, pady=(4, 6))

        tk.Label(
            ds_header,
            text="Full Dataset View (All Records, Scrollable):",
            font=FONT_SUBHEADING,
            fg=BW_THEME['fg_primary'],
            bg=BW_THEME['bg_surface']
        ).pack(side='left')

        self.full_ds_badge = tk.Label(
            ds_header,
            text="No file loaded",
            font=FONT_SMALL,
            fg=BW_THEME['fg_secondary'],
            bg=BW_THEME['bg_surface_alt'],
            padx=10,
            pady=3,
            highlightbackground=BW_THEME['border'],
            highlightthickness=1
        )
        self.full_ds_badge.pack(side='right')

        # Full Dataset Treeview with BOTH Vertical & Horizontal Scrollbars
        table_container = tk.Frame(left_card, bg=BW_THEME['bg_surface'])
        table_container.pack(fill='both', expand=True, padx=18, pady=(0, 16))

        cols = ('row_idx', 'utc_time', 'x_err', 'y_err', 'z_err', 'clk_err', 'sat_id')
        self.full_table = ttk.Treeview(table_container, columns=cols, show='headings', height=14)
        self.full_table.heading('row_idx', text='#')
        self.full_table.heading('utc_time', text='UTC Time')
        self.full_table.heading('x_err', text='X Error (m)')
        self.full_table.heading('y_err', text='Y Error (m)')
        self.full_table.heading('z_err', text='Z Error (m)')
        self.full_table.heading('clk_err', text='Clock Error (m)')
        self.full_table.heading('sat_id', text='Satellite ID')

        self.full_table.column('row_idx', width=45, anchor='center')
        self.full_table.column('utc_time', width=160, anchor='w')
        self.full_table.column('x_err', width=110, anchor='e')
        self.full_table.column('y_err', width=110, anchor='e')
        self.full_table.column('z_err', width=110, anchor='e')
        self.full_table.column('clk_err', width=120, anchor='e')
        self.full_table.column('sat_id', width=95, anchor='center')

        # Scrollbars
        v_scroll = ttk.Scrollbar(table_container, orient='vertical', command=self.full_table.yview)
        h_scroll = ttk.Scrollbar(table_container, orient='horizontal', command=self.full_table.xview)
        self.full_table.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        v_scroll.pack(side='right', fill='y')
        h_scroll.pack(side='bottom', fill='x')
        self.full_table.pack(side='left', fill='both', expand=True)

        # ----------------- Right: Compact Model Configuration Panel -----------------
        right_card = tk.Frame(split, bg=BW_THEME['bg_surface'], highlightbackground=BW_THEME['border'], highlightthickness=1)
        right_card.grid(row=0, column=1, sticky='nsew', padx=(12, 0))

        # Header
        top_right = tk.Frame(right_card, bg=BW_THEME['bg_surface'])
        top_right.pack(fill='x', padx=18, pady=(16, 8))

        tk.Label(
            top_right,
            text="2. Model Configuration",
            font=FONT_HEADING,
            fg=BW_THEME['fg_primary'],
            bg=BW_THEME['bg_surface']
        ).pack(anchor='w')

        tk.Label(
            top_right,
            text="Machine learning pipeline settings:",
            font=FONT_BODY,
            fg=BW_THEME['fg_secondary'],
            bg=BW_THEME['bg_surface']
        ).pack(anchor='w', pady=(2, 10))

        # Model Selector
        m_box = tk.Frame(right_card, bg=BW_THEME['bg_surface'])
        m_box.pack(fill='x', padx=18, pady=(0, 10))

        tk.Label(m_box, text="Forecasting Model:", font=FONT_BODY_BOLD, fg=BW_THEME['fg_primary'], bg=BW_THEME['bg_surface']).pack(anchor='w', pady=(0, 3))
        model_choices = [
            "Harmonic Ridge (PS-08 Winner)",
            "BiLSTM-GRU (Deep Neural Net)",
            "Random Forest Regressor",
            "Transformer (Attention Network)",
            "Gaussian Process Regressor"
        ]
        self.model_combo = ttk.Combobox(m_box, textvariable=self.selected_model, values=model_choices, state='readonly', font=FONT_BODY)
        self.model_combo.pack(fill='x', ipady=4)

        # Orbit Profile Selector
        o_box = tk.Frame(right_card, bg=BW_THEME['bg_surface'])
        o_box.pack(fill='x', padx=18, pady=(0, 10))

        tk.Label(o_box, text="Orbit Profile:", font=FONT_BODY_BOLD, fg=BW_THEME['fg_primary'], bg=BW_THEME['bg_surface']).pack(anchor='w', pady=(0, 3))
        self.orbit_combo = ttk.Combobox(o_box, textvariable=self.detected_orbit, values=["Auto-Detect (GEO)", "GEO", "MEO-1", "MEO-2"], state='readonly', font=FONT_BODY)
        self.orbit_combo.pack(fill='x', ipady=4)

        # Horizon Selector
        h_box = tk.Frame(right_card, bg=BW_THEME['bg_surface'])
        h_box.pack(fill='x', padx=18, pady=(0, 10))

        tk.Label(h_box, text="Forecast Horizon:", font=FONT_BODY_BOLD, fg=BW_THEME['fg_primary'], bg=BW_THEME['bg_surface']).pack(anchor='w', pady=(0, 3))
        self.horizon_combo = ttk.Combobox(h_box, textvariable=self.forecast_horizon_str, values=["24 Hours (15-min cadence)", "12 Hours (15-min cadence)", "48 Hours (15-min cadence)"], state='readonly', font=FONT_BODY)
        self.horizon_combo.pack(fill='x', ipady=4)

        # Model Architecture & Hyperparameter Summary Box (Keeps panel well-filled & informative)
        info_box = tk.Frame(right_card, bg=BW_THEME['bg_surface_alt'], highlightbackground=BW_THEME['border'], highlightthickness=1)
        info_box.pack(fill='x', padx=18, pady=(0, 14))

        tk.Label(
            info_box,
            text="Model Architecture & Protocol",
            font=FONT_BODY_BOLD,
            fg=BW_THEME['fg_primary'],
            bg=BW_THEME['bg_surface_alt']
        ).pack(anchor='w', padx=12, pady=(10, 4))

        info_text = (
            "• Targets: X error, Y error, Z error, Clock bias\n"
            "• Training: 7-day multi-satellite ephemeris\n"
            "• Evaluation Criterion: Priority-1 Shapiro-Wilk W\n"
            "• Reference Normality Score: W = 0.9810, p = 0.5840\n"
            "• Tie-Breakers: Mean bias, Std Dev, Q-Q plots"
        )
        tk.Label(
            info_box,
            text=info_text,
            font=FONT_SMALL,
            justify='left',
            fg=BW_THEME['fg_secondary'],
            bg=BW_THEME['bg_surface_alt']
        ).pack(anchor='w', padx=12, pady=(0, 10))

        # Action Button Box
        action_box = tk.Frame(right_card, bg=BW_THEME['bg_surface'])
        action_box.pack(fill='x', padx=18, pady=(0, 16))

        self.compute_btn = tk.Button(
            action_box,
            text="Compute ML Forecast Predictions ➔",
            font=FONT_BODY_BOLD,
            bg=BW_THEME['btn_bg'],
            fg=BW_THEME['btn_fg'],
            activebackground='#dddddd',
            relief='flat',
            cursor='hand2',
            command=self._start_computation
        )
        self.compute_btn.pack(fill='x', ipady=10)

        self.status_lbl = tk.Label(
            action_box,
            text="Ready. Load a dataset to compute.",
            font=FONT_SMALL,
            fg=BW_THEME['fg_secondary'],
            bg=BW_THEME['bg_surface']
        )
        self.status_lbl.pack(anchor='center', pady=(8, 0))

    # =========================================================================
    # PAGE 2: Output of the ML Model & 8th-Day Ground Truth Upload
    # =========================================================================
    def _build_page2(self) -> None:
        p2 = self.page2

        # Header Navigation Bar
        nav_bar = tk.Frame(p2, bg=BW_THEME['bg_app'])
        nav_bar.pack(fill='x', padx=28, pady=(18, 10))

        back_btn = tk.Button(
            nav_bar,
            text="⬅ Back to Ingestion (Page 1)",
            font=FONT_BODY_BOLD,
            bg=BW_THEME['btn_alt_bg'],
            fg=BW_THEME['btn_alt_fg'],
            relief='flat',
            cursor='hand2',
            command=lambda: self.show_page(1)
        )
        back_btn.pack(side='left', ipadx=10, ipady=4, padx=(0, 16))

        tk.Label(
            nav_bar,
            text="Page 2: ML Model Output Predictions & 8th Day Input",
            font=FONT_TITLE,
            fg=BW_THEME['fg_primary'],
            bg=BW_THEME['bg_app']
        ).pack(side='left')

        export_btn = tk.Button(
            nav_bar,
            text="Export Predictions (CSV) 💾",
            font=FONT_BODY_BOLD,
            bg=BW_THEME['btn_bg'],
            fg=BW_THEME['btn_fg'],
            relief='flat',
            cursor='hand2',
            command=self._export_predictions_csv
        )
        export_btn.pack(side='right', ipadx=10, ipady=4)

        # Top Model Banner
        self.p2_banner = tk.Label(
            p2,
            text="Model: Not computed yet",
            font=FONT_BODY,
            bg=BW_THEME['bg_surface_alt'],
            fg=BW_THEME['fg_primary'],
            padx=16,
            pady=8,
            highlightbackground=BW_THEME['border'],
            highlightthickness=1
        )
        self.p2_banner.pack(fill='x', padx=28, pady=(0, 14))

        # Main Table Card: Full Predictions Output
        pred_card = tk.Frame(p2, bg=BW_THEME['bg_surface'], highlightbackground=BW_THEME['border'], highlightthickness=1)
        pred_card.pack(fill='both', expand=True, padx=28, pady=(0, 14))

        tk.Label(
            pred_card,
            text="ML Model Predicted Output Values (Series of Predictions):",
            font=FONT_HEADING,
            fg=BW_THEME['fg_primary'],
            bg=BW_THEME['bg_surface']
        ).pack(anchor='w', padx=18, pady=(14, 8))

        # Scrollable Predictions Table
        pred_frame = tk.Frame(pred_card, bg=BW_THEME['bg_surface'])
        pred_frame.pack(fill='both', expand=True, padx=18, pady=(0, 14))

        p_cols = ('row_idx', 'utc_time', 'pred_x', 'pred_y', 'pred_z', 'pred_clk')
        self.pred_table = ttk.Treeview(pred_frame, columns=p_cols, show='headings', height=12)
        self.pred_table.heading('row_idx', text='#')
        self.pred_table.heading('utc_time', text='UTC Forecast Epoch')
        self.pred_table.heading('pred_x', text='Predicted X Error (m)')
        self.pred_table.heading('pred_y', text='Predicted Y Error (m)')
        self.pred_table.heading('pred_z', text='Predicted Z Error (m)')
        self.pred_table.heading('pred_clk', text='Predicted Clock Bias (m)')

        self.pred_table.column('row_idx', width=45, anchor='center')
        self.pred_table.column('utc_time', width=190, anchor='w')
        self.pred_table.column('pred_x', width=160, anchor='e')
        self.pred_table.column('pred_y', width=160, anchor='e')
        self.pred_table.column('pred_z', width=160, anchor='e')
        self.pred_table.column('pred_clk', width=180, anchor='e')

        pv_scroll = ttk.Scrollbar(pred_frame, orient='vertical', command=self.pred_table.yview)
        ph_scroll = ttk.Scrollbar(pred_frame, orient='horizontal', command=self.pred_table.xview)
        self.pred_table.configure(yscrollcommand=pv_scroll.set, xscrollcommand=ph_scroll.set)

        pv_scroll.pack(side='right', fill='y')
        ph_scroll.pack(side='bottom', fill='x')
        self.pred_table.pack(side='left', fill='both', expand=True)

        # Bottom Card: 8th-Day Ground Truth Input & Compare Action
        comp_card = tk.Frame(p2, bg=BW_THEME['bg_surface'], highlightbackground=BW_THEME['border'], highlightthickness=1)
        comp_card.pack(fill='x', padx=28, pady=(0, 18))

        top_comp = tk.Frame(comp_card, bg=BW_THEME['bg_surface'])
        top_comp.pack(fill='x', padx=18, pady=(12, 6))

        tk.Label(
            top_comp,
            text="Day-8 Ground Truth Data & Comparison:",
            font=FONT_HEADING,
            fg=BW_THEME['fg_primary'],
            bg=BW_THEME['bg_surface']
        ).pack(side='left')

        self.gt_badge = tk.Label(
            top_comp,
            text="No 8th-day file loaded",
            font=FONT_SMALL,
            fg=BW_THEME['fg_secondary'],
            bg=BW_THEME['bg_surface_alt'],
            padx=10,
            pady=3,
            highlightbackground=BW_THEME['border'],
            highlightthickness=1
        )
        self.gt_badge.pack(side='right')

        # Upload Bar
        upload_box = tk.Frame(comp_card, bg=BW_THEME['bg_surface'])
        upload_box.pack(fill='x', padx=18, pady=(0, 14))

        upload_btn = tk.Button(
            upload_box,
            text="Upload 8th Day Data (CSV/SP3/RNX) 📁",
            font=FONT_BODY_BOLD,
            bg=BW_THEME['btn_alt_bg'],
            fg=BW_THEME['btn_alt_fg'],
            relief='flat',
            cursor='hand2',
            command=self._browse_8th_day_file
        )
        upload_btn.pack(side='left', padx=(0, 10), ipady=6, ipadx=10)

        sample_gt_btn = tk.Button(
            upload_box,
            text="Load Sample Day-8 Test",
            font=FONT_BODY,
            bg=BW_THEME['btn_alt_bg'],
            fg=BW_THEME['btn_alt_fg'],
            relief='flat',
            cursor='hand2',
            command=lambda: self._load_8th_day_file(DEFAULT_SAMPLE_TEST)
        )
        sample_gt_btn.pack(side='left', padx=(0, 16), ipady=6, ipadx=8)

        # Compare Button to advance to Page 3
        self.compare_btn = tk.Button(
            upload_box,
            text="Compare & View Error Distribution ➔",
            font=FONT_BODY_BOLD,
            bg=BW_THEME['btn_bg'],
            fg=BW_THEME['btn_fg'],
            activebackground='#dddddd',
            relief='flat',
            cursor='hand2',
            state='disabled',
            command=self._run_comparison_and_goto_page3
        )
        self.compare_btn.pack(side='right', ipady=8, ipadx=16)

    # =========================================================================
    # PAGE 3: Error Distribution Graphs & Shapiro-Wilk Statistical Results
    # =========================================================================
    def _build_page3(self) -> None:
        p3 = self.page3

        # Header Navigation Bar
        nav_bar = tk.Frame(p3, bg=BW_THEME['bg_app'])
        nav_bar.pack(fill='x', padx=28, pady=(18, 10))

        back_p2_btn = tk.Button(
            nav_bar,
            text="⬅ Back to Predictions (Page 2)",
            font=FONT_BODY_BOLD,
            bg=BW_THEME['btn_alt_bg'],
            fg=BW_THEME['btn_alt_fg'],
            relief='flat',
            cursor='hand2',
            command=lambda: self.show_page(2)
        )
        back_p2_btn.pack(side='left', ipadx=10, ipady=4, padx=(0, 12))

        back_p1_btn = tk.Button(
            nav_bar,
            text="Ingestion (Page 1)",
            font=FONT_BODY,
            bg=BW_THEME['btn_alt_bg'],
            fg=BW_THEME['btn_alt_fg'],
            relief='flat',
            cursor='hand2',
            command=lambda: self.show_page(1)
        )
        back_p1_btn.pack(side='left', ipadx=8, ipady=4, padx=(0, 16))

        tk.Label(
            nav_bar,
            text="Page 3: Error Distribution & Shapiro-Wilk Hypothesis Tests",
            font=FONT_TITLE,
            fg=BW_THEME['fg_primary'],
            bg=BW_THEME['bg_app']
        ).pack(side='left')

        # Top Card: Shapiro-Wilk Normality & Hypothesis Test Table
        stat_card = tk.Frame(p3, bg=BW_THEME['bg_surface'], highlightbackground=BW_THEME['border'], highlightthickness=1)
        stat_card.pack(fill='x', padx=28, pady=(0, 12))

        tk.Label(
            stat_card,
            text="Shapiro-Wilk W Statistic, p-values & Hypothesis Test Results (α = 0.05):",
            font=FONT_HEADING,
            fg=BW_THEME['fg_primary'],
            bg=BW_THEME['bg_surface']
        ).pack(anchor='w', padx=18, pady=(12, 6))

        sh_frame = tk.Frame(stat_card, bg=BW_THEME['bg_surface'])
        sh_frame.pack(fill='x', padx=18, pady=(0, 14))

        sh_cols = ('target', 'w_stat', 'p_val', 'alpha', 'hypothesis', 'bias', 'std', 'mae', 'rmse')
        self.shapiro_table = ttk.Treeview(sh_frame, columns=sh_cols, show='headings', height=5)
        self.shapiro_table.heading('target', text='Target Component')
        self.shapiro_table.heading('w_stat', text='Shapiro-Wilk W')
        self.shapiro_table.heading('p_val', text='p-value')
        self.shapiro_table.heading('alpha', text='α Level')
        self.shapiro_table.heading('hypothesis', text='Hypothesis Test Result (H0: Normal)')
        self.shapiro_table.heading('bias', text='|Bias| (m)')
        self.shapiro_table.heading('std', text='Std Dev (m)')
        self.shapiro_table.heading('mae', text='MAE (m)')
        self.shapiro_table.heading('rmse', text='RMSE (m)')

        self.shapiro_table.column('target', width=130, anchor='w')
        self.shapiro_table.column('w_stat', width=105, anchor='center')
        self.shapiro_table.column('p_val', width=105, anchor='center')
        self.shapiro_table.column('alpha', width=75, anchor='center')
        self.shapiro_table.column('hypothesis', width=200, anchor='center')
        self.shapiro_table.column('bias', width=80, anchor='e')
        self.shapiro_table.column('std', width=80, anchor='e')
        self.shapiro_table.column('mae', width=80, anchor='e')
        self.shapiro_table.column('rmse', width=80, anchor='e')

        self.shapiro_table.tag_configure('pass', foreground='#ffffff')
        self.shapiro_table.tag_configure('reject', foreground='#aaaaaa')
        self.shapiro_table.tag_configure('summary', background=BW_THEME['bg_surface_alt'], font=FONT_TABLE_HEAD)

        self.shapiro_table.pack(fill='x')

        # Bottom Card: Embedded Matplotlib Error Distribution Plots
        plot_card = tk.Frame(p3, bg=BW_THEME['bg_surface'], highlightbackground=BW_THEME['border'], highlightthickness=1)
        plot_card.pack(fill='both', expand=True, padx=28, pady=(0, 16))

        plot_ctrl = tk.Frame(plot_card, bg=BW_THEME['bg_surface'])
        plot_ctrl.pack(fill='x', padx=18, pady=(10, 4))

        tk.Label(
            plot_ctrl,
            text="Residual Error Distribution Visuals:",
            font=FONT_HEADING,
            fg=BW_THEME['fg_primary'],
            bg=BW_THEME['bg_surface']
        ).pack(side='left')

        # Plot Type Switcher
        r_qq = tk.Radiobutton(
            plot_ctrl,
            text="Normal Q-Q Plot (Judge Priority 3)",
            variable=self.current_plot_type,
            value="qq",
            font=FONT_BODY,
            bg=BW_THEME['bg_surface'],
            fg=BW_THEME['fg_primary'],
            selectcolor=BW_THEME['bg_input'],
            command=self._render_plot
        )
        r_qq.pack(side='right', padx=(10, 0))

        r_dist = tk.Radiobutton(
            plot_ctrl,
            text="Error Distribution (Histogram + KDE)",
            variable=self.current_plot_type,
            value="distribution",
            font=FONT_BODY,
            bg=BW_THEME['bg_surface'],
            fg=BW_THEME['fg_primary'],
            selectcolor=BW_THEME['bg_input'],
            command=self._render_plot
        )
        r_dist.pack(side='right', padx=(10, 10))

        # Matplotlib Canvas Frame
        self.plot_container = tk.Frame(plot_card, bg=BW_THEME['bg_input'])
        self.plot_container.pack(fill='both', expand=True, padx=18, pady=(0, 12))

        # Configure Matplotlib rcParams for Times New Roman & Black/White
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']

        self.fig, self.axes = plt.subplots(2, 2, figsize=(7.5, 3.8), facecolor=BW_THEME['bg_input'])
        self.fig.tight_layout(pad=2.2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_container)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        for ax in self.axes.flat:
            ax.set_facecolor(BW_THEME['bg_surface'])
            ax.tick_params(colors=BW_THEME['fg_secondary'], labelsize=9)
            for spine in ax.spines.values():
                spine.set_color(BW_THEME['border'])
            ax.text(0.5, 0.5, "Awaiting Comparison", color=BW_THEME['fg_secondary'], ha='center', va='center', transform=ax.transAxes, fontsize=11, fontfamily=FONT_FAMILY)
        self.canvas.draw()

    # =========================================================================
    # Event Handlers & Core Business Logic
    # =========================================================================
    def _browse_input_file(self) -> None:
        ft = self.input_file_type.get()
        if ft == 'csv':
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

    def _load_sample_file(self, path: Path) -> None:
        if not path.exists():
            messagebox.showwarning("File Missing", f"Could not find sample at: {path}")
            return
        self.input_file_type.set("csv")
        self._load_file_data(path)

    def _load_file_data(self, path: Path) -> None:
        try:
            self.status_lbl.config(text=f"Loading {path.name}...", fg=BW_THEME['fg_primary'])
            self.update_idletasks()

            df = load_dataset_file(path)
            self.input_file_path = path
            self.input_df = df
            self.file_entry_var.set(str(path))

            detected = detect_series_type(df, path)
            self.detected_orbit.set(f"Auto-Detect ({detected})")

            # Load the WHOLE dataset into the scrollable Treeview (as requested!)
            self.full_table.delete(*self.full_table.get_children())
            
            # Efficient bulk insertion of all rows
            for idx, row in df.iterrows():
                t_str = row['utc_time'].strftime('%Y-%m-%d %H:%M') if pd.notnull(row['utc_time']) else ""
                x_str = f"{row['x_error_m']:.4f}" if 'x_error_m' in row and pd.notnull(row['x_error_m']) else "—"
                y_str = f"{row['y_error_m']:.4f}" if 'y_error_m' in row and pd.notnull(row['y_error_m']) else "—"
                z_str = f"{row['z_error_m']:.4f}" if 'z_error_m' in row and pd.notnull(row['z_error_m']) else "—"
                c_str = f"{row['clock_error_m']:.4f}" if 'clock_error_m' in row and pd.notnull(row['clock_error_m']) else "—"
                s_id = str(row['satellite_id']) if 'satellite_id' in row and pd.notnull(row['satellite_id']) else "—"
                self.full_table.insert('', 'end', values=(idx + 1, t_str, x_str, y_str, z_str, c_str, s_id))

            t_min = df['utc_time'].min().strftime('%Y-%m-%d')
            t_max = df['utc_time'].max().strftime('%Y-%m-%d')
            self.full_ds_badge.config(
                text=f"{len(df):,} Total Rows | {t_min} to {t_max} | Profile: {detected}"
            )
            self.status_lbl.config(
                text=f"Loaded {path.name} successfully ({len(df):,} rows). Ready to compute.",
                fg=BW_THEME['fg_primary']
            )

        except Exception as exc:
            messagebox.showerror("Ingestion Error", f"Failed to load dataset: {exc}")
            self.status_lbl.config(text=f"Error loading file: {exc}", fg=BW_THEME['fg_secondary'])

    def _start_computation(self) -> None:
        if self.input_df is None or self.input_df.empty:
            messagebox.showwarning("No Data", "Please select or load a training dataset first.")
            return

        self.compute_btn.config(state='disabled', text="Computing ML Predictions... ⏳")
        self.status_lbl.config(text="Executing model inference in background...", fg=BW_THEME['fg_primary'])

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
        self.compute_btn.config(state='normal', text="Compute ML Forecast Predictions ➔")
        self.status_lbl.config(text=f"Computation failed: {err_msg}", fg=BW_THEME['fg_secondary'])
        messagebox.showerror("Computation Error", f"Model prediction error: {err_msg}")

    def _on_computation_finished(self) -> None:
        self.compute_btn.config(state='normal', text="Compute ML Forecast Predictions ➔")
        self.status_lbl.config(text="Computation complete! Advancing to Page 2...", fg=BW_THEME['fg_primary'])

        # Populate Page 2
        self._populate_predictions_page()

        # Advance to Page 2
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
            self.pred_table.insert('', 'end', values=(idx + 1, t_str, px, py, pz, pc))

        model_name = self.selected_model.get()
        orbit_val = self.detected_orbit.get()
        t_start = self.predictions_df['utc_time'].min().strftime('%Y-%m-%d %H:%M')
        t_end = self.predictions_df['utc_time'].max().strftime('%Y-%m-%d %H:%M')
        count = len(self.predictions_df)
        self.p2_banner.config(
            text=f"Model: {model_name}   |   Profile: {orbit_val}   |   Total Predicted Epochs: {count}   ({t_start} → {t_end})"
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

    def _load_8th_day_file(self, path: Path) -> None:
        try:
            df = load_dataset_file(path)
            self.ground_truth_path = path
            self.ground_truth_df = df

            t_min = df['utc_time'].min().strftime('%m/%d %H:%M')
            t_max = df['utc_time'].max().strftime('%m/%d %H:%M')
            self.gt_badge.config(
                text=f"{path.name} ({len(df):,} obs, {t_min} - {t_max})"
            )
            self.compare_btn.config(state='normal')
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

            # Statistical evaluation
            merged, metrics, summary = compare_and_evaluate(aligned_preds, self.ground_truth_df, alpha=0.05)
            self.eval_merged_df = merged
            self.eval_metrics = metrics
            self.eval_summary = summary

            # Populate Shapiro-Wilk Table on Page 3
            self._populate_shapiro_table(metrics, summary)

            # Render Matplotlib plots on Page 3
            self._render_plot()

            # Advance directly to Page 3
            self.show_page(3)

        except Exception as exc:
            messagebox.showerror("Comparison Error", f"Evaluation failed: {exc}")

    def _populate_shapiro_table(self, metrics: List[TargetMetrics], summary: Dict[str, Any]) -> None:
        self.shapiro_table.delete(*self.shapiro_table.get_children())

        for m in metrics:
            tag = 'pass' if not m.reject_normality else 'reject'
            self.shapiro_table.insert('', 'end', values=(
                m.target_label,
                f"{m.shapiro_w:.4f}",
                f"{m.p_value:.4e}" if m.p_value < 0.001 else f"{m.p_value:.4f}",
                "0.05",
                m.hypothesis_result_str,
                f"{m.mean_bias:.4f}",
                f"{m.std_dev:.4f}",
                f"{m.mae:.4f}",
                f"{m.rmse:.4f}"
            ), tags=(tag,))

        # Macro Average Row
        w_avg = summary['average_shapiro_w']
        p_avg = summary['average_p_value']
        mae_avg = summary['overall_mae']
        rmse_avg = summary['overall_rmse']
        pass_rate = f"{summary['total_tests'] - summary['rejected_count']}/{summary['total_tests']} Passed"

        self.shapiro_table.insert('', 'end', values=(
            "★ Macro Average",
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
        self.fig.patch.set_facecolor(BW_THEME['bg_input'])

        targets_to_plot = [t for t in TARGETS if f'residual_{t}' in self.eval_merged_df.columns]

        for i, ax in enumerate(axes.flat):
            if i >= len(targets_to_plot):
                ax.axis('off')
                continue

            target = targets_to_plot[i]
            res_col = f'residual_{target}'
            res_vals = self.eval_merged_df[res_col].dropna().to_numpy()
            label = TARGET_LABELS.get(target, target)

            # Pure Black & White Axes Style
            ax.set_facecolor(BW_THEME['bg_surface'])
            ax.tick_params(colors='#ffffff', labelsize=10)
            for spine in ax.spines.values():
                spine.set_color('#ffffff')

            m = next((item for item in self.eval_metrics if item.target == target), None)
            w_str = f"W = {m.shapiro_w:.4f}" if m else ""

            if plot_mode == 'distribution':
                # Black and White Histogram with White Density Curve
                counts, bins, _ = ax.hist(
                    res_vals, bins=16, density=True,
                    color='#333333', edgecolor='#ffffff', linewidth=1.0
                )
                if len(res_vals) > 1 and np.std(res_vals) > 0:
                    x_axis = np.linspace(res_vals.min(), res_vals.max(), 120)
                    pdf = stats.norm.pdf(x_axis, np.mean(res_vals), np.std(res_vals))
                    ax.plot(x_axis, pdf, color='#ffffff', linewidth=2.0, linestyle='-', label='Gaussian Fit')

                ax.set_title(f"{label}  ({w_str})", color='#ffffff', fontsize=12, fontweight='bold', fontfamily=FONT_FAMILY)
                ax.set_xlabel("Residual (m)", color='#cccccc', fontsize=11, fontfamily=FONT_FAMILY)
                ax.grid(alpha=0.2, color='#ffffff', linestyle=':')

            elif plot_mode == 'qq':
                # Normal Q-Q Plot in Black and White
                stats.probplot(res_vals, dist="norm", plot=ax)
                ax.get_lines()[0].set_marker('o')
                ax.get_lines()[0].set_markersize(4)
                ax.get_lines()[0].set_markerfacecolor('#ffffff')
                ax.get_lines()[0].set_markeredgecolor('#ffffff')
                ax.get_lines()[1].set_color('#aaaaaa')
                ax.get_lines()[1].set_linewidth(1.8)
                ax.get_lines()[1].set_linestyle('--')

                ax.set_title(f"Q-Q: {label}  ({w_str})", color='#ffffff', fontsize=12, fontweight='bold', fontfamily=FONT_FAMILY)
                ax.set_xlabel("Theoretical Normal Quantiles", color='#cccccc', fontsize=11, fontfamily=FONT_FAMILY)
                ax.set_ylabel("Residual Quantiles", color='#cccccc', fontsize=11, fontfamily=FONT_FAMILY)
                ax.grid(alpha=0.2, color='#ffffff', linestyle=':')

        self.fig.tight_layout(pad=2.2)
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
