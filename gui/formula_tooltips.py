"""Formula Tooltip Manager & Visual Formula Card Generator for NeuroNav GUI.

Provides interactive '?' logos for table headers with floating formula cards
reproducing the mathematical definitions for:
- Shapiro-Wilk Normality Statistic (W)
- Shapiro-Wilk p-value (Royston Normalizing Transformation)
- Null Hypothesis Decision Rule (H0: alpha=0.05)
- Residual Bias / Mean Error (m)
- Residual Standard Deviation (m)
- Mean Absolute Error (MAE) (m)
- Root Mean Square Error (RMSE) (m)
- Coefficient of Determination (R^2 Score)
- Maximum Absolute Error (Max AE) (m)
"""
from __future__ import annotations

import io
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw, ImageFont, ImageTk


def create_help_icon(
    size: int = 16,
    bg: Tuple[int, int, int, int] | str = (148, 163, 184, 45),
    fg: Tuple[int, int, int, int] | str = (71, 85, 105, 150),
    border: Tuple[int, int, int, int] | str = (100, 116, 139, 100)
) -> Image.Image:
    """Create a crisp translucent circular '?' badge icon using supersampling."""
    scale = 4
    s = size * scale
    img = Image.new('RGBA', (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    bw = max(1, int(1.1 * scale))
    d.ellipse([bw, bw, s - bw - 1, s - bw - 1], fill=bg, outline=border, width=bw)
    try:
        font = ImageFont.truetype('segoeui.ttf', int(9.5 * scale))
    except Exception:
        font = ImageFont.load_default()
    bbox = d.textbbox((0, 0), '?', font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = (s - tw) // 2 - bbox[0]
    ty = (s - th) // 2 - bbox[1] - int(0.5 * scale)
    d.text((tx, ty), '?', fill=fg, font=font)
    return img.resize((size, size), Image.Resampling.LANCZOS)


FORMULA_CARD_DEFS = {
    'w_stat': {
        'header_tag': 'SHAPIRO-WILK NORMALITY TEST STATISTIC (W)',
        'items': [
            {'type': 'text', 'content': 'The Shapiro-Wilk statistic is:'},
            {'type': 'boxed_math', 'content': r'$W = \frac{\left(\sum_{i=1}^n a_i x_{(i)}\right)^2}{\sum_{i=1}^n (x_i - \bar{x})^2}$'},
        ],
        'footer': r'$x_{(i)}$: ordered residuals, $\bar{x}$: mean residual, $a_i$: weights from covariance matrix.'
    },
    'p_val': {
        'header_tag': 'SHAPIRO-WILK NORMALITY P-VALUE',
        'items': [
            {'type': 'text', 'content': 'Shapiro-Wilk test p-value probability:'},
            {'type': 'math_inline', 'content': r'$p = P(W \leq W_{\mathrm{obs}} \mid H_0)$'},
            {'type': 'text', 'content': 'Evaluated via Royston normalizing transformation:'},
            {'type': 'boxed_math', 'content': r'$z = \frac{(1 - W)^\gamma - \mu}{\sigma} \sim \mathcal{N}(0, 1) \rightarrow p = 1 - \Phi(z)$'},
        ],
        'footer': r'$p \geq 0.05 \rightarrow$ Residuals conform to Gaussian distribution ($H_0$ accepted).'
    },
    'h0_res': {
        'header_tag': 'NULL HYPOTHESIS DECISION RULE (α = 0.05)',
        'items': [
            {'type': 'text', 'content': 'Null Hypothesis H0: Residuals follow a Gaussian distribution'},
            {'type': 'text', 'content': 'Decision rule at significance level α = 0.05:'},
            {'type': 'boxed_math', 'content': r'$\mathrm{Result} = 0\ \mathrm{(Normal)}\ \mathrm{if}\ p \geq 0.05\quad\mathrm{else}\quad 1\ \mathrm{(Reject)}$'},
        ],
        'footer': r'Official ISRO SIH criterion: 0 (Normal) indicates successful residual calibration.'
    },
    'bias': {
        'header_tag': 'BIAS / MEAN RESIDUAL ERROR (M)',
        'items': [
            {'type': 'text', 'content': 'Formula:'},
            {'type': 'boxed_math', 'content': r'$\mathrm{Bias} = \bar{e} = \frac{1}{n} \sum_{i=1}^n e_i$'},
            {'type': 'text', 'content': 'Since:'},
            {'type': 'math_inline', 'content': r'$e_i = y_i - \hat{y}_i$'},
            {'type': 'text', 'content': 'we have:'},
            {'type': 'boxed_math', 'content': r'$\mathrm{Bias} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)$'},
        ],
        'footer': r'$y_i$: true value, $\hat{y}_i$: forecasted value at epoch $i$, $n$: number of epochs.'
    },
    'std': {
        'header_tag': 'RESIDUAL STANDARD DEVIATION (M)',
        'items': [
            {'type': 'text', 'content': 'Standard deviation tells you how widely errors are spread around mean:'},
            {'type': 'text', 'content': 'The population standard deviation formula is:'},
            {'type': 'boxed_math', 'content': r'$\sigma = \sqrt{\frac{1}{n} \sum_{i=1}^n (e_i - \bar{e})^2}$'},
            {'type': 'text', 'content': 'If sample standard deviation is being used:'},
            {'type': 'boxed_math', 'content': r'$s = \sqrt{\frac{1}{n - 1} \sum_{i=1}^n (e_i - \bar{e})^2}$'},
        ],
        'footer': r'$e_i$: residual error ($y_i - \hat{y}_i$), $\bar{e}$: bias (mean residual error).'
    },
    'mae': {
        'header_tag': 'MEAN ABSOLUTE ERROR (MAE) (M)',
        'items': [
            {'type': 'text', 'content': 'First calculate absolute error:'},
            {'type': 'math_inline', 'content': r'$|e_i| = |y_i - \hat{y}_i|$'},
            {'type': 'text', 'content': 'Then average them:'},
            {'type': 'boxed_math', 'content': r'$\mathrm{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i| = \frac{1}{n}\sum_{i=1}^n |e_i|$'},
        ],
        'footer': r'Linear metric of expected orbital error magnitude in meters.'
    },
    'rmse': {
        'header_tag': 'ROOT MEAN SQUARE ERROR (RMSE) (M)',
        'items': [
            {'type': 'text', 'content': 'First calculate squared residuals:'},
            {'type': 'math_inline', 'content': r'$e_i^2 = (y_i - \hat{y}_i)^2$'},
            {'type': 'text', 'content': 'Then average and take square root:'},
            {'type': 'boxed_math', 'content': r'$\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2} = \sqrt{\frac{1}{n}\sum_{i=1}^n e_i^2}$'},
        ],
        'footer': r'Penalizes larger trajectory outliers quadratically (meters).'
    },
    'r2': {
        'header_tag': 'COEFFICIENT OF DETERMINATION (R² SCORE)',
        'items': [
            {'type': 'text', 'content': 'Residual and Total Sum of Squares:'},
            {'type': 'math_inline', 'content': r'$\mathrm{SS}_{\mathrm{res}} = \sum_{i=1}^n (y_i - \hat{y}_i)^2, \quad \mathrm{SS}_{\mathrm{tot}} = \sum_{i=1}^n (y_i - \bar{y})^2$'},
            {'type': 'text', 'content': 'Then we have:'},
            {'type': 'boxed_math', 'content': r'$R^2 = 1 - \frac{\mathrm{SS}_{\mathrm{res}}}{\mathrm{SS}_{\mathrm{tot}}} = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$'},
        ],
        'footer': r'Proportion of orbital trajectory variance explained by the model.'
    },
    'max_ae': {
        'header_tag': 'MAXIMUM ABSOLUTE ERROR (MAX AE) (M)',
        'items': [
            {'type': 'text', 'content': 'First calculate absolute residuals:'},
            {'type': 'math_inline', 'content': r'$|e_i| = |y_i - \hat{y}_i|$'},
            {'type': 'text', 'content': 'Then find the peak single-epoch deviation:'},
            {'type': 'boxed_math', 'content': r'$\mathrm{Max\ AE} = \max_{1 \leq i \leq n} |y_i - \hat{y}_i| = \max_{1 \leq i \leq n} |e_i|$'},
        ],
        'footer': r'Worst-case single-epoch prediction error in meters.'
    }
}


def render_parameter_card_image(card_def: dict) -> Image.Image:
    """Render a formula card definition into a PIL Image with pixel-accurate layout."""
    items = card_def['items']
    footer = card_def.get('footer', '')

    pad_top = 22
    pad_bot = 20
    h_header = 30
    h_gap = 14

    item_heights = []
    for it in items:
        itype = it['type']
        if itype == 'text':
            item_heights.append(24)
        elif itype == 'math_inline':
            item_heights.append(34)
        elif itype == 'boxed_math':
            item_heights.append(66)

    h_footer = 26 if footer else 0
    total_px = pad_top + h_header + h_gap + sum(item_heights) + (len(items) - 1) * h_gap + (h_gap + h_footer if footer else 0) + pad_bot
    width_px = 560
    dpi = 120

    fig = plt.figure(figsize=(width_px / dpi, total_px / dpi), dpi=dpi)
    fig.patch.set_facecolor('#FFFFFF')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor('#FFFFFF')
    ax.axis('off')

    # Header tag in crisp dark black/slate
    y_header = 1.0 - (pad_top / total_px)
    ax.text(0.06, y_header, card_def['header_tag'], color='#0F172A', fontsize=10.5, weight='bold', va='top', fontfamily='sans-serif')

    cur_px = pad_top + h_header + h_gap
    for it, h in zip(items, item_heights):
        itype = it['type']
        center_y = 1.0 - ((cur_px + h / 2.0) / total_px)
        top_y = 1.0 - (cur_px / total_px)

        if itype == 'text':
            ax.text(0.06, top_y, it['content'], color='#1E293B', fontsize=9.5, va='top', fontfamily='sans-serif')
        elif itype == 'math_inline':
            ax.text(0.50, center_y, it['content'], color='#09090B', fontsize=11.5, ha='center', va='center')
        elif itype == 'boxed_math':
            bbox_props = dict(boxstyle='square,pad=0.45', facecolor='#F8FAFC', edgecolor='#94A3B8', linewidth=1.1)
            ax.text(0.50, center_y, it['content'], color='#09090B', fontsize=11.5, ha='center', va='center', bbox=bbox_props)

        cur_px += h + h_gap

    if footer:
        y_foot = 1.0 - ((total_px - pad_bot) / total_px)
        ax.text(0.06, y_foot, footer, color='#64748B', fontsize=8.2, va='bottom', fontfamily='sans-serif')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.04, facecolor='#FFFFFF')
    plt.close(fig)
    buf.seek(0)
    raw_img = Image.open(buf).convert('RGBA')

    # Apply 12px rounded corners with transparent border outside
    w, h = raw_img.size
    pad = 2
    radius = 12
    out_w, out_h = w + 2 * pad, h + 2 * pad

    # Canvas with key color (1, 2, 3) for transparentcolor handling
    rounded = Image.new('RGBA', (out_w, out_h), (1, 2, 3, 0))

    # Rounded mask for white card
    mask = Image.new('L', (w, h), 0)
    d_mask = ImageDraw.Draw(mask)
    d_mask.rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=255)

    raw_img.putalpha(mask)
    rounded.paste(raw_img, (pad, pad), raw_img)

    # Clean outer rounded border
    d_out = ImageDraw.Draw(rounded)
    d_out.rounded_rectangle([pad, pad, pad + w - 1, pad + h - 1], radius=radius, outline='#CBD5E1', width=1)

    return rounded


class FormulaTooltipManager:
    """Manages '?' logo placement and floating formula tooltips on Treeview tables."""

    _cached_pil_cards: Dict[str, Image.Image] = {}

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.icon_normal_pil = create_help_icon(16, bg=(148, 163, 184, 45), fg=(71, 85, 105, 150), border=(100, 116, 139, 100))
        self.icon_hover_pil = create_help_icon(16, bg=(37, 99, 235, 160), fg=(255, 255, 255, 220), border=(29, 78, 216, 190))
        self.icon_normal = ImageTk.PhotoImage(self.icon_normal_pil)
        self.icon_hover = ImageTk.PhotoImage(self.icon_hover_pil)

        self.tip_window: Optional[tk.Toplevel] = None
        self.tip_label: Optional[tk.Label] = None
        self.tip_photo_ref: Optional[ImageTk.PhotoImage] = None
        self.active_tree: Optional[ttk.Treeview] = None
        self.active_col: Optional[str] = None
        self.is_pinned: bool = False
        self._photo_card_cache: Dict[str, ImageTk.PhotoImage] = {}

        # Warm up card generator in background or on first demand
        self._preload_cards()

    def _preload_cards(self) -> None:
        """Pre-render and cache all mathematical formula cards."""
        for key, card_def in FORMULA_CARD_DEFS.items():
            if key not in self._cached_pil_cards:
                self._cached_pil_cards[key] = render_parameter_card_image(card_def)
            if key not in self._photo_card_cache:
                self._photo_card_cache[key] = ImageTk.PhotoImage(self._cached_pil_cards[key])

    def get_card_photo(self, key: str) -> Optional[ImageTk.PhotoImage]:
        """Retrieve PhotoImage for the specified formula key."""
        if key in self._photo_card_cache:
            return self._photo_card_cache[key]
        if key in FORMULA_CARD_DEFS:
            img = render_parameter_card_image(FORMULA_CARD_DEFS[key])
            self._cached_pil_cards[key] = img
            photo = ImageTk.PhotoImage(img)
            self._photo_card_cache[key] = photo
            return photo
        return None

    def attach_to_tree(self, tree: ttk.Treeview, col_mapping: Dict[str, str]) -> None:
        """Attach '?' logo and tooltip listeners to a Treeview widget.
        
        Args:
            tree: The ttk.Treeview widget.
            col_mapping: Dict mapping column ID (e.g. 'w_stat') to formula key (e.g. 'w_stat').
        """
        # Set '?' logo image on each mapped heading
        for col_id in col_mapping.keys():
            try:
                tree.heading(col_id, image=self.icon_normal)
            except Exception:
                pass

        tree.bind("<Motion>", lambda e: self._on_motion(e, tree, col_mapping), add="+")
        tree.bind("<Leave>", lambda e: self._on_leave(e, tree), add="+")
        tree.bind("<Button-1>", lambda e: self._on_click(e, tree, col_mapping), add="+")

    def _get_column_bounds(self, tree: ttk.Treeview, col_index: int) -> Tuple[int, int]:
        """Compute the left and right x-coordinates of a column in pixels."""
        all_cols = tree['columns']
        x_left = sum(tree.column(c, 'width') for c in all_cols[:col_index])
        w = tree.column(all_cols[col_index], 'width')
        return x_left, x_left + w

    def _on_motion(self, event: tk.Event, tree: ttk.Treeview, col_mapping: Dict[str, str]) -> None:
        """Handle cursor movements across table headings."""
        if self.is_pinned:
            return

        region = tree.identify_region(event.x, event.y)
        if region != 'heading':
            self._reset_hover(tree)
            self.hide_tooltip()
            return

        col_id_str = tree.identify_column(event.x)
        if not col_id_str or not col_id_str.startswith('#'):
            self._reset_hover(tree)
            self.hide_tooltip()
            return

        try:
            col_idx = int(col_id_str.replace('#', '')) - 1
            all_cols = tree['columns']
            if col_idx < 0 or col_idx >= len(all_cols):
                self._reset_hover(tree)
                self.hide_tooltip()
                return
            col_name = all_cols[col_idx]
        except Exception:
            self._reset_hover(tree)
            self.hide_tooltip()
            return

        if col_name in col_mapping:
            formula_key = col_mapping[col_name]
            x_left, x_right = self._get_column_bounds(tree, col_idx)

            # Change cursor to hand2 to indicate interactivity
            tree.configure(cursor='hand2')

            # Highlight icon if hovering in the column header
            if self.active_col != col_name:
                self._reset_hover(tree)
                try:
                    tree.heading(col_name, image=self.icon_hover)
                except Exception:
                    pass
                self.active_tree = tree
                self.active_col = col_name

            # Calculate anchor coordinates on screen
            root_x = tree.winfo_rootx() + max(x_left + 10, min(event.x - 40, x_right - 200))
            root_y = tree.winfo_rooty() + 32  # Just below header row

            self.show_tooltip(formula_key, root_x, root_y)
        else:
            self._reset_hover(tree)
            self.hide_tooltip()

    def _on_leave(self, event: tk.Event, tree: ttk.Treeview) -> None:
        """Handle cursor leaving the Treeview widget."""
        if not self.is_pinned:
            self._reset_hover(tree)
            self.hide_tooltip()

    def _on_click(self, event: tk.Event, tree: ttk.Treeview, col_mapping: Dict[str, str]) -> None:
        """Handle clicking on a parameter header to toggle pin state."""
        region = tree.identify_region(event.x, event.y)
        if region != 'heading':
            if self.is_pinned:
                self.is_pinned = False
                self._reset_hover(tree)
                self.hide_tooltip()
            return

        col_id_str = tree.identify_column(event.x)
        if not col_id_str or not col_id_str.startswith('#'):
            return

        col_idx = int(col_id_str.replace('#', '')) - 1
        all_cols = tree['columns']
        if col_idx < 0 or col_idx >= len(all_cols):
            return
        col_name = all_cols[col_idx]

        if col_name in col_mapping:
            # Toggle pin state
            self.is_pinned = not self.is_pinned
            if not self.is_pinned:
                self._reset_hover(tree)
                self.hide_tooltip()
        else:
            if self.is_pinned:
                self.is_pinned = False
                self._reset_hover(tree)
                self.hide_tooltip()

    def _reset_hover(self, tree: ttk.Treeview) -> None:
        """Reset cursor and heading icon back to default state."""
        try:
            tree.configure(cursor='')
            if self.active_col:
                tree.heading(self.active_col, image=self.icon_normal)
        except Exception:
            pass
        self.active_col = None

    def show_tooltip(self, formula_key: str, x: int, y: int) -> None:
        """Display floating tooltip card at specified screen coordinates."""
        photo = self.get_card_photo(formula_key)
        if not photo:
            return

        card_w = photo.width()
        card_h = photo.height()

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()

        # Clamp x and y to fit inside screen
        if x + card_w > screen_w - 15:
            x = screen_w - card_w - 15
        if x < 10:
            x = 10

        if y + card_h > screen_h - 40:
            # Shift above header if overflowing bottom
            y = max(10, y - card_h - 40)

        if not self.tip_window or not self.tip_window.winfo_exists():
            self.tip_window = tk.Toplevel(self.root)
            self.tip_window.overrideredirect(True)
            self.tip_window.attributes('-topmost', True)
            try:
                self.tip_window.attributes('-transparentcolor', '#010203')
                self.tip_window.attributes('-alpha', 0.98)
            except Exception:
                pass
            self.tip_window.configure(bg='#010203')

            # Dismiss when clicking on the tooltip itself or pressing Escape
            self.tip_window.bind("<Button-1>", lambda e: self.hide_tooltip(force=True))
            self.root.bind("<Escape>", lambda e: self.hide_tooltip(force=True), add="+")

            self.tip_label = tk.Label(self.tip_window, image=photo, bg='#010203', borderwidth=0)
            self.tip_label.pack()
        else:
            if self.tip_label:
                self.tip_label.configure(image=photo)

        self.tip_photo_ref = photo
        self.tip_window.geometry(f"+{int(x)}+{int(y)}")
        self.tip_window.deiconify()
        self.tip_window.lift()

    def hide_tooltip(self, force: bool = False) -> None:
        """Hide the floating formula tooltip window."""
        if not force and self.is_pinned:
            return
        self.is_pinned = False
        if self.tip_window and self.tip_window.winfo_exists():
            self.tip_window.withdraw()
