import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tkinter as tk
from gui.gui_app import NeuroNavApp
from gui.formula_tooltips import FORMULA_CARD_DEFS

def test_gui_tooltips():
    app = NeuroNavApp()
    app.geometry("1420x940+5000+5000")  # Position offscreen for headless testing
    app.update_idletasks()
    app.update()
    
    # 1. Verify formula tooltip manager initialized
    assert hasattr(app, 'formula_tooltip_mgr')
    mgr = app.formula_tooltip_mgr
    assert len(mgr._photo_card_cache) == 9
    
    # 2. Check shapiro_table headings
    sh_cols = ('w_stat', 'p_val', 'h0_res', 'bias', 'std', 'mae', 'rmse', 'r2', 'max_ae')
    for c in sh_cols:
        cfg = app.shapiro_table.heading(c)
        assert cfg['image'], f"Header {c} is missing '?' logo image!"
        print(f"Verified shapiro_table header '{c}' has logo: {cfg['image']}")
        
    # 3. Check cand_table headings (must NOT have '?' logo as requested)
    cand_cols = ('model', 'shapiro_w', 'p_value', 'h0_test', 'res_mean', 'res_std', 'mae_3d', 'mae_clk')
    for c in cand_cols:
        cfg = app.cand_table.heading(c)
        assert not cfg['image'], f"Header {c} on cand_table should NOT have '?' logo, but found: {cfg['image']}"
        print(f"Verified cand_table header '{c}' is clean (no logo): {cfg['image']}")
        
    # 4. Switch to Page 3
    app.show_page(3)
    app.update()
    
    # 5. Simulate mouse motion over 'w_stat' heading
    # Calculate column x coordinate
    col_idx = list(app.shapiro_table['columns']).index('w_stat')
    all_cols = app.shapiro_table['columns']
    x_left = sum(app.shapiro_table.column(c, 'width') for c in all_cols[:col_idx])
    w = app.shapiro_table.column('w_stat', 'width')
    event_x = x_left + w - 10  # right corner near '?' logo
    event_y = 10  # heading row
    
    fake_event = type('Event', (), {'x': event_x, 'y': event_y})()
    col_mapping = {
        'w_stat': 'w_stat',
        'p_val': 'p_val',
        'h0_res': 'h0_res',
        'bias': 'bias',
        'std': 'std',
        'mae': 'mae',
        'rmse': 'rmse',
        'r2': 'r2',
        'max_ae': 'max_ae',
    }
    mgr._on_motion(fake_event, app.shapiro_table, col_mapping)
    app.update()
    
    assert mgr.tip_window is not None
    assert mgr.tip_window.winfo_viewable()
    assert mgr.active_col == 'w_stat'
    print("Tooltip successfully displayed for 'w_stat' on hover!")
    
    # 6. Simulate mouse leave
    mgr._on_leave(fake_event, app.shapiro_table)
    app.update()
    assert not mgr.tip_window.winfo_viewable()
    print("Tooltip successfully hidden on mouse leave!")
    
    # Clean up
    app.destroy()
    print("All GUI tooltip tests passed successfully!")

if __name__ == '__main__':
    test_gui_tooltips()
