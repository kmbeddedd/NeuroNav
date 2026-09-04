from __future__ import annotations
import argparse
import json
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
DEFAULT_REPORT = Path('results/ps08_day8/benchmark_report.json')

def load_report(path: Path) -> dict:
    report = json.loads(path.read_text(encoding='utf-8'))
    required = {'winner', 'ranking', 'models', 'evaluation_protocol'}
    missing = required - set(report)
    if missing:
        raise ValueError(f'Invalid benchmark report; missing {sorted(missing)}')
    return report

def open_path(path: Path) -> None:
    if sys.platform == 'win32':
        subprocess.Popen(['explorer', str(path.resolve())])
    elif sys.platform == 'darwin':
        subprocess.Popen(['open', str(path.resolve())])
    else:
        subprocess.Popen(['xdg-open', str(path.resolve())])

class ComparisonWindow(tk.Tk):

    def __init__(self, report_path: Path) -> None:
        super().__init__()
        self.report_path = report_path
        self.report = load_report(report_path)
        self.title('PS-08 Day-8 Model Benchmark')
        self.geometry('1120x650')
        self.minsize(930, 540)
        self.configure(bg='#0b1220')
        self._build()

    def _build(self) -> None:
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('Treeview', background='#111c30', fieldbackground='#111c30', foreground='#e8eef8', rowheight=34)
        style.configure('Treeview.Heading', background='#1f3152', foreground='white', font=('Segoe UI', 10, 'bold'))
        style.map('Treeview', background=[('selected', '#176b87')])
        header = tk.Frame(self, bg='#0b1220')
        header.pack(fill='x', padx=24, pady=(20, 12))
        tk.Label(header, text='PS-08 Day-8 Benchmark', bg='#0b1220', fg='white', font=('Segoe UI', 22, 'bold')).pack(anchor='w')
        tk.Label(header, text='Official score: mean Shapiro-Wilk W over X, Y, Z and clock residuals (higher is better)', bg='#0b1220', fg='#9fb0c8', font=('Segoe UI', 10)).pack(anchor='w', pady=(4, 0))
        winner = self.report['winner']
        winner_metrics = self.report['models'][winner]
        banner = tk.Frame(self, bg='#123c34', highlightbackground='#29c99a', highlightthickness=1)
        banner.pack(fill='x', padx=24, pady=(0, 16))
        tk.Label(banner, text='BEST MODEL', bg='#123c34', fg='#78efc9', font=('Segoe UI', 10, 'bold')).pack(side='left', padx=(18, 10), pady=14)
        tk.Label(banner, text=winner, bg='#123c34', fg='white', font=('Segoe UI', 16, 'bold')).pack(side='left', pady=12)
        tk.Label(banner, text=f"Gaussianity score: {100 * winner_metrics['average_shapiro_w']:.2f}%   •   MAE: {winner_metrics['overall_mae_m']:.4f} m", bg='#123c34', fg='#d7f8ee', font=('Segoe UI', 10)).pack(side='right', padx=18)
        columns = ('rank', 'model', 'score', 'w', 'p', 'decision', 'bias', 'std', 'mae', 'rmse')
        table = ttk.Treeview(self, columns=columns, show='headings', height=9)
        headings = {'rank': 'Rank', 'model': 'Model', 'score': 'Score', 'w': 'Avg W', 'p': 'Avg p', 'decision': 'Rejected', 'bias': '|Bias| m', 'std': 'Avg std m', 'mae': 'MAE m', 'rmse': 'RMSE m'}
        widths = {'rank': 55, 'model': 170, 'score': 85, 'w': 85, 'p': 85, 'decision': 80, 'bias': 90, 'std': 95, 'mae': 90, 'rmse': 90}
        for column in columns:
            table.heading(column, text=headings[column])
            table.column(column, width=widths[column], anchor='center', stretch=column == 'model')
        table.tag_configure('winner', background='#164b40', foreground='#ffffff')
        for model in self.report['ranking']:
            metrics = self.report['models'][model]
            table.insert('', 'end', values=(metrics['rank'], model, f"{100 * metrics['average_shapiro_w']:.2f}%", f"{metrics['average_shapiro_w']:.6f}", f"{metrics['average_p_value']:.5f}", f"{metrics['rejected_test_count']}/{metrics['normality_test_count']}", f"{metrics['mean_absolute_bias']:.4f}", f"{metrics['average_residual_std']:.4f}", f"{metrics['overall_mae_m']:.4f}", f"{metrics['overall_rmse_m']:.4f}"), tags=('winner',) if model == winner else ())
        table.pack(fill='both', expand=True, padx=24)
        reference = self.report['evaluation_protocol']['published_reference']
        footer = tk.Frame(self, bg='#0b1220')
        footer.pack(fill='x', padx=24, pady=18)
        tk.Label(footer, text=f"Published reference: W={reference['shapiro_w']:.4f}, p={reference['p_value']:.4f}, decision={reference['hypothesis_result']}  •  α=0.05", bg='#0b1220', fg='#9fb0c8', font=('Segoe UI', 9)).pack(side='left')
        ttk.Button(footer, text='Open winner Q-Q plot', command=self._open_qq).pack(side='right', padx=(8, 0))
        ttk.Button(footer, text='Open results folder', command=lambda: open_path(self.report_path.parent)).pack(side='right')

    def _open_qq(self) -> None:
        slug = self.report['winner'].lower().replace('-', '_').replace(' ', '_')
        path = self.report_path.parent / f'qq_{slug}.png'
        if not path.exists():
            messagebox.showerror('Missing plot', f'Could not find {path}')
            return
        open_path(path)

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description='Open the PS-08 model comparison window')
    parser.add_argument('--report', type=Path, default=DEFAULT_REPORT)
    parser.add_argument('--check', action='store_true', help='Validate the report without opening a GUI')
    args = parser.parse_args(argv)
    report = load_report(args.report)
    if args.check:
        print(f"Report OK. Winner: {report['winner']}. Models: {len(report['ranking'])}")
        return
    ComparisonWindow(args.report).mainloop()
if __name__ == '__main__':
    main()
