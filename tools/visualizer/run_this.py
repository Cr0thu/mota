# -*- coding: utf-8 -*-
"""
10层魔塔可视化界面 V2
支持 PPO 训练动态演示 + 手动节点选择
"""
import os
import sys
import time
import math
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from animation_environment import Mota
from environment import Terrain
from route_player import load_route, visualizer_pos_text
from stage_reward import stage_action_priors, stage_potential, transition_reward

PPO = None
compute_sword_reward = None
PPO_REWARD_RATE = None
action_prior_logits = None
MapGNNEncoder = None
TabularQLearningAgent = None
PROJECT_ROOT = BASE_DIR.parent.parent
ROUTE_SEARCH_DIRS = (
    PROJECT_ROOT / 'artifacts' / 'manual_exploration_20260524',
    PROJECT_ROOT / 'artifacts' / 'expert',
    PROJECT_ROOT / 'artifacts' / 'runs',
)
ROUTE_IGNORED_PARTS = {
    'archive',
    'archived',
    'archive_stale_20260524',
    'obsolete_expert_routes',
    'obsolete_run_dirs',
    'invalid_shop_routes',
    'stale',
    'old',
    '__pycache__',
}

COLORS = {
    'bg': '#e8eef8',
    'surface': '#ffffff',
    'surface_alt': '#f8fafc',
    'border': '#d1d5db',
    'toolbar': '#dbeafe',
    'toolbar_soft': '#ffffff',
    'toolbar_text': '#0f172a',
    'panel': '#ffffff',
    'map_bg': '#020617',
    'text': '#111827',
    'accent': '#2563eb',
    'accent_dark': '#1d4ed8',
    'success': '#059669',
    'success_dark': '#047857',
    'danger': '#dc2626',
    'danger_dark': '#b91c1c',
    'warning': '#d97706',
    'muted': '#64748b',
    'muted_bg': '#f1f5f9',
    'status': '#0f172a',
}

METRIC_TINTS = {
    'HP': '#fef2f2',
    'ATK': '#fff7ed',
    'DEF': '#eff6ff',
    'MDEF': '#f5f3ff',
    'Money': '#ecfdf5',
    'Exp': '#eef2ff',
    '黄钥匙': '#fefce8',
    '蓝钥匙': '#eff6ff',
    '红钥匙': '#fff1f2',
    '楼层': '#f8fafc',
    '阶段': '#faf5ff',
    'Phi': '#f0fdfa',
    'Q状态': '#fff7ed',
    '最近': '#fef2f2',
    '累计': '#eff6ff',
    'Base': '#f8fafc',
    '塑形': '#ecfeff',
    '回环罚': '#fffbeb',
    '终止罚': '#fff1f2',
}


class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip = None
        widget.bind('<Enter>', self.show)
        widget.bind('<Leave>', self.hide)

    def show(self, _event=None):
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 18
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f'+{x}+{y}')
        self.tip.configure(bg='#1d4ed8')
        label = tk.Label(
            self.tip,
            text=self.text,
            justify=tk.LEFT,
            bg='#ffffff',
            fg='#0f172a',
            activebackground='#ffffff',
            activeforeground='#0f172a',
            highlightbackground='#93c5fd',
            highlightcolor='#93c5fd',
            highlightthickness=1,
            relief=tk.FLAT,
            borderwidth=0,
            font=('Arial', 10),
            wraplength=320,
            padx=11,
            pady=8,
        )
        label.pack(padx=1, pady=1)

    def hide(self, _event=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


def _parent_bg(widget, fallback=COLORS['bg']):
    try:
        return widget.cget('bg')
    except tk.TclError:
        return fallback


def _rounded_rect(canvas, x1, y1, x2, y2, radius, **kwargs):
    radius = max(1, min(radius, (x2 - x1) / 2, (y2 - y1) / 2))
    points = [
        x1 + radius, y1,
        x2 - radius, y1,
        x2, y1,
        x2, y1 + radius,
        x2, y2 - radius,
        x2, y2,
        x2 - radius, y2,
        x1 + radius, y2,
        x1, y2,
        x1, y2 - radius,
        x1, y1 + radius,
        x1, y1,
    ]
    return canvas.create_polygon(points, smooth=True, splinesteps=24, **kwargs)


class FlatButton(tk.Canvas):
    """Canvas-backed rounded button; macOS Tk ignores native Button bg/fg colors."""

    def __init__(self, parent, text, command, bg, active_bg, fg, width=None):
        self.command = command
        self._text = text
        self._normal_bg = bg
        self._active_bg = active_bg
        self._normal_fg = fg
        self._current_bg = bg
        self._state = tk.NORMAL
        self._font = ('Arial', 10, 'bold')
        width_px = max(48, (width or len(text)) * 12 + 24)
        super().__init__(
            parent,
            width=width_px,
            height=34,
            bg=_parent_bg(parent),
            highlightthickness=0,
            bd=0,
            cursor='pointinghand',
        )
        self._draw()
        self.bind('<Configure>', lambda _event: self._draw())
        self.bind('<Enter>', self._on_enter)
        self.bind('<Leave>', self._on_leave)
        self.bind('<Button-1>', self._on_click)

    def _is_disabled(self):
        return str(self._state) == tk.DISABLED

    def _draw(self):
        self.delete('all')
        w = max(1, self.winfo_width())
        h = max(1, self.winfo_height())
        fill = '#9ca3af' if self._is_disabled() else self._current_bg
        fg = '#e5e7eb' if self._is_disabled() else self._normal_fg
        _rounded_rect(self, 1, 1, w - 1, h - 1, 13, fill=fill, outline='')
        self.create_text(
            w / 2,
            h / 2,
            text=self._text,
            fill=fg,
            font=self._font,
            anchor='center',
        )

    def _on_enter(self, _event=None):
        if not self._is_disabled():
            self._current_bg = self._active_bg
            self._draw()

    def _on_leave(self, _event=None):
        if not self._is_disabled():
            self._current_bg = self._normal_bg
            self._draw()

    def _on_click(self, _event=None):
        if not self._is_disabled() and self.command:
            self.command()

    def configure(self, cnf=None, **kwargs):
        cnf = cnf or {}
        if cnf:
            kwargs.update(cnf)
        if 'bg' in kwargs:
            self._normal_bg = kwargs['bg']
        if 'background' in kwargs:
            self._normal_bg = kwargs['background']
        if 'activebackground' in kwargs:
            self._active_bg = kwargs.pop('activebackground')
        if 'fg' in kwargs:
            self._normal_fg = kwargs['fg']
            kwargs.pop('fg')
        if 'foreground' in kwargs:
            self._normal_fg = kwargs['foreground']
            kwargs.pop('foreground')
        if 'text' in kwargs:
            self._text = kwargs.pop('text')
        state = kwargs.get('state')
        if 'bg' in kwargs:
            kwargs.pop('bg')
        if 'background' in kwargs:
            kwargs.pop('background')
        if 'activeforeground' in kwargs:
            kwargs.pop('activeforeground')
        if 'state' in kwargs:
            self._state = kwargs.pop('state')
        if kwargs:
            super().configure(**kwargs)
        if state == tk.DISABLED or state == 'disabled':
            self._state = tk.DISABLED
            super().configure(cursor='arrow')
        elif state == tk.NORMAL or state == 'normal':
            self._state = tk.NORMAL
            self._current_bg = self._normal_bg
            super().configure(cursor='pointinghand')
        self._draw()

    config = configure


class MetricCard(tk.Canvas):
    def __init__(self, parent, title, accent):
        self.title = title
        self.value = '--'
        self.accent = accent
        self.fill = METRIC_TINTS.get(title, COLORS['muted_bg'])
        self.outline = '#cbd5e1'
        super().__init__(
            parent,
            width=82,
            height=48,
            bg=_parent_bg(parent, COLORS['surface']),
            highlightthickness=0,
            bd=0,
        )
        self.bind('<Configure>', lambda _event: self._draw())
        self._draw()

    def _draw(self):
        self.delete('all')
        w = max(1, self.winfo_width())
        h = max(1, self.winfo_height())
        _rounded_rect(self, 1, 1, w - 1, h - 1, 12, fill=self.fill, outline=self.outline)
        _rounded_rect(self, 7, 6, w - 7, 10, 3, fill=self.accent, outline='')
        self.create_text(10, 22, text=self.title, fill='#475569',
                         font=('Arial', 7, 'bold'), anchor='w')
        self.create_text(10, 37, text=self.value, fill=COLORS['text'],
                         font=('Consolas', 11, 'bold'), anchor='w')

    def configure(self, cnf=None, **kwargs):
        cnf = cnf or {}
        if cnf:
            kwargs.update(cnf)
        if 'text' in kwargs:
            self.value = kwargs.pop('text')
        if 'bg' in kwargs:
            kwargs.pop('bg')
        if kwargs:
            super().configure(**kwargs)
        self._draw()

    config = configure


class RoundedPanel(tk.Frame):
    def __init__(self, parent, fill=COLORS['surface'], outline=COLORS['border'], radius=18, padding=3):
        self.fill = fill
        self.outline = outline
        self.radius = radius
        self.padding = padding
        super().__init__(parent, bg=_parent_bg(parent, COLORS['bg']))
        self.canvas = tk.Canvas(self, bg=_parent_bg(parent, COLORS['bg']), highlightthickness=0, bd=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.body = tk.Frame(self.canvas, bg=fill)
        self._window = self.canvas.create_window(padding, padding, anchor='nw', window=self.body)
        self.canvas.bind('<Configure>', self._on_configure)
        self.body.bind('<Configure>', self._on_body_configure)

    def _on_body_configure(self, _event=None):
        req_h = self.body.winfo_reqheight() + self.padding * 2
        if req_h > 1:
            self.canvas.configure(height=req_h)

    def _on_configure(self, event):
        w = max(1, event.width)
        h = max(1, event.height)
        self.canvas.delete('panel-bg')
        _rounded_rect(
            self.canvas,
            1,
            1,
            w - 1,
            h - 1,
            self.radius,
            fill=self.fill,
            outline=self.outline,
            tags='panel-bg',
        )
        self.canvas.tag_lower('panel-bg')
        inner_w = max(1, w - self.padding * 2)
        inner_h = max(1, h - self.padding * 2)
        self.canvas.coords(self._window, self.padding, self.padding)
        self.canvas.itemconfigure(self._window, width=inner_w, height=inner_h)


class ScrollableFrame(tk.Frame):
    def __init__(self, parent, width=520):
        super().__init__(parent, bg=COLORS['bg'])
        self.canvas = tk.Canvas(
            self,
            width=width,
            bg=COLORS['bg'],
            highlightthickness=0,
            bd=0,
        )
        self.scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.body = tk.Frame(self.canvas, bg=COLORS['bg'])
        self._window = self.canvas.create_window((0, 0), window=self.body, anchor='nw')

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.body.bind('<Configure>', self._on_body_configure)
        self.canvas.bind('<Configure>', self._on_canvas_configure)
        self.canvas.bind('<Enter>', self._bind_mousewheel)
        self.canvas.bind('<Leave>', self._unbind_mousewheel)

    def _on_body_configure(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self._window, width=event.width)

    def _bind_mousewheel(self, _event=None):
        self.canvas.bind_all('<MouseWheel>', self._on_mousewheel)
        self.canvas.bind_all('<Button-4>', self._on_linux_scroll_up)
        self.canvas.bind_all('<Button-5>', self._on_linux_scroll_down)

    def _unbind_mousewheel(self, _event=None):
        self.canvas.unbind_all('<MouseWheel>')
        self.canvas.unbind_all('<Button-4>')
        self.canvas.unbind_all('<Button-5>')

    def _on_mousewheel(self, event):
        if event.delta == 0:
            return
        if abs(event.delta) >= 240:
            units = max(-2, min(2, int(-event.delta / 120)))
        else:
            units = -1 if event.delta > 0 else 1
        self.canvas.yview_scroll(units, 'units')

    def _on_linux_scroll_up(self, _event=None):
        self.canvas.yview_scroll(-1, 'units')

    def _on_linux_scroll_down(self, _event=None):
        self.canvas.yview_scroll(1, 'units')


# ============================================================================
#  主窗口
# ============================================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('50层魔塔 - 前十层RL可视化')
        self.geometry('1500x920')
        self.minsize(1200, 780)

        # 环境
        self.env = None
        # PPO agent
        self.agent = None
        self.q_agent = None
        # 训练控制
        self.training = False
        self.q_training = False
        self.stop_training = False
        self.train_speed_ms = 40   # 每步间隔毫秒，越小越快
        self.speed_ms_var = tk.IntVar(value=self.train_speed_ms)
        self._train_after_id = None
        self._q_after_id = None
        self._route_after_id = None
        self.route_playing = False
        self.route_playback = None
        self.route_choices = {}
        self._demo_recent_positions = []
        self._demo_last_action_was_stair = False
        # 手动模式
        self.action_list = []
        self.action_iids = []
        self.action_metrics = []
        self.selected_action_idx = -1
        self._map_press_xy = None
        self.status_var = tk.StringVar(value='就绪：单击地图上的可行动作目标即可执行；右侧列表可查看说明。')
        self.view_floor_var = tk.StringVar(value='视图楼层: --')
        self.reward_total = 0.0
        self.reward_last_row = None
        self.reward_labels = {}
        self.reward_action_var = tk.StringVar(value='最近动作：--')
        self.reward_detail_text = None
        self.action_summary_var = tk.StringVar(value='动作建议：等待环境初始化。')

        self._build_ui()
        self._reset_reward_monitor()
        self._init_env()
        self._bind_shortcuts()

    # ------------------------------------------------------------------------
    #  构建界面
    # ------------------------------------------------------------------------
    def _configure_styles(self):
        style = ttk.Style(self)
        try:
            style.theme_use('clam')
        except tk.TclError:
            pass
        style.configure(
            'Mota.Treeview',
            background=COLORS['surface'],
            fieldbackground=COLORS['surface'],
            foreground=COLORS['text'],
            rowheight=28,
            borderwidth=0,
            font=('Arial', 10),
        )
        style.configure(
            'Mota.Treeview.Heading',
            background='#e2e8f0',
            foreground='#334155',
            relief='flat',
            font=('Arial', 10, 'bold'),
        )
        style.map(
            'Mota.Treeview',
            background=[('selected', COLORS['accent'])],
            foreground=[('selected', '#ffffff')],
        )
        style.configure('Vertical.TScrollbar', gripcount=0, background='#cbd5e1', troughcolor='#f1f5f9')

    def _make_button(self, parent, text, command, variant='secondary', width=None):
        palette = {
            'primary': (COLORS['accent'], COLORS['accent_dark'], '#ffffff'),
            'success': (COLORS['success'], COLORS['success_dark'], '#ffffff'),
            'danger': (COLORS['danger'], COLORS['danger_dark'], '#ffffff'),
            'dark': ('#475569', '#334155', '#ffffff'),
            'ghost': ('#eef2ff', '#dbeafe', '#1e3a8a'),
            'secondary': ('#f1f5f9', '#e2e8f0', '#111827'),
            'light': ('#ffffff', '#eff6ff', '#0f172a'),
        }
        bg, active_bg, fg = palette.get(variant, palette['secondary'])
        return FlatButton(parent, text=text, command=command, bg=bg, active_bg=active_bg, fg=fg, width=width)

    def _make_toolbar_group(
        self,
        parent,
        title=None,
        side=tk.LEFT,
        padx=(0, 10),
        fill=tk.Y,
        tint=None,
        accent=None,
    ):
        tint = tint or COLORS['toolbar_soft']
        accent = accent or COLORS['accent']
        panel = RoundedPanel(parent, fill=tint, outline='#cbd5e1', radius=18, padding=3)
        panel.pack(side=side, fill=fill, padx=padx, pady=0)
        body = panel.body
        body.configure(bg=tint)
        if title:
            tk.Frame(body, bg=accent, height=4).pack(fill=tk.X, padx=10, pady=(7, 2))
            tk.Label(
                body,
                text=title,
                bg=tint,
                fg=accent,
                font=('Arial', 8, 'bold'),
            ).pack(anchor='w', padx=10, pady=(0, 2))
        return body

    def _make_section(self, parent, title, subtitle=None, fill=tk.X, expand=False, accent=None, tint=None):
        accent = accent or COLORS['accent']
        tint = tint or '#eff6ff'
        outer = RoundedPanel(parent, fill=COLORS['surface'], outline='#cfd8e3', radius=18, padding=4)
        outer.pack(fill=fill, expand=expand, padx=10, pady=7)
        content = outer.body
        header = tk.Frame(content, bg=tint)
        header.pack(fill=tk.X, padx=8, pady=(8, 5))
        tk.Frame(header, bg=accent, width=5).pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8), pady=7)
        header_text = tk.Frame(header, bg=tint)
        header_text.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=7)
        tk.Label(header_text, text=title, bg=tint, fg=accent,
                 font=('Arial', 12, 'bold')).pack(anchor='w')
        if subtitle:
            tk.Label(header_text, text=subtitle, bg=tint, fg='#475569',
                     font=('Arial', 9), wraplength=410, justify=tk.LEFT).pack(anchor='w', pady=(2, 0))
        body = tk.Frame(content, bg=COLORS['surface'])
        body.pack(fill=tk.BOTH, expand=True, padx=12, pady=(4, 12))
        return outer, body

    def _build_ui(self):
        self.configure(bg=COLORS['bg'])
        self.option_add('*Font', 'Arial 10')
        self._configure_styles()

        # 顶部工具栏
        toolbar = tk.Frame(self, bg=COLORS['toolbar'], height=132)
        toolbar.pack(fill=tk.X, side=tk.TOP)
        toolbar.pack_propagate(False)

        title_row = tk.Frame(toolbar, bg=COLORS['toolbar'])
        title_row.pack(fill=tk.X, padx=18, pady=(10, 5))
        title_box = tk.Frame(title_row, bg=COLORS['toolbar'])
        title_box.pack(side=tk.LEFT)
        tk.Label(title_box, text='50层魔塔 · 前十层实验台', bg=COLORS['toolbar'], fg=COLORS['toolbar_text'],
                 font=('Arial', 18, 'bold')).pack(anchor='w')
        tk.Label(title_box, text='宏动作交互 / 路线回放 / Q-learning 检查 / Reward 监视',
                 bg=COLORS['toolbar'], fg='#475569', font=('Arial', 10)).pack(anchor='w', pady=(1, 0))
        title_actions = tk.Frame(title_row, bg=COLORS['toolbar'])
        title_actions.pack(side=tk.RIGHT, pady=2)
        self.help_btn = self._make_button(title_actions, '帮助', self._show_help, variant='light', width=5)
        self.help_btn.pack(side=tk.RIGHT, padx=(8, 0))
        self.player_floor_chip = tk.Label(
            title_actions,
            textvariable=self.view_floor_var,
            bg='#dbeafe',
            fg='#1e3a8a',
            padx=12,
            pady=7,
            font=('Consolas', 10, 'bold'),
        )
        self.player_floor_chip.pack(side=tk.RIGHT)

        controls_row = tk.Frame(toolbar, bg=COLORS['toolbar'])
        controls_row.pack(fill=tk.X, padx=18, pady=(0, 10))

        mode_box = self._make_toolbar_group(controls_row, '模式', tint='#eef2ff', accent='#4f46e5')
        self.mode_var = tk.StringVar(value='manual')
        self.mode_buttons = {}
        mode_row = tk.Frame(mode_box, bg='#eef2ff')
        mode_row.pack(anchor='w', padx=8, pady=(0, 8))
        for text, value, width in [('手动', 'manual', 5), ('PPO', 'train', 5), ('Q学习', 'q', 6), ('贪婪', 'demo', 5)]:
            btn = self._make_button(mode_row, text, lambda v=value: self._set_mode(v), variant='ghost', width=width)
            btn.pack(side=tk.LEFT, padx=2)
            self.mode_buttons[value] = btn

        speed_box = self._make_toolbar_group(controls_row, '演示延迟', tint='#f0fdfa', accent='#0d9488')
        speed_row = tk.Frame(speed_box, bg='#f0fdfa')
        speed_row.pack(anchor='w', padx=8, pady=(0, 8))
        self.speed_spin = tk.Spinbox(
            speed_row,
            from_=0,
            to=1000,
            increment=20,
            width=5,
            textvariable=self.speed_ms_var,
            command=self._on_speed_change,
            relief=tk.FLAT,
            bd=0,
            justify='center',
            bg='#f8fafc',
            fg=COLORS['text'],
            insertbackground=COLORS['text'],
        )
        self.speed_spin.pack(side=tk.LEFT, padx=(0, 4), ipady=4)
        self.speed_spin.bind('<Return>', self._on_speed_change)
        self.speed_buttons = {}
        for label, value in [('快', 0), ('中', 40), ('慢', 160)]:
            btn = self._make_button(speed_row, label, lambda v=value: self._set_speed(v), variant='ghost', width=3)
            btn.pack(side=tk.LEFT, padx=1)
            self.speed_buttons[value] = btn

        control_box = self._make_toolbar_group(controls_row, '操作', fill=tk.Y, tint='#fff1f2', accent='#e11d48')
        control_row = tk.Frame(control_box, bg='#fff1f2')
        control_row.pack(anchor='w', padx=8, pady=(0, 8))
        self.reset_btn = self._make_button(control_row, '重置', self._reset_env, variant='danger', width=5)
        self.reset_btn.pack(side=tk.LEFT, padx=2)
        self.undo_btn = self._make_button(control_row, '回退', self._back_step, variant='ghost', width=5)
        self.undo_btn.pack(side=tk.LEFT, padx=2)

        view_box = self._make_toolbar_group(controls_row, '视图', fill=tk.Y, tint='#fffbeb', accent='#d97706')
        view_row = tk.Frame(view_box, bg='#fffbeb')
        view_row.pack(anchor='w', padx=8, pady=(0, 8))
        self.floor_down_btn = self._make_button(view_row, '下层', self._floor_down, variant='ghost', width=5)
        self.floor_down_btn.pack(side=tk.LEFT, padx=2)
        self.floor_up_btn = self._make_button(view_row, '上层', self._floor_up, variant='ghost', width=5)
        self.floor_up_btn.pack(side=tk.LEFT, padx=2)
        self.player_floor_btn = self._make_button(view_row, '勇士层', self._go_player_floor, variant='ghost', width=6)
        self.player_floor_btn.pack(side=tk.LEFT, padx=2)
        self.line_btn = self._make_button(view_row, '隐藏连线', self._toggle_lines, variant='ghost', width=7)
        self.line_btn.pack(side=tk.LEFT, padx=2)

        hint = tk.Label(
            controls_row,
            text='地图目标单击即执行；楼层按钮只切换观察视图。',
            bg=COLORS['toolbar'],
            fg='#475569',
            anchor='e',
            justify=tk.RIGHT,
            font=('Arial', 9),
            wraplength=270,
        )
        hint.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))

        ToolTip(self.floor_down_btn, '只切换左侧地图视图，不会移动勇士。')
        ToolTip(self.floor_up_btn, '只切换左侧地图视图，不会移动勇士。')
        ToolTip(self.player_floor_btn, '把地图视图切回勇士当前所在楼层。')
        ToolTip(self.line_btn, '蓝线表示当前图上节点连接关系，可隐藏以便看地图。')
        ToolTip(self.reset_btn, '重置到 10 层魔塔初始状态。')
        ToolTip(self.undo_btn, '撤销上一步宏动作，快捷键 Backspace 或 Cmd+Z。')

        # 主体左右分栏
        main = tk.Frame(self, bg=COLORS['bg'])
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧地图
        left_frame = tk.Frame(main, bg=COLORS['map_bg'], bd=0, highlightthickness=1,
                              highlightbackground='#2563eb')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        map_header = tk.Frame(left_frame, bg='#1e3a8a', height=46)
        map_header.grid(row=0, column=0, sticky='we')
        map_header.grid_propagate(False)
        tk.Label(map_header, text='地图视图', bg='#1e3a8a', fg='#ffffff',
                 font=('Arial', 12, 'bold')).pack(side=tk.LEFT, padx=14)
        tk.Label(map_header, textvariable=self.view_floor_var, bg='#1e3a8a',
                 fg='#bfdbfe', font=('Consolas', 11, 'bold')).pack(side=tk.RIGHT, padx=14)
        map_canvas_frame = tk.Frame(left_frame, bg=COLORS['map_bg'])
        map_canvas_frame.grid(row=1, column=0, sticky='nsew', padx=10, pady=10)
        left_frame.rowconfigure(1, weight=1)
        left_frame.columnconfigure(0, weight=1)
        self.map_frame = map_canvas_frame

        # 右侧面板
        right_shell = tk.Frame(main, width=580, bg=COLORS['bg'])
        right_shell.pack(side=tk.RIGHT, fill=tk.Y)
        right_shell.pack_propagate(False)
        self.right_scroll = ScrollableFrame(right_shell, width=570)
        self.right_scroll.pack(fill=tk.BOTH, expand=True)
        right_frame = self.right_scroll.body

        # --- 状态区 ---
        _state_outer, state_frame = self._make_section(
            right_frame,
            '角色状态',
            '关键资源实时变化，用于判断当前路线质量。',
            accent='#0ea5e9',
            tint='#eff6ff',
        )
        self.state_labels = {}
        state_items = [
            ('HP', '#ef4444'), ('ATK', '#f59e0b'), ('DEF', '#0ea5e9'), ('MDEF', '#8b5cf6'),
            ('Money', '#10b981'), ('Exp', '#6366f1'), ('黄钥匙', '#ca8a04'), ('蓝钥匙', '#2563eb'),
            ('红钥匙', '#dc2626'), ('楼层', '#475569'), ('阶段', '#7c3aed'), ('Phi', '#14b8a6'),
            ('Q状态', '#f97316'),
        ]
        state_columns = 5
        for i, (name, accent) in enumerate(state_items):
            card = MetricCard(state_frame, name, accent)
            card.grid(row=i // state_columns, column=i % state_columns, sticky='we', padx=3, pady=3)
            state_frame.columnconfigure(i % state_columns, weight=1, uniform='state')
            self.state_labels[name] = card

        # --- Reward 监视器 ---
        _reward_outer, reward_body = self._make_section(
            right_frame,
            'Reward 监视器',
            '显示最近一步实际 reward，重点检查楼梯循环是否被正反馈。',
            accent='#f97316',
            tint='#fff7ed',
        )
        reward_grid = tk.Frame(reward_body, bg=COLORS['surface'])
        reward_grid.pack(fill=tk.X)
        reward_items = [
            ('最近', 'last', '#ef4444'),
            ('累计', 'total', '#2563eb'),
            ('Base', 'base', '#64748b'),
            ('塑形', 'shape', '#0ea5e9'),
            ('回环罚', 'stair_penalty', '#d97706'),
            ('终止罚', 'timeout_penalty', '#dc2626'),
        ]
        for i, (title, key, accent) in enumerate(reward_items):
            card = MetricCard(reward_grid, title, accent)
            card.grid(row=i // 3, column=i % 3, sticky='we', padx=3, pady=3)
            reward_grid.columnconfigure(i % 3, weight=1, uniform='reward')
            self.reward_labels[key] = card
        tk.Label(
            reward_body,
            textvariable=self.reward_action_var,
            bg=COLORS['surface'],
            fg=COLORS['text'],
            anchor='w',
            justify=tk.LEFT,
            font=('Arial', 9, 'bold'),
            wraplength=400,
        ).pack(fill=tk.X, pady=(6, 2))
        self.reward_detail_text = tk.Text(
            reward_body,
            height=4,
            wrap=tk.WORD,
            font=('Consolas', 9),
            bg=COLORS['muted_bg'],
            fg=COLORS['text'],
            relief=tk.FLAT,
            bd=0,
            padx=8,
            pady=6,
        )
        self.reward_detail_text.pack(fill=tk.X)
        self.reward_detail_text.config(state=tk.DISABLED)

        # --- 手动模式面板 ---
        self.manual_panel, manual_body = self._make_section(
            right_frame,
            '手动控制',
            '单击列表查看细节；单击地图目标、按 Enter 或点执行按钮来执行。',
            fill=tk.BOTH,
            expand=True,
            accent='#7c3aed',
            tint='#faf5ff',
        )

        manual_hint = tk.Label(
            manual_body,
            text='地图上单击可行动作目标会直接执行；列表单击只负责查看说明，避免误走。',
            justify=tk.LEFT,
            anchor='w',
            fg=COLORS['muted'],
            bg=COLORS['surface'],
            wraplength=400,
        )
        manual_hint.pack(fill=tk.X, pady=(0, 6))

        route_frame = tk.Frame(manual_body, bg='#ecfeff', highlightbackground='#67e8f9',
                               highlightthickness=1)
        route_frame.pack(fill=tk.X, pady=(0, 8))
        tk.Label(route_frame, text='本地算法路线回放', bg='#ecfeff', fg='#0e7490',
                 font=('Arial', 10, 'bold')).pack(anchor='w', padx=9, pady=(8, 2))
        self.route_status_var = tk.StringVar(value='未加载路线')
        tk.Label(route_frame, textvariable=self.route_status_var, bg='#ecfeff',
                 fg='#155e75', anchor='w', justify=tk.LEFT,
                 wraplength=390).pack(fill=tk.X, padx=9, pady=(0, 6))
        route_select = tk.Frame(route_frame, bg='#ecfeff')
        route_select.pack(fill=tk.X, padx=8, pady=(0, 6))
        self.route_choice_var = tk.StringVar(value='')
        self.route_combo = ttk.Combobox(
            route_select,
            textvariable=self.route_choice_var,
            state='readonly',
            height=16,
            style='Mota.TCombobox',
        )
        self.route_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.route_combo.bind('<<ComboboxSelected>>', self._on_route_selected)
        self.refresh_routes_btn = self._make_button(route_select, '刷新', self._refresh_route_choices,
                                                    variant='ghost', width=5)
        self.refresh_routes_btn.pack(side=tk.LEFT, padx=(6, 0))
        route_btns = tk.Frame(route_frame, bg='#ecfeff')
        route_btns.pack(fill=tk.X, padx=8, pady=(0, 8))
        self.sync_start_btn = self._make_button(route_btns, '同步算法起点', self._sync_algorithm_start,
                                                variant='primary', width=10)
        self.sync_start_btn.pack(side=tk.LEFT, padx=(0, 4))
        self.route_step_btn = self._make_button(route_btns, '单步路线', self._route_step,
                                                variant='ghost', width=8)
        self.route_step_btn.pack(side=tk.LEFT, padx=4)
        self.route_play_btn = self._make_button(route_btns, '播放路线', self._toggle_route_playback,
                                                variant='success', width=8)
        self.route_play_btn.pack(side=tk.LEFT, padx=(4, 0))
        ToolTip(self.sync_start_btn, '重置到主算法常用起点：2F 小偷剧情后，HP=400，金币=4。')
        ToolTip(self.refresh_routes_btn, '重新扫描 artifacts/manual_exploration_20260524、artifacts/expert 和 artifacts/runs 下的 JSONL 宏动作路线。')
        ToolTip(self.route_combo, '自动读取本地路线；选中后会直接加载。')
        ToolTip(self.route_step_btn, '按路线文件执行下一步宏动作。')
        ToolTip(self.route_play_btn, '按顶部演示延迟连续播放路线；再次点击可暂停。')
        self._refresh_route_choices(silent=True)

        # 动作列表 Treeview
        action_summary = tk.Label(
            manual_body,
            textvariable=self.action_summary_var,
            justify=tk.LEFT,
            anchor='w',
            fg=COLORS['text'],
            bg='#fff7ed',
            wraplength=410,
            padx=9,
            pady=7,
            highlightbackground='#fed7aa',
            highlightthickness=1,
        )
        action_summary.pack(fill=tk.X, pady=(0, 7))
        cols = ('序号', '坐标', '类别', 'ID', 'Q', 'R', 'Path', 'Δ伤', '说明')
        self.action_tree = ttk.Treeview(manual_body, columns=cols, show='headings',
                                        height=13, style='Mota.Treeview')
        for c in cols:
            self.action_tree.heading(c, text=c)
            if c == '序号':
                self.action_tree.column(c, width=38, anchor='center')
            elif c == '说明':
                self.action_tree.column(c, width=116, anchor='w')
            elif c in {'Q', 'R'}:
                self.action_tree.column(c, width=52, anchor='e')
            elif c in {'Path', 'Δ伤'}:
                self.action_tree.column(c, width=46, anchor='e')
            else:
                self.action_tree.column(c, width=60, anchor='center')
        self.action_tree.tag_configure('enemy', background='#fef2f2', foreground='#7f1d1d')
        self.action_tree.tag_configure('item', background='#ecfdf5', foreground='#064e3b')
        self.action_tree.tag_configure('terrain', background='#eff6ff', foreground='#1e3a8a')
        self.action_tree.tag_configure('story', background='#faf5ff', foreground='#581c87')
        self.action_tree.tag_configure('q_best', background='#dcfce7', foreground='#064e3b')
        self.action_tree.tag_configure('r_bad', background='#fff7ed', foreground='#9a3412')
        self.action_tree.tag_configure('other', background=COLORS['surface'], foreground=COLORS['text'])
        self.action_tree.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        self.action_tree.bind('<<TreeviewSelect>>', self._on_action_select)
        self.action_tree.bind('<Return>', lambda _event: self._execute_selected())

        btn_frm = tk.Frame(manual_body, bg=COLORS['surface'])
        btn_frm.pack(fill=tk.X, pady=(0, 8))
        self.execute_btn = self._make_button(btn_frm, '执行选中', self._execute_selected, variant='success')
        self.execute_btn.pack(side=tk.LEFT, padx=(0, 6), fill=tk.X, expand=True)
        self.refresh_btn = self._make_button(btn_frm, '刷新', self._refresh_action_list, variant='secondary')
        self.refresh_btn.pack(side=tk.LEFT, padx=3)
        self.back_btn = self._make_button(btn_frm, '回退一步', self._back_step, variant='ghost')
        self.back_btn.pack(side=tk.LEFT, padx=(3, 0))

        detail_frame = tk.Frame(manual_body, bg='#f5f3ff', highlightbackground='#c4b5fd',
                                highlightthickness=1)
        detail_frame.pack(fill=tk.X)
        tk.Label(detail_frame, text='选中动作说明', bg='#f5f3ff', fg='#5b21b6',
                 font=('Arial', 10, 'bold')).pack(anchor='w', padx=9, pady=(8, 0))
        self.action_detail = tk.Text(detail_frame, height=6, wrap=tk.WORD, font=('Arial', 10),
                                     bg='#f5f3ff', fg=COLORS['text'], relief=tk.FLAT,
                                     bd=0, padx=8, pady=6)
        self.action_detail.pack(fill=tk.X, padx=1, pady=(2, 8))
        self.action_detail.config(state=tk.DISABLED)

        # --- 训练/演示面板 ---
        self.train_panel, train_body = self._make_section(
            right_frame,
            'PPO 训练',
            '队友版本目前主要用于拿 5F 剑的演示，不代表完整前十层策略。',
            fill=tk.BOTH,
            expand=True,
            accent='#059669',
            tint='#ecfdf5',
        )
        # 默认隐藏，切换模式时显示

        train_hint = tk.Label(
            train_body,
            text='当前队友版本的 PPO 目标主要是“拿 5F 剑”，不是完整击败 10F 骷髅队长。'
                 '训练/演示会加载 torch，可能需要等待。',
            justify=tk.LEFT,
            anchor='w',
            fg=COLORS['muted'],
            bg=COLORS['surface'],
            wraplength=400,
        )
        train_hint.grid(row=0, column=0, columnspan=2, sticky='we', pady=(0, 8))

        tk.Label(train_body, text='训练回合数', bg=COLORS['surface'], fg=COLORS['text'],
                 font=('Arial', 10, 'bold')).grid(row=1, column=0, sticky='w', pady=4)
        self.rounds_var = tk.StringVar(value='500')
        tk.Entry(train_body, textvariable=self.rounds_var, width=12, relief=tk.FLAT,
                 bg=COLORS['muted_bg']).grid(row=1, column=1, sticky='we', padx=(8, 0), ipady=5)

        tk.Label(train_body, text='模型路径', bg=COLORS['surface'], fg=COLORS['text'],
                 font=('Arial', 10, 'bold')).grid(row=2, column=0, sticky='w', pady=4)
        self.model_path_var = tk.StringVar(value='model/ppo_10floor.pth')
        tk.Entry(train_body, textvariable=self.model_path_var, width=28, relief=tk.FLAT,
                 bg=COLORS['muted_bg']).grid(row=2, column=1, sticky='we', padx=(8, 0), ipady=5)
        train_body.columnconfigure(1, weight=1)

        tk.Label(train_body, text='Q表路径', bg=COLORS['surface'], fg=COLORS['text'],
                 font=('Arial', 10, 'bold')).grid(row=3, column=0, sticky='w', pady=4)
        self.q_model_path_var = tk.StringVar(value=str(PROJECT_ROOT / 'artifacts/runs/visualizer_qlearning/q_table_quality_shield_v4.json'))
        tk.Entry(train_body, textvariable=self.q_model_path_var, width=28, relief=tk.FLAT,
                 bg=COLORS['muted_bg']).grid(row=3, column=1, sticky='we', padx=(8, 0), ipady=5)

        btn_frm2 = tk.Frame(train_body, bg=COLORS['surface'])
        btn_frm2.grid(row=4, column=0, columnspan=2, pady=(10, 4), sticky='we')
        self.train_btn = self._make_button(btn_frm2, '开始训练', self._start_train_generator,
                                           variant='success', width=10)
        self.train_btn.pack(side=tk.LEFT, padx=3)
        self._make_button(btn_frm2, '停止训练', self._stop_training, variant='danger',
                          width=10).pack(side=tk.LEFT, padx=3)
        self._make_button(btn_frm2, '贪婪演示', self._start_demo, variant='primary',
                          width=10).pack(side=tk.LEFT, padx=3)

        q_btn_frm = tk.Frame(train_body, bg=COLORS['surface'])
        q_btn_frm.grid(row=5, column=0, columnspan=2, pady=(2, 8), sticky='we')
        self.q_train_btn = self._make_button(q_btn_frm, 'Q训练', self._start_q_train_generator,
                                             variant='success', width=8)
        self.q_train_btn.pack(side=tk.LEFT, padx=3)
        self._make_button(q_btn_frm, 'Q停止', self._stop_q_training, variant='danger',
                          width=8).pack(side=tk.LEFT, padx=3)
        self._make_button(q_btn_frm, 'Q贪婪演示', self._start_q_demo, variant='primary',
                          width=10).pack(side=tk.LEFT, padx=3)
        self._make_button(q_btn_frm, '刷新Q值', self._refresh_action_list, variant='ghost',
                          width=8).pack(side=tk.LEFT, padx=3)
        ToolTip(self.q_train_btn, '使用 tabular Q-learning 训练当前宏动作环境，并把 Q(s,a) 显示在手动动作表里。')

        self.train_log = scrolledtext.ScrolledText(train_body, height=10, width=40, font=('Consolas', 9),
                                                   bg='#0f172a', fg='#e2e8f0',
                                                   insertbackground='#e2e8f0', relief=tk.FLAT)
        self.train_log.grid(row=6, column=0, columnspan=2, pady=5, sticky='nsew')
        train_body.rowconfigure(6, weight=1)

        # 训练进度标签
        self.train_status = tk.Label(train_body, text='就绪', fg=COLORS['muted'],
                                     bg=COLORS['surface'], font=('Arial', 10, 'bold'))
        self.train_status.grid(row=7, column=0, columnspan=2, sticky='w', pady=(4, 0))

        status = tk.Label(self, textvariable=self.status_var, anchor='w',
                          bg=COLORS['status'], fg='#ffffff', padx=14, pady=7,
                          font=('Arial', 10))
        status.pack(fill=tk.X, side=tk.BOTTOM)

        # 默认显示手动面板
        self._switch_mode()
        self._update_speed_buttons()

    # ------------------------------------------------------------------------
    #  初始化环境
    # ------------------------------------------------------------------------
    def _init_env(self):
        self.env = Mota(self.map_frame)
        self.env.build_env('10層魔塔')
        self.env.create_nodes()
        self.env.create_map()
        self.env.anima_frame.grid(row=0, column=0, sticky='nsew')
        self.env.build_anima_frame(bg='#333333')
        self.env.anima_frame.canvas.bind('<ButtonPress-1>', self._on_map_press, add='+')
        self.env.anima_frame.canvas.bind('<ButtonRelease-1>', self._on_map_click, add='+')
        self._refresh_state()
        self._refresh_action_list()
        self._update_view_controls()

    def _refresh_state(self):
        p = self.env.player
        z, y, x = self.env.n2p[self.env.observation[-1]]
        try:
            phi = stage_potential(self.env)
            stage_text = phi.stage
            phi_text = self._format_metric(phi.total)
        except Exception:
            stage_text = '--'
            phi_text = '--'
        q_text = '--'
        if self.q_agent is not None:
            q_text = self._compact_count(len(getattr(self.q_agent, 'q', {})))
        values = {
            'HP': f'{p.hp}', 'ATK': f'{p.atk}', 'DEF': f'{p.def_}',
            'MDEF': f'{p.mdef}', 'Money': f'{p.money}', 'Exp': f'{p.exp}',
            '黄钥匙': f'{p.items.get("yellowKey", 0)}',
            '蓝钥匙': f'{p.items.get("blueKey", 0)}',
            '红钥匙': f'{p.items.get("redKey", 0)}',
            '楼层': f'{z + 1}F',
            '阶段': stage_text,
            'Phi': phi_text,
            'Q状态': q_text,
        }
        for name, val in values.items():
            self.state_labels[name].config(text=val)

    def _compact_count(self, value):
        value = int(value)
        if value >= 1_000_000:
            return f'{value / 1_000_000:.1f}M'
        if value >= 1_000:
            return f'{value / 1_000:.1f}k'
        return str(value)

    # ------------------------------------------------------------------------
    #  Reward 监视器
    # ------------------------------------------------------------------------
    def _reset_reward_monitor(self, note='等待动作执行；选择动作时可在说明框看到即时 reward 估计。'):
        self.reward_total = 0.0
        self.reward_last_row = None
        self.reward_action_var.set('最近动作：--')
        self._update_reward_monitor(note=note)

    def _update_reward_monitor(self, note=None):
        if not getattr(self, 'reward_labels', None):
            return
        row = self.reward_last_row or {}
        values = {
            'last': row.get('total_reward', 0.0),
            'total': self.reward_total,
            'base': row.get('base_reward', 0.0),
            'shape': row.get('shape_reward', 0.0),
            'stair_penalty': -row.get('stair_penalty', 0.0),
            'timeout_penalty': -row.get('timeout_penalty', 0.0),
        }
        for key, value in values.items():
            if key in self.reward_labels:
                self.reward_labels[key].config(text=self._format_reward_value(value))
        if row:
            self.reward_action_var.set(
                f'最近动作：{row.get("source", "手动")} · {row.get("action_label", "--")} · '
                f'{row.get("before_pos", "--")} -> {row.get("after_pos", "--")}'
            )
            detail = (
                f'total={row["total_reward"]:.2f} | base={row["base_reward"]:.2f} | '
                f'shape={row["shape_reward"]:.2f} | loop={-row["stair_penalty"]:.2f} | '
                f'timeout={-row["timeout_penalty"]:.2f}\n'
                f'ending={row.get("ending", "--")} | stage={row.get("stage_before", "--")} -> '
                f'{row.get("stage_after", "--")}\n'
                f'{self._format_reward_components(row.get("reward_components"))}'
            )
        else:
            detail = note or '等待动作执行。'
        if self.reward_detail_text is not None:
            self.reward_detail_text.config(state=tk.NORMAL)
            self.reward_detail_text.delete('1.0', tk.END)
            self.reward_detail_text.insert(tk.END, detail)
            self.reward_detail_text.config(state=tk.DISABLED)

    def _format_reward_value(self, value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return '--'
        if abs(value) >= 1000:
            return f'{value:.0f}'
        if abs(value) >= 100:
            return f'{value:.1f}'
        return f'{value:.2f}'

    def _format_metric(self, value):
        if value is None:
            return '--'
        try:
            value = float(value)
        except (TypeError, ValueError):
            return '--'
        if abs(value) >= 1000:
            return f'{value:.0f}'
        if abs(value) >= 100:
            return f'{value:.1f}'
        return f'{value:.2f}'

    def _format_reward_components(self, components):
        if not components:
            return '说明：楼梯只作为路径连接边，不给正负 reward；回环问题由动作 mask 处理。'
        top = sorted(components.items(), key=lambda pair: abs(pair[1]), reverse=True)[:5]
        return 'components: ' + ', '.join(f'{key}={value:.1f}' for key, value in top)

    def _base_reward(self, before_state, after_state, ending):
        if ending == 'stop':
            return -9999.0
        before = np.asarray(before_state, dtype=np.float32)
        after = np.asarray(after_state, dtype=np.float32)
        if PPO_REWARD_RATE is None:
            rate = np.zeros_like(after, dtype=np.float32)
        else:
            rate = np.asarray(PPO_REWARD_RATE, dtype=np.float32)
            if rate.shape[0] != after.shape[0]:
                padded = np.zeros_like(after, dtype=np.float32)
                n = min(rate.shape[0], after.shape[0])
                padded[:n] = rate[:n]
                rate = padded
        return float(np.sum((after - before) * rate))

    def _is_stair_action(self, action):
        return getattr(action, 'id', '') in {'upFloor', 'downFloor'}

    def _local_sword_reward_before_penalty(self, action, base_reward, ending, before_pos, after_pos):
        # Mirrors tools/visualizer/PPO.py::compute_sword_reward without importing torch.
        sword_pos = (4, 11, 11)
        reward = float(base_reward)
        action_pos = self.env.n2p.get(action, before_pos)
        action_pos = action_pos[:3]
        before_pos = before_pos[:3]
        after_pos = after_pos[:3]

        if self._is_stair_action(action):
            return reward
        if action_pos == sword_pos:
            reward += 10000.0
        if after_pos[0] == sword_pos[0]:
            dist = abs(after_pos[1] - sword_pos[1]) + abs(after_pos[2] - sword_pos[2])
            reward += max(0.0, 30.0 - dist * 2.0)
        reward -= 10.0
        return reward

    def _make_reward_row(
        self,
        action,
        before_state,
        after_state,
        ending,
        before_pos,
        after_pos,
        source='手动',
        base_reward=None,
        total_reward=None,
        stair_penalty=0.0,
        timeout_penalty=0.0,
        reward_components=None,
        stage_before=None,
        stage_after=None,
        phi_before=None,
        phi_after=None,
    ):
        before_pos = before_pos[:3]
        after_pos = after_pos[:3]
        if base_reward is None:
            base_reward = self._base_reward(before_state, after_state, ending)
        reward_before_penalty = self._local_sword_reward_before_penalty(
            action, base_reward, ending, before_pos, after_pos
        )
        if total_reward is None:
            total_reward = reward_before_penalty - stair_penalty - timeout_penalty
            shape_reward = reward_before_penalty - base_reward
        else:
            shape_reward = total_reward - base_reward + stair_penalty + timeout_penalty
        return {
            'source': source,
            'action_label': f'{getattr(action, "class_", "?")}:{getattr(action, "id", "?")}',
            'before_pos': self._format_pos(before_pos),
            'after_pos': self._format_pos(after_pos),
            'ending': ending,
            'base_reward': float(base_reward),
            'shape_reward': float(shape_reward),
            'stair_penalty': float(stair_penalty),
            'timeout_penalty': float(timeout_penalty),
            'total_reward': float(total_reward),
            'reward_components': reward_components or {},
            'stage_before': stage_before,
            'stage_after': stage_after,
            'phi_before': phi_before,
            'phi_after': phi_after,
        }

    def _stage_reward_info(self, action, before_state, after_state, ending, before_phi):
        try:
            return transition_reward(self.env, action, before_state, after_state, ending, before_phi)
        except Exception as exc:
            self._log(f'[Reward] 阶段 reward 计算失败，回退到旧 reward: {exc}')
            return None

    def _make_stage_reward_row(self, action, before_state, after_state, ending, before_pos, after_pos, before_phi, source):
        info = self._stage_reward_info(action, before_state, after_state, ending, before_phi)
        if info is None:
            return self._make_reward_row(action, before_state, after_state, ending, before_pos, after_pos, source=source)
        return self._make_reward_row(
            action,
            before_state,
            after_state,
            ending,
            before_pos,
            after_pos,
            source=source,
            base_reward=info.components.get('env_step', 0.0),
            total_reward=info.total,
            reward_components=info.components,
            stage_before=info.before.stage,
            stage_after=info.after.stage,
            phi_before=info.before.total,
            phi_after=info.after.total,
        )

    def _record_reward(self, row):
        self.reward_last_row = row
        self.reward_total += row['total_reward']
        self._update_reward_monitor()

    def _estimate_action_reward(self, action):
        if self.env is None or action is None:
            return None
        before = self.env.get_player_state().copy()
        before_pos = self.env.n2p[self.env.observation[-1]][:3]
        before_phi = stage_potential(self.env)
        messages = list(getattr(self.env, 'last_messages', []))
        stepped = False
        try:
            ending = self.env.step(action, refresh_frame=False)
            stepped = True
            after = self.env.get_player_state().copy()
            after_pos = self.env.n2p[self.env.observation[-1]][:3]
            row = self._make_stage_reward_row(
                action,
                before,
                after,
                ending,
                before_pos,
                after_pos,
                before_phi,
                source='选中估计',
            )
        except Exception:
            row = None
        finally:
            if stepped:
                try:
                    self.env.back_step(1)
                except Exception:
                    pass
            self.env.last_messages = messages
        return row

    def _format_reward_estimate(self, row):
        if not row:
            return 'Reward 估计：暂时无法计算。'
        return (
            f'Reward 估计：total {row["total_reward"]:.2f} = '
            f'base {row["base_reward"]:.2f} + shape {row["shape_reward"]:.2f} '
            f'- loop {row["stair_penalty"]:.2f} - timeout {row["timeout_penalty"]:.2f}'
        )

    def _log(self, msg):
        self.train_log.insert(tk.END, msg + '\n')
        self.train_log.see(tk.END)

    def _set_status(self, msg):
        self.status_var.set(msg)
        try:
            self._log(msg)
        except Exception:
            pass

    def _bind_shortcuts(self):
        self.bind_all('<Return>', lambda _event: self._execute_selected())
        self.bind_all('<BackSpace>', lambda _event: self._back_step())
        self.bind_all('<Command-z>', lambda _event: self._back_step())
        self.bind_all('<Control-z>', lambda _event: self._back_step())
        self.bind_all('<Key-r>', lambda _event: self._reset_env())
        self.bind_all('<Key-R>', lambda _event: self._reset_env())
        self.bind_all('<Key-l>', lambda _event: self._toggle_lines())
        self.bind_all('<Key-L>', lambda _event: self._toggle_lines())
        self.bind_all('<Key-plus>', lambda _event: self._floor_up())
        self.bind_all('<Key-equal>', lambda _event: self._floor_up())
        self.bind_all('<Key-minus>', lambda _event: self._floor_down())
        self.bind_all('<Key-question>', lambda _event: self._show_help())

    def _show_help(self):
        messagebox.showinfo(
            '操作说明',
            '这个窗口使用“宏动作”，不是键盘方向键逐格移动。\n\n'
            '手动操作：\n'
            '1. 右侧列表显示当前可达且可触发的目标。\n'
            '2. 单击一行动作，会在地图上用光标标出目标，并在下方显示说明。\n'
            '3. 单击地图上的可行动作目标格即可执行。\n'
            '4. 也可以按 Enter，或点击“执行选中”来执行右侧选中的动作。\n'
            '5. 顶部“回退”或 Backspace/Cmd+Z 可撤销上一步。\n\n'
            '顶部按钮：\n'
            '- 查看上层/查看下层：只切换观察视图，不移动勇士。\n'
            '- 回到勇士楼层：把视图切回勇士当前所在楼层。\n'
            '- 隐藏/显示连线：切换蓝色连接线。\n\n'
            '快捷键：Enter 执行选中动作，Backspace/Cmd+Z 回退，R 重置，L 切换连线，+/- 查看楼层。',
        )

    def _ensure_rl_components(self):
        """Load torch-based PPO code only when the user starts training/demo."""
        global PPO, compute_sword_reward, PPO_REWARD_RATE, action_prior_logits, MapGNNEncoder
        if PPO is not None:
            return True
        try:
            from PPO import PPO as _PPO
            from PPO import PPO_REWARD_RATE as _PPO_REWARD_RATE
            from PPO import compute_sword_reward as _compute_sword_reward
            from PPO import action_prior_logits as _action_prior_logits
            from GNN import MapGNNEncoder as _MapGNNEncoder
        except Exception as exc:
            self._log(f'[系统] RL 依赖不可用: {exc}')
            messagebox.showerror(
                '缺少 RL 依赖',
                '手动可视化可以继续使用，但 PPO 训练/演示需要安装 torch。\n\n'
                f'当前错误: {exc}',
            )
            return False
        PPO = _PPO
        PPO_REWARD_RATE = _PPO_REWARD_RATE
        compute_sword_reward = _compute_sword_reward
        action_prior_logits = _action_prior_logits
        MapGNNEncoder = _MapGNNEncoder
        return True

    def _ensure_q_components(self):
        """Load the lightweight tabular Q-learning baseline."""
        global TabularQLearningAgent, action_prior_logits
        if TabularQLearningAgent is None:
            try:
                from q_learning import TabularQLearningAgent as _TabularQLearningAgent
            except Exception as exc:
                self._log(f'[系统] Q-learning 组件不可用: {exc}')
                messagebox.showerror('Q-learning 加载失败', str(exc))
                return False
            TabularQLearningAgent = _TabularQLearningAgent
        if action_prior_logits is None:
            try:
                from PPO import action_prior_logits as _action_prior_logits
                action_prior_logits = _action_prior_logits
            except Exception as exc:
                self._log(f'[系统] 资源先验不可用，Q-learning 将只显示 Q 值: {exc}')
        return True

    def _get_q_agent(self, silent=False):
        if self.q_agent is not None:
            return self.q_agent
        if not self._ensure_q_components():
            return None
        path = Path(self.q_model_path_var.get()) if hasattr(self, 'q_model_path_var') else None
        if path and path.exists():
            try:
                self.q_agent = TabularQLearningAgent.load(path)
                if not silent:
                    self._log(f'[Q] 已加载 Q 表: {path}')
                return self.q_agent
            except Exception as exc:
                if not silent:
                    self._log(f'[Q] 加载 Q 表失败，将新建: {exc}')
        self.q_agent = TabularQLearningAgent(alpha=0.25, gamma=0.97, epsilon=0.25, prior_weight=1.0)
        return self.q_agent

    def _action_prior_values(self, actions):
        if not actions:
            return []
        stage_priors = stage_action_priors(self.env, actions)
        if action_prior_logits is None and not self._ensure_q_components():
            return stage_priors
        if action_prior_logits is None:
            return stage_priors
        try:
            ppo_priors = list(action_prior_logits(self.env, actions))
        except Exception as exc:
            self._log(f'[Q] 资源先验计算失败: {exc}')
            return stage_priors
        return [stage + 0.25 * ppo for stage, ppo in zip(stage_priors, ppo_priors)]

    def _compute_action_metrics(self, actions):
        agent = self._get_q_agent(silent=True)
        priors = self._action_prior_values(actions)
        if agent is None:
            q_rows = [{'q': 0.0, 'prior': priors[i] if i < len(priors) else 0.0, 'score': 0.0}
                      for i in range(len(actions))]
        else:
            q_rows = agent.action_values(self.env, actions, priors)
        metrics = []
        for index, action in enumerate(actions):
            estimate = self._estimate_action_reward(action)
            reward = None if estimate is None else float(estimate.get('total_reward', 0.0))
            row = q_rows[index] if index < len(q_rows) else {'q': 0.0, 'prior': 0.0, 'score': 0.0}
            metrics.append({
                'q': float(row.get('q', 0.0)),
                'prior': float(row.get('prior', 0.0)),
                'bonus': float(row.get('bonus', 0.0)),
                'score': float(row.get('score', 0.0)),
                'reward': reward,
                'path_len': self._reachable_hop_distance(action),
                'damage_drop': self._action_damage_drop(action),
                'unlock_value': self._estimated_unlock_value(action),
                'reachable': True,
                'masked': False,
            })
        return metrics

    # ------------------------------------------------------------------------
    #  模式切换
    # ------------------------------------------------------------------------
    def _set_mode(self, mode):
        self.mode_var.set(mode)
        self._switch_mode()

    def _update_mode_buttons(self):
        if not hasattr(self, 'mode_buttons'):
            return
        current = self.mode_var.get()
        for mode, btn in self.mode_buttons.items():
            if mode == current:
                btn.config(bg=COLORS['accent'], activebackground=COLORS['accent_dark'], fg='#ffffff')
            else:
                btn.config(bg='#eef2ff', activebackground='#dbeafe', fg='#1e3a8a')

    def _switch_mode(self):
        mode = self.mode_var.get()
        self._update_mode_buttons()
        if mode == 'manual':
            self.manual_panel.pack(fill=tk.BOTH, expand=True, padx=10, pady=7)
            self.train_panel.pack_forget()
            if self.env:
                self._refresh_action_list()
            self._set_status('手动模式：单击地图目标执行；右侧列表用于查看动作说明。')
        else:
            self.manual_panel.pack_forget()
            self.train_panel.pack(fill=tk.BOTH, expand=True, padx=10, pady=7)
            if mode == 'q':
                self._set_status('Q-learning 模式：训练后可在手动动作表查看每个候选动作的 Q(s,a)。')
            else:
                self._set_status('训练/演示模式：当前 PPO 目标主要是拿 5F 剑。')

    def _on_speed_change(self, _event=None):
        try:
            value = int(self.speed_ms_var.get())
        except (tk.TclError, ValueError):
            value = 40
        value = max(0, min(1000, value))
        self.speed_ms_var.set(value)
        self.train_speed_ms = value
        self._update_speed_buttons()
        self.status_var.set(f'演示延迟已设为 {value} ms；数值越小越快。')

    def _set_speed(self, value):
        self.speed_ms_var.set(value)
        self._on_speed_change()

    def _update_speed_buttons(self):
        if not hasattr(self, 'speed_buttons'):
            return
        value = self.train_speed_ms
        nearest = min(self.speed_buttons, key=lambda k: abs(k - value))
        for speed, btn in self.speed_buttons.items():
            if speed == nearest:
                btn.config(bg=COLORS['accent'], activebackground=COLORS['accent_dark'], fg='#ffffff')
            else:
                btn.config(bg='#eef2ff', activebackground='#dbeafe', fg='#1e3a8a')

    # ------------------------------------------------------------------------
    #  地图控制
    # ------------------------------------------------------------------------
    def _reset_env(self):
        if self._q_after_id:
            self.after_cancel(self._q_after_id)
            self._q_after_id = None
        self.q_training = False
        if hasattr(self, 'q_train_btn'):
            self.q_train_btn.config(state='normal')
        self._stop_route_playback()
        if self.route_playback is not None:
            self.route_playback.reset()
        self.env.reset(refresh_frame=True)
        self._reset_reward_monitor()
        self._refresh_state()
        if self.mode_var.get() == 'manual':
            self._refresh_action_list()
        self._update_view_controls()
        self._update_route_status()
        self._set_status('[系统] 环境已重置')

    def _sync_algorithm_start(self, silent=False, stop_route=True):
        if self.training or self.q_training:
            messagebox.showwarning('提示', '请先停止训练')
            return False
        if stop_route:
            self._stop_route_playback()
            if self.route_playback is not None:
                self.route_playback.reset()
        self.env.reset(refresh_frame=True)
        start_pos = (1, 7, 3)
        start_node = self.env.p2n.get(start_pos)
        if start_node is None:
            messagebox.showerror('同步失败', f'找不到算法起点节点: {self._format_pos(start_pos)}')
            return False
        self._apply_algorithm_start_map_state()
        self.env.player.money = 4
        ending = self.env.step(start_node, refresh_frame=True)
        second_thief = self.env.p2n.get((1, 9, 1))
        if second_thief is not None and getattr(second_thief, 'id', '') == 'thief':
            second_thief.disabled = False
            second_thief.activated = True
            self.env.anima_frame.hide_tile((1, 9, 1))
            self.env.anima_frame.hide_line((1, 9, 1))
        self.env.player.hp = 400
        self.env.player.atk = 10
        self.env.player.def_ = 10
        self.env.player.mdef = 0
        self.env.player.money = 4
        self.env.anima_frame.change_floor(start_pos[0])
        self._reset_reward_monitor(note='已同步到算法起点；后续 reward 从这里重新累计。')
        self._refresh_state()
        self._refresh_action_list()
        self._update_view_controls()
        self._update_route_status()
        messages = list(self.env.last_messages)
        if not silent:
            self._show_story_messages(messages)
            self._set_status(
                f'[同步] 已切到算法起点 {self._format_pos(start_pos)}：'
                f'1F三只史莱姆、3F魔王剧情和2F小偷剧情均已折叠，HP=400，金币=4，结果 {ending}。'
            )
        return True

    def _floor_up(self):
        f = self.env.anima_frame.now_floor + 1
        if f <= self.env.anima_frame.max_floor_num:
            self.env.anima_frame.change_floor(f)
            self._update_view_controls()
            self._set_status(f'[视图] 正在查看 {f + 1}F；这不会移动勇士。')
        else:
            self._set_status('[视图] 已经是最高楼层。')

    def _floor_down(self):
        f = self.env.anima_frame.now_floor - 1
        if f >= 0:
            self.env.anima_frame.change_floor(f)
            self._update_view_controls()
            self._set_status(f'[视图] 正在查看 {f + 1}F；这不会移动勇士。')
        else:
            self._set_status('[视图] 已经是最低楼层。')

    def _go_player_floor(self):
        z, _y, _x = self.env.n2p[self.env.observation[-1]]
        self.env.anima_frame.change_floor(z)
        self._update_view_controls()
        self._set_status(f'[视图] 已回到勇士所在楼层 {z + 1}F。')

    def _toggle_lines(self):
        vis = not self.env.anima_frame.line_visible
        self.env.anima_line_visible(vis)
        self.line_btn.config(text='隐藏连线' if vis else '显示连线')
        self._set_status('[视图] 已显示蓝色连线。' if vis else '[视图] 已隐藏蓝色连线。')

    def _update_view_controls(self):
        if not self.env:
            return
        view_floor = self.env.anima_frame.now_floor
        hero_floor = self.env.n2p[self.env.observation[-1]][0]
        self.view_floor_var.set(f'视图: {view_floor + 1}F / 勇士: {hero_floor + 1}F')
        self.floor_down_btn.config(state=tk.NORMAL if view_floor > 0 else tk.DISABLED)
        self.floor_up_btn.config(
            state=tk.NORMAL if view_floor < self.env.anima_frame.max_floor_num else tk.DISABLED
        )
        self.line_btn.config(text='隐藏连线' if self.env.anima_frame.line_visible else '显示连线')
        if hasattr(self, 'undo_btn'):
            state = tk.NORMAL if len(self.env.observation) > 1 else tk.DISABLED
            self.undo_btn.config(state=state)
        if hasattr(self, 'back_btn'):
            state = tk.NORMAL if len(self.env.observation) > 1 else tk.DISABLED
            self.back_btn.config(state=state)

    # ------------------------------------------------------------------------
    #  本地算法路线回放
    # ------------------------------------------------------------------------
    def _route_dirs(self):
        return ROUTE_SEARCH_DIRS

    def _iter_route_paths(self):
        seen = set()
        for route_dir in self._route_dirs():
            if not route_dir.exists():
                continue
            for path in route_dir.rglob('*.jsonl'):
                try:
                    rel_parts = path.relative_to(route_dir).parts[:-1]
                except ValueError:
                    rel_parts = path.parts[:-1]
                if any(part.lower() in ROUTE_IGNORED_PARTS for part in rel_parts):
                    continue
                if path in seen:
                    continue
                if not self._is_playable_route_file(path):
                    continue
                seen.add(path)
                yield path

    def _is_playable_route_file(self, path):
        try:
            with path.open(encoding='utf8') as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    action = row.get('action') if isinstance(row, dict) else None
                    return isinstance(action, dict) and 'target' in action
        except (OSError, json.JSONDecodeError):
            return False
        return False

    def _route_label(self, path):
        try:
            return str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path)

    def _route_sort_key(self, path):
        name = path.name
        success_rank = 0 if 'manual_success_no_shop_true_10f_trap' in name else 1
        return (success_rank, -path.stat().st_mtime)

    def _refresh_route_choices(self, silent=False):
        if not hasattr(self, 'route_combo'):
            return
        paths = sorted(self._iter_route_paths(), key=self._route_sort_key)
        self.route_choices = {}
        labels = []
        for path in paths:
            label = self._route_label(path)
            if label in self.route_choices:
                label = str(path)
            self.route_choices[label] = path
            labels.append(label)
        self.route_combo['values'] = labels
        current = self.route_choice_var.get()
        if labels and current not in self.route_choices:
            self.route_choice_var.set(labels[0])
        elif not labels:
            self.route_choice_var.set('')
            self.route_playback = None
        self._update_route_status()
        if not silent:
            self._set_status(f'[路线] 已扫描 {len(labels)} 条路线。')

    def _on_route_selected(self, _event=None):
        self._load_selected_route()

    def _load_selected_route(self):
        self._stop_route_playback()
        label = self.route_choice_var.get()
        if label not in self.route_choices:
            self._refresh_route_choices(silent=True)
        path = self.route_choices.get(self.route_choice_var.get())
        if path is None:
            messagebox.showwarning('没有路线', '未找到可播放的 .jsonl 路线。请确认 artifacts/manual_exploration_20260524、artifacts/expert 或 artifacts/runs 中存在路线文件。')
            return False
        try:
            self.route_playback = load_route(path)
        except Exception as exc:
            messagebox.showerror('路线加载失败', str(exc))
            self._set_status(f'[路线] 加载失败: {exc}')
            return False
        self._update_route_status()
        first = self.route_playback.current_step()
        target = visualizer_pos_text(first.target_pos) if first else '--'
        self._set_status(f'[路线] 已加载 {self.route_playback.path.name}，共 {self.route_playback.total} 步，第一步目标 {target}。')
        return True

    def _update_route_status(self):
        if not hasattr(self, 'route_status_var'):
            return
        if self.route_playback is None:
            self.route_status_var.set('未加载路线')
            return
        step = self.route_playback.current_step()
        if step is None:
            self.route_status_var.set(f'{self.route_playback.progress_text()} · 已结束')
            return
        self.route_status_var.set(
            f'{self.route_playback.progress_text()} · 下一步 {step.label} -> {visualizer_pos_text(step.target_pos)}'
        )

    def _ensure_route_start(self):
        if self.route_playback is None or self.route_playback.done:
            return True
        step = self.route_playback.current_step()
        expected = step.before_pos
        if expected is None:
            return True
        current = self.env.n2p[self.env.observation[-1]][:3]
        if current == expected:
            return True
        if self.route_playback.align_to_current(self.env):
            return True
        if self.route_playback.cursor == 0 and expected == (1, 7, 3):
            return self._sync_algorithm_start(silent=True, stop_route=False)
        self._stop_route_playback()
        messagebox.showwarning(
            '路线状态不匹配',
            f'路线第 {self.route_playback.cursor + 1} 步期望当前位置 {visualizer_pos_text(expected)}，'
            f'但当前是 {self._format_pos(current)}。\n请回退、重置或重新同步起点后再播放。',
        )
        return False

    def _clear_visual_cell(self, pos):
        node = self.env.p2n.get(pos)
        if node is not None:
            node.activated = True
        self.env.anima_frame.hide_tile(pos)
        self.env.anima_frame.hide_line(pos)

    def _apply_algorithm_start_map_state(self):
        # The algorithm starts after the compulsory 1F slimes and the MT3/MT2 story chain.
        for pos in [(0, 1, 3), (0, 1, 4), (0, 1, 5)]:
            self._clear_visual_cell(pos)
        for pos in [(2, 7, 5), (2, 8, 5), (2, 9, 4), (2, 9, 6), (2, 10, 5), (2, 9, 5)]:
            self._clear_visual_cell(pos)

    def _route_step(self):
        if self.route_playback is None:
            if not self._load_selected_route():
                return False
        if self.route_playback.done:
            self._stop_route_playback()
            self._update_route_status()
            self._set_status('[路线] 已播放到末尾。')
            return False
        if not self._ensure_route_start():
            return False
        step = self.route_playback.current_step()
        try:
            action, actions, completes_step, matched_cursor = self.route_playback.find_visualizer_action(self.env)
        except Exception as exc:
            self._stop_route_playback()
            messagebox.showerror('路线执行失败', str(exc))
            return False
        if matched_cursor is not None and matched_cursor != self.route_playback.cursor:
            self.route_playback.cursor = matched_cursor
            step = self.route_playback.current_step()
        if action is None:
            if step.is_visualizer_noop():
                self.route_playback.advance()
                self._update_route_status()
                self._set_status(f'[路线] 跳过可视化中已简化的动作: {step.label}')
                return True
            if self._sync_hidden_route_state(step):
                self.route_playback.advance()
                self._refresh_state()
                self._refresh_action_list()
                self._update_route_status()
                self._set_status(f'[路线] 同步隐藏路线状态: {step.label}')
                return True
            if self._sync_internal_route_step(step):
                self.route_playback.advance()
                self._refresh_state()
                self._refresh_action_list()
                self._update_route_status()
                self._set_status(f'[路线] 同步纯移动/楼梯段: {step.label}')
                return True
            available = ', '.join(self._format_pos(self.env.n2p[a]) for a in actions[:8])
            self._stop_route_playback()
            messagebox.showwarning(
                '找不到路线动作',
                f'第 {self.route_playback.cursor + 1} 步目标 {visualizer_pos_text(step.target_pos)} '
                f'不是当前合法宏动作。\n当前可选: {available or "无"}',
            )
            self._set_status(f'[路线] 动作不匹配: {step.label} -> {visualizer_pos_text(step.target_pos)}')
            return False
        before = self.env.get_player_state().copy()
        before_pos = self.env.n2p[self.env.observation[-1]][:3]
        before_phi = stage_potential(self.env)
        ending = self.env.step(action, refresh_frame=True)
        after = self.env.get_player_state().copy()
        after_pos = self.env.n2p[self.env.observation[-1]][:3]
        self._record_reward(
            self._make_stage_reward_row(
                action,
                before,
                after,
                ending,
                before_pos,
                after_pos,
                before_phi,
                source='路线',
            )
        )
        if completes_step:
            self.route_playback.advance()
        messages = list(self.env.last_messages)
        self._refresh_state()
        self._refresh_action_list()
        self._update_route_status()
        pos = self.env.n2p[action]
        self._show_story_messages(messages)
        self._set_status(
            f'[路线] {self.route_playback.cursor}/{self.route_playback.total} '
            f'{step.label} -> {self._format_pos(pos)}'
            f'{"，完成该路线步" if completes_step else "，路径中间动作"}，结果 {ending}'
        )
        if ending != 'continue':
            self._stop_route_playback()
            messagebox.showinfo('路线结束', f'结束状态: {ending}\n剩余生命: {self.env.player.hp}')
            return False
        return True

    def _sync_hidden_route_state(self, step):
        """Synchronize hidden solver events that are absent from visualizer data.

        Some original-route rows, such as hidden fake-wall rewards, do not have
        a concrete node in the simplified visualizer map but do change HP or
        inventory.  These rows are not training actions; this sync only keeps
        route playback faithful to the simulator trace.
        """

        if not step.needs_visualizer_state_sync():
            return False
        if not step.after:
            return False
        pos = step.after_pos or step.target_pos
        before_pos = self.env.n2p[self.env.observation[-1]][:3]
        before = self.env.get_player_state().copy()
        node = self.env.p2n.get(pos)
        if node is None:
            node = Terrain({'cls': 'terrains', 'id': 'background', 'noPass': False})
            node.activated = True
            self.env.n2p[node] = pos
            self.env.p2n[pos] = node
            for node2 in self.env.n2p:
                node2.links.clear()
            self.env.create_nodes()
        else:
            node.activated = True
        self._apply_route_state(step.after)
        self.env.observation.append(node)
        after = self.env.get_player_state().copy()
        if hasattr(self.env, 'anima_frame') and self.env.anima_frame.canvas is not None:
            self.env.anima_frame.delete_new_lines()
            self.env.anima_frame.player_move(pos)
            for node2 in self.env.get_actions():
                self.env.anima_frame.show_line(pos, self.env.n2p[node2])
            self.env.anima_frame.top_player()
            self.env.refresh_state_table()
        self._record_reward(
            self._make_reward_row(
                node,
                before,
                after,
                'continue',
                before_pos,
                pos,
                source='路线隐藏同步',
                base_reward=0.0,
                total_reward=0.0,
                reward_components={'route_hidden_sync': 0.0},
            )
        )
        return True

    def _apply_route_state(self, state):
        for key, attr in [('hp', 'hp'), ('atk', 'atk'), ('def', 'def_'), ('mdef', 'mdef'), ('money', 'money')]:
            if key in state:
                setattr(self.env.player, attr, int(state[key]))
        for key, value in state.get('keys', {}).items():
            self.env.player.items[key] = int(value)
        if isinstance(state.get('flags'), dict):
            self.env.flags = dict(state['flags'])

    def _sync_internal_route_step(self, step):
        """Advance a solver-only movement/stair segment in the visualizer.

        The solver route records stairs and long empty walks. The visualizer
        hides stairs from the action list, so some of these route rows have no
        direct executable action. If the row does not change resources, we can
        safely move the displayed hero to the solver row's `after` position.
        """

        if not step.is_state_noop():
            return False
        pos = step.after_pos or step.target_pos
        node = self.env.p2n.get(pos)
        if node is None or getattr(node, 'class_', None) != 'terrains':
            return False
        before_pos = self.env.n2p[self.env.observation[-1]][:3]
        before = self.env.get_player_state().copy()
        try:
            node.activate(self.env.player)
        except Exception:
            return False
        self.env.observation.append(node)
        after = self.env.get_player_state().copy()
        if hasattr(self.env, 'anima_frame') and self.env.anima_frame.canvas is not None:
            self.env.anima_frame.delete_new_lines()
            self.env.anima_frame.player_move(pos)
            for node2 in self.env.get_actions():
                self.env.anima_frame.show_line(pos, self.env.n2p[node2])
            self.env.anima_frame.top_player()
            self.env.refresh_state_table()
        self._record_reward(
            self._make_reward_row(
                node,
                before,
                after,
                'continue',
                before_pos,
                pos,
                source='路线同步',
                base_reward=0.0,
                total_reward=0.0,
                reward_components={'route_sync': 0.0},
            )
        )
        return True

    def _toggle_route_playback(self):
        if self.route_playing:
            self._stop_route_playback()
            self._set_status('[路线] 已暂停播放。')
            return
        if self.training or self.q_training:
            messagebox.showwarning('提示', '请先停止训练')
            return
        if self.route_playback is None:
            if not self._load_selected_route():
                return
        self.route_playing = True
        self.route_play_btn.config(text='暂停路线', bg=COLORS['warning'], activebackground='#b45309')
        self._set_status('[路线] 开始播放。')
        self._route_play_loop()

    def _route_play_loop(self):
        if not self.route_playing:
            return
        ok = self._route_step()
        if not ok or self.route_playback is None or self.route_playback.done:
            self._stop_route_playback()
            self._update_route_status()
            return
        if not self.route_playing:
            return
        self._route_after_id = self.after(self.train_speed_ms, self._route_play_loop)

    def _stop_route_playback(self):
        if self._route_after_id:
            self.after_cancel(self._route_after_id)
            self._route_after_id = None
        self.route_playing = False
        if hasattr(self, 'route_play_btn'):
            self.route_play_btn.config(text='播放路线', bg=COLORS['success'], activebackground=COLORS['success_dark'])

    # ------------------------------------------------------------------------
    #  手动模式 - 动作列表
    # ------------------------------------------------------------------------
    def _refresh_action_list(self):
        for item in self.action_tree.get_children():
            self.action_tree.delete(item)
        self.action_list = self.env.get_feasible_actions()
        self.action_metrics = self._compute_action_metrics(self.action_list)
        self.action_iids = []
        best_idx = self._best_metric_index(self.action_metrics)
        for idx, action in enumerate(self.action_list, 1):
            pos = self.env.n2p[action]
            metric = self.action_metrics[idx - 1] if idx - 1 < len(self.action_metrics) else {}
            iid = self.action_tree.insert(
                '',
                'end',
                values=(
                    idx,
                    self._format_pos(pos),
                    action.class_,
                    action.id,
                    self._format_metric(metric.get('q')),
                    self._format_metric(metric.get('reward')),
                    self._format_metric(metric.get('path_len')),
                    self._format_metric(metric.get('damage_drop')),
                    self._short_action_desc(action),
                ),
                tags=(self._action_tag(action, metric, idx - 1 == best_idx),),
            )
            self.action_iids.append(iid)
        self.selected_action_idx = -1
        self._update_action_summary()
        self._update_action_detail(None)
        self._update_view_controls()
        if self.q_agent is not None and 'Q状态' in self.state_labels:
            self.state_labels['Q状态'].config(text=self._compact_count(len(getattr(self.q_agent, 'q', {}))))
        self._set_status(f'[动作] 当前有 {len(self.action_list)} 个可行动作。')

    def _best_metric_index(self, metrics):
        if not metrics:
            return -1
        return max(range(len(metrics)), key=lambda idx: metrics[idx].get('score', float('-inf')))

    def _update_action_summary(self):
        if not getattr(self, 'action_summary_var', None):
            return
        if not self.action_list or not self.action_metrics:
            self.action_summary_var.set('动作建议：当前没有可行动作。')
            return
        def reward_value(idx, default):
            value = self.action_metrics[idx].get('reward')
            return default if value is None else value

        best_idx = self._best_metric_index(self.action_metrics)
        best_reward_idx = max(
            range(len(self.action_metrics)),
            key=lambda idx: reward_value(idx, float('-inf')),
        )
        worst_reward_idx = min(
            range(len(self.action_metrics)),
            key=lambda idx: reward_value(idx, float('inf')),
        )

        def action_line(prefix, idx, key):
            action = self.action_list[idx]
            metric = self.action_metrics[idx]
            pos = self.env.n2p[action]
            value = self._format_metric(metric.get(key))
            return f'{prefix} #{idx + 1} {self._format_pos(pos)} {action.class_}:{action.id} ({key}={value})'

        lines = [action_line('Q推荐', best_idx, 'score')]
        if best_reward_idx != best_idx:
            lines.append(action_line('即时R最高', best_reward_idx, 'reward'))
        if self.action_metrics[worst_reward_idx].get('reward') is not None:
            lines.append(action_line('高风险', worst_reward_idx, 'reward'))
        self.action_summary_var.set('动作建议：' + ' | '.join(lines))

    def _on_action_select(self, event):
        sel = self.action_tree.selection()
        if not sel:
            return
        values = self.action_tree.item(sel[0], 'values')
        idx = int(values[0]) - 1
        self.selected_action_idx = idx
        if 0 <= idx < len(self.action_list):
            action = self.action_list[idx]
            pos = self.env.n2p[action]
            self.env.anima_frame.show_cursor(pos)
            self._update_view_controls()
            self._update_action_detail(action)
            self._set_status(f'[选中] {self._format_pos(pos)} {action.class_}:{action.id}')

    def _execute_selected(self):
        if self.selected_action_idx < 0 or self.selected_action_idx >= len(self.action_list):
            self._set_status('[提示] 请先从右侧列表选择一个动作；也可以单击地图上的可行动作目标。')
            messagebox.showwarning('提示', '请先从列表中选择一个动作')
            return
        action = self.action_list[self.selected_action_idx]
        before = self.env.get_player_state().copy()
        before_pos = self.env.n2p[self.env.observation[-1]]
        before_phi = stage_potential(self.env)
        ending = self.env.step(action, refresh_frame=True)
        after = self.env.get_player_state().copy()
        messages = list(self.env.last_messages)
        self._refresh_state()
        self._refresh_action_list()
        pos = self.env.n2p[action]
        now_pos = self.env.n2p[self.env.observation[-1]]
        self._record_reward(
            self._make_stage_reward_row(
                action,
                before,
                after,
                ending,
                before_pos[:3],
                now_pos[:3],
                before_phi,
                source='手动',
            )
        )
        self._set_status(
            f'[执行] {self._format_pos(before_pos)} -> {self._format_pos(pos)}，'
            f'当前 {self._format_pos(now_pos)}，结果 {ending}'
        )
        self._show_story_messages(messages)
        if ending != 'continue':
            messagebox.showinfo('回合结束', f'结束状态: {ending}\n剩余生命: {self.env.player.hp}')

    def _back_step(self):
        if len(self.env.observation) <= 1:
            self._set_status('[回退] 已经在初始状态，不能继续回退。')
            return
        self.env.back_step(1, refresh_frame=True)
        self._reset_reward_monitor(note='已回退一步；当前 reward 累计已清零，继续执行会重新统计。')
        self._refresh_state()
        self._refresh_action_list()
        self._set_status('[回退] 已回退一步。')

    def _on_map_press(self, event):
        self._map_press_xy = (event.x, event.y)

    def _on_map_click(self, event):
        if self._map_press_xy is not None:
            dx = event.x - self._map_press_xy[0]
            dy = event.y - self._map_press_xy[1]
            if dx * dx + dy * dy > 36:
                return
        canvas = self.env.anima_frame.canvas
        tile_size = self.env.anima_frame.tile_size
        x = int(canvas.canvasx(event.x) // tile_size)
        y = int(canvas.canvasy(event.y) // tile_size)
        z = self.env.anima_frame.now_floor
        target = (z, y, x)
        for idx, action in enumerate(self.action_list):
            if self.env.n2p[action][:3] == target:
                self.selected_action_idx = idx
                if idx < len(self.action_iids):
                    self.action_tree.selection_set(self.action_iids[idx])
                    self.action_tree.see(self.action_iids[idx])
                self._update_action_detail(action)
                self._execute_selected()
                return
        self._set_status(f'[地图] {self._format_pos(target)} 不是当前可执行目标；请点击蓝/黄连线端点或右侧动作。')

    def _format_pos(self, pos):
        z, y, x = pos[:3]
        return f'{z + 1}F({y},{x})'

    def _reachable_hop_distance(self, target_action):
        if self.env is None or target_action is None:
            return None
        start = self.env.observation[-1]
        if start is target_action:
            return 0
        visited = {start}
        queue = [(start, 0)]
        while queue:
            node, dist = queue.pop(0)
            for nxt in getattr(node, 'links', []):
                if nxt in visited or getattr(nxt, 'disabled', False):
                    continue
                if nxt is target_action:
                    return dist + 1
                visited.add(nxt)
                if getattr(nxt, 'activated', False) or self.env._is_internal_path_node(nxt):
                    queue.append((nxt, dist + 1))
        return None

    def _enemy_damage_with_stats(self, action, atk, defense, hp=None):
        p_damage = atk - getattr(action, 'def_', 0)
        if p_damage <= 0:
            return None
        rounds = math.ceil(action.hp / p_damage) - 1
        e_damage = max(action.atk - defense, 0)
        damage = e_damage * rounds
        if action.special == 1:
            damage += e_damage
        elif action.special == 11:
            damage += (hp or self.env.player.hp) // action.skw['value']
        elif action.special == 22:
            damage += action.skw['damage']
        damage -= self.env.player.mdef
        return max(0, int(damage))

    def _action_damage_drop(self, action):
        if action is None or getattr(action, 'class_', '') != 'enemies':
            return 0.0
        p = self.env.player
        current = self._enemy_damage_with_stats(action, p.atk, p.def_, p.hp)
        if current is None:
            return 0.0
        atk_plus = self._enemy_damage_with_stats(action, p.atk + 1, p.def_, p.hp)
        def_plus = self._enemy_damage_with_stats(action, p.atk, p.def_ + 1, p.hp)
        return float(max(0, current - (atk_plus if atk_plus is not None else current))
                     + max(0, current - (def_plus if def_plus is not None else current)))

    def _visual_item_value(self, node):
        if getattr(node, 'class_', '') != 'items':
            return 0.0
        item_id = getattr(node, 'id', '')
        if item_id == 'redGem':
            return 120.0 + self._action_damage_drop(node)
        if item_id == 'blueGem':
            return 110.0 + self._action_damage_drop(node)
        if item_id == 'sword1':
            return 900.0
        if item_id == 'shield1':
            return 900.0
        if item_id == 'redKey':
            return 700.0
        if item_id == 'blueKey':
            return 260.0
        if item_id == 'yellowKey':
            return 120.0
        if 'Potion' in item_id:
            return 180.0 if self.env.player.hp < 700 else 45.0
        return 10.0

    def _estimated_unlock_value(self, action):
        if action is None or getattr(action, 'class_', '') != 'enemies':
            return 0.0
        pos = self.env.n2p.get(action)
        if pos is None:
            return 0.0
        z, y, x = pos[:3]
        value = float(getattr(action, 'money', 0)) * 0.4
        for npos in ((z, y - 1, x), (z, y + 1, x), (z, y, x - 1), (z, y, x + 1)):
            node = self.env.p2n.get(npos)
            if node is not None and not getattr(node, 'activated', False) and not getattr(node, 'disabled', False):
                value += self._visual_item_value(node)
        return value

    def _short_action_desc(self, action):
        if action.class_ == 'enemies':
            damage = self._predict_enemy_damage(action)
            if damage is None:
                return '战斗'
            return f'战斗 预计损血 {damage}'
        if action.class_ == 'items':
            return self._format_item_effect(action)
        if action.class_ == 'terrains':
            if 'Door' in action.id:
                return '开门/消耗钥匙'
            if 'Stair' in action.id:
                return '楼梯/换层'
            return '地形事件'
        if action.class_ == 'npcs':
            return 'NPC/剧情事件'
        if action.class_ == 'afterEvent':
            return '机关/楼层事件'
        if action.class_ == 'endFlag':
            return '通关标记'
        return '可触发事件'

    def _action_tag(self, action, metric=None, is_best=False):
        metric = metric or {}
        if is_best and metric.get('score', 0.0) > 0:
            return 'q_best'
        reward = metric.get('reward')
        if reward is not None and reward < -250:
            return 'r_bad'
        if action.class_ == 'enemies':
            return 'enemy'
        if action.class_ == 'items':
            return 'item'
        if action.class_ == 'terrains':
            return 'terrain'
        if action.class_ in {'npcs', 'afterEvent'}:
            return 'story'
        return 'other'

    def _format_item_effect(self, action):
        effects = getattr(action, 'effects', {})
        cls_ = getattr(action, 'cls_', '')
        if cls_ == 'item':
            return f'获得 {action.id}'
        if effects:
            parts = [f'{k}+{v}' for k, v in effects.items() if isinstance(v, (int, float))]
            if parts:
                return ', '.join(parts)
        return f'获得 {action.id}'

    def _predict_enemy_damage(self, action):
        p = self.env.player
        p_damage = p.atk - getattr(action, 'def_', 0)
        if p_damage <= 0:
            return None
        rounds = action.hp // p_damage - (action.hp % p_damage == 0)
        e_damage = max(action.atk - p.def_, 0)
        damage = e_damage * rounds
        if action.special == 1:
            damage += e_damage
        elif action.special == 11:
            damage += p.hp // action.skw['value']
        elif action.special == 22:
            damage += action.skw['damage']
        damage -= p.mdef
        return max(0, min(damage, p.hp))

    def _describe_action(self, action):
        pos = self.env.n2p[action]
        lines = [
            f'目标：{self._format_pos(pos)}',
            f'类型：{action.class_}',
            f'ID：{action.id}',
            f'作用：{self._short_action_desc(action)}',
        ]
        if action.class_ == 'enemies':
            damage = self._predict_enemy_damage(action)
            drop = self._action_damage_drop(action)
            unlock = self._estimated_unlock_value(action)
            lines.extend([
                f'怪物属性：HP {action.hp}, ATK {action.atk}, DEF {action.def_}, 金币 {action.money}',
                '战斗结果：无法破防/会死亡' if damage is None else f'预计损血：{damage}',
                f'临界收益：ATK+1/DEF+1 合计预计少损血 {drop:.0f}；击杀后邻近解锁价值约 {unlock:.1f}',
            ])
        elif action.class_ == 'items':
            lines.append(f'道具效果：{self._format_item_effect(action)}')
        elif action.class_ == 'terrains':
            if action.id == 'yellowDoor':
                lines.append('需要并消耗 1 把黄钥匙。')
            elif action.id == 'blueDoor':
                lines.append('需要并消耗 1 把蓝钥匙。')
            elif action.id == 'redDoor':
                lines.append('需要并消耗 1 把红钥匙。')
            elif 'Stair' in action.id:
                lines.append('这是楼梯连接边；执行会换层，但 reward 中不把上下楼作为目标收益。')
        elif action.class_ == 'npcs':
            lines.append('这是剧情/NPC 事件，执行后可能改变属性、钥匙或地图机关。')
        lines.append('')
        lines.append('提示：顶部“查看上层/下层”只是看地图；只有执行这里的动作才会移动勇士。')
        return '\n'.join(lines)

    def _update_action_detail(self, action):
        if not hasattr(self, 'action_detail'):
            return
        self.action_detail.config(state=tk.NORMAL)
        self.action_detail.delete('1.0', tk.END)
        if action is None:
            self.action_detail.insert(
                tk.END,
                '未选择动作。\n单击右侧动作查看说明；单击地图目标、按 Enter 或点执行按钮来执行。',
            )
        else:
            text = self._describe_action(action)
            estimate = self._estimate_action_reward(action)
            text += '\n' + self._format_reward_estimate(estimate)
            metric = self._metric_for_action(action, estimate)
            text += (
                '\nQ-learning 反馈：'
                f'Q(s,a) {self._format_metric(metric.get("q"))} | '
                f'即时R {self._format_metric(metric.get("reward"))} | '
                f'资源先验 {self._format_metric(metric.get("prior"))} | '
                f'探索 {self._format_metric(metric.get("bonus"))} | '
                f'选择分 {self._format_metric(metric.get("score"))}'
            )
            text += (
                '\n图节点信息：'
                f'reachable={int(bool(metric.get("reachable", True)))} | '
                f'masked={int(bool(metric.get("masked", False)))} | '
                f'path_len={self._format_metric(metric.get("path_len"))} | '
                f'damage_drop={self._format_metric(metric.get("damage_drop"))} | '
                f'unlock_value={self._format_metric(metric.get("unlock_value"))}'
            )
            self.action_detail.insert(tk.END, text)
        self.action_detail.config(state=tk.DISABLED)

    def _metric_for_action(self, action, estimate=None):
        for known_action, metric in zip(self.action_list, self.action_metrics):
            if known_action is action:
                return metric
        priors = self._action_prior_values([action])
        agent = self._get_q_agent(silent=True)
        prior = priors[0] if priors else 0.0
        if estimate is None:
            estimate = self._estimate_action_reward(action)
        reward = None if estimate is None else float(estimate.get('total_reward', 0.0))
        if agent is not None:
            row = agent.action_values(self.env, [action], priors)[0]
            return {
                'q': float(row.get('q', 0.0)),
                'prior': float(row.get('prior', prior)),
                'bonus': float(row.get('bonus', 0.0)),
                'score': float(row.get('score', 0.0)),
                'reward': reward,
                'path_len': self._reachable_hop_distance(action),
                'damage_drop': self._action_damage_drop(action),
                'unlock_value': self._estimated_unlock_value(action),
                'reachable': True,
                'masked': False,
            }
        return {
            'q': 0.0,
            'prior': float(prior),
            'bonus': 0.0,
            'score': float(prior),
            'reward': reward,
            'path_len': self._reachable_hop_distance(action),
            'damage_drop': self._action_damage_drop(action),
            'unlock_value': self._estimated_unlock_value(action),
            'reachable': True,
            'masked': False,
        }

    # ------------------------------------------------------------------------
    #  Tabular Q-learning
    # ------------------------------------------------------------------------
    def _start_q_train_generator(self):
        if self.training or self.q_training:
            messagebox.showwarning('提示', '已有训练正在进行中')
            return
        agent = self._get_q_agent()
        if agent is None:
            return
        try:
            rounds = int(self.rounds_var.get())
        except ValueError:
            messagebox.showerror('错误', '回合数必须是整数')
            return
        if rounds <= 0:
            return
        agent.epsilon = 0.25
        save_path = Path(self.q_model_path_var.get())
        self.q_training = True
        self.stop_training = False
        self.q_train_btn.config(state='disabled')
        self.train_status.config(text='Q-learning 训练中...', fg='green')
        self._q_train_state = {
            'rounds': rounds,
            'episode': 0,
            'step_count': 0,
            'max_steps': 180,
            'save_path': save_path,
            'sword_collected': False,
            'best_hp': 0,
            'last_td': 0.0,
        }
        self._log(
            f'[Q] 开始 {rounds} 回合训练；alpha={agent.alpha:.2f}, '
            f'gamma={agent.gamma:.2f}, epsilon={agent.epsilon:.2f}, prior_weight={agent.prior_weight:.2f}'
        )
        self._q_episode_start()

    def _q_episode_start(self):
        if not self.q_training or self.stop_training or self._q_train_state['episode'] >= self._q_train_state['rounds']:
            self._q_train_finish()
            return
        self.env.reset(refresh_frame=True)
        self._reset_reward_monitor(note=f'Q Episode {self._q_train_state["episode"] + 1} 开始。')
        self._refresh_state()
        self._refresh_action_list()
        self._q_train_state['step_count'] = 0
        self._q_step_loop()

    def _q_step_loop(self):
        if not self.q_training or self.stop_training:
            self._q_train_finish()
            return
        agent = self._get_q_agent(silent=True)
        actions = self.env.get_feasible_actions()
        if not actions:
            self._q_episode_end('stop')
            return
        priors = self._action_prior_values(actions)
        state_key = agent.state_key(self.env)
        action, index, mode = agent.choose_action(self.env, actions, priors)
        action_key = agent.action_key(self.env, action)

        before_phi = stage_potential(self.env)
        before = self.env.get_player_state().copy()
        before_pos = self.env.n2p[self.env.observation[-1]][:3]
        ending = self.env.step(action, refresh_frame=True)
        after = self.env.get_player_state().copy()
        after_pos = self.env.n2p[self.env.observation[-1]][:3]
        reward_info = self._stage_reward_info(action, before, after, ending, before_phi)
        base_reward = reward_info.components.get('env_step', 0.0) if reward_info else self._base_reward(before, after, ending)
        reward = reward_info.total if reward_info else self._local_sword_reward_before_penalty(
            action, base_reward, ending, before_pos, after_pos
        )
        done = ending != 'continue'

        if reward_info is not None and reward_info.after.stage != reward_info.before.stage:
            self._q_train_state['sword_collected'] = True
            self._q_train_state['best_hp'] = max(self._q_train_state['best_hp'], self.env.player.hp)
            self._log(
                f'[Q Episode {self._q_train_state["episode"] + 1}] '
                f'阶段推进 {reward_info.before.stage}->{reward_info.after.stage}，HP={self.env.player.hp}'
            )
        if reward_info is not None and reward_info.after.stage == 'done':
            done = True

        next_step_count = self._q_train_state['step_count'] + 1
        timeout_penalty = 0.0
        if not done and next_step_count >= self._q_train_state['max_steps']:
            timeout_penalty = 500.0
            reward -= timeout_penalty
            ending = 'timeout'
            done = True

        next_actions = [] if done else self.env.get_feasible_actions()
        next_state_key = None if done else agent.state_key(self.env)
        next_action_keys = [agent.action_key(self.env, candidate) for candidate in next_actions]
        td = agent.update(state_key, action_key, reward, next_state_key, next_action_keys, done)
        self._q_train_state['last_td'] = td['td_error']

        self._record_reward(
            self._make_reward_row(
                action,
                before,
                after,
                ending,
                before_pos,
                after_pos,
                source=f'Q训练/{mode}',
                base_reward=base_reward,
                total_reward=reward,
                timeout_penalty=timeout_penalty,
                reward_components={} if reward_info is None else reward_info.components,
                stage_before=None if reward_info is None else reward_info.before.stage,
                stage_after=None if reward_info is None else reward_info.after.stage,
                phi_before=None if reward_info is None else reward_info.before.total,
                phi_after=None if reward_info is None else reward_info.after.total,
            )
        )
        self._refresh_state()
        self._refresh_action_list()
        self.update_idletasks()
        self._q_train_state['step_count'] = next_step_count

        if done:
            self._q_episode_end(ending)
        else:
            self._q_after_id = self.after(self.train_speed_ms, self._q_step_loop)

    def _q_episode_end(self, ending):
        agent = self._get_q_agent(silent=True)
        agent.episodes += 1
        ep = self._q_train_state['episode'] + 1
        self._log(
            f'[Q Episode {ep}/{self._q_train_state["rounds"]}] '
            f'steps={self._q_train_state["step_count"]} ending={ending} '
            f'hp={self.env.player.hp} td={self._q_train_state["last_td"]:.2f}'
        )
        if ep % 20 == 0:
            agent.save(self._q_train_state['save_path'])
            self._log(f'[Q] 已保存中间 Q 表: {self._q_train_state["save_path"]}')
        self.train_status.config(text=f'Q Episode {ep}/{self._q_train_state["rounds"]}')
        self._q_train_state['episode'] = ep
        self._q_after_id = self.after(self.train_speed_ms, self._q_episode_start)

    def _q_train_finish(self):
        if self._q_after_id:
            self.after_cancel(self._q_after_id)
            self._q_after_id = None
        was_running = self.q_training
        self.q_training = False
        if hasattr(self, 'q_train_btn'):
            self.q_train_btn.config(state='normal')
        agent = self._get_q_agent(silent=True)
        if agent is not None and hasattr(self, '_q_train_state'):
            agent.save(self._q_train_state['save_path'])
            if was_running:
                self._log(f'[Q] 训练结束，Q 表已保存至: {self._q_train_state["save_path"]}')
                if self._q_train_state['sword_collected']:
                    self._log(f'[Q] 训练中曾拿到剑，最佳 HP={self._q_train_state["best_hp"]}')
        self.train_status.config(text='Q-learning 训练结束', fg='gray')
        self._refresh_action_list()

    def _stop_q_training(self):
        self.stop_training = True
        if self._q_after_id:
            self.after_cancel(self._q_after_id)
            self._q_after_id = None
        self.q_training = False
        if hasattr(self, 'q_train_btn'):
            self.q_train_btn.config(state='normal')
        self._log('[Q] 已请求停止')
        self.train_status.config(text='Q-learning 已停止', fg='red')

    def _start_q_demo(self):
        if self.training or self.q_training:
            messagebox.showwarning('提示', '请先停止训练')
            return
        if not self._ensure_q_components():
            return
        path = Path(self.q_model_path_var.get())
        if path.exists():
            try:
                self.q_agent = TabularQLearningAgent.load(path)
                self._log(f'[Q演示] 已加载 Q 表: {path}')
            except Exception as exc:
                messagebox.showerror('Q 表加载失败', str(exc))
                return
        agent = self._get_q_agent()
        if agent is None:
            return
        agent.epsilon = 0.0
        self.mode_var.set('q')
        self._switch_mode()
        self.env.reset(refresh_frame=True)
        self._reset_reward_monitor(note='Q 贪婪演示开始；动作表会显示当前 Q 值。')
        self._refresh_state()
        self._refresh_action_list()
        self._q_demo_state = {'step_count': 0, 'max_steps': 180}
        self.train_status.config(text='Q 贪婪演示中...', fg='blue')
        self._q_demo_step()

    def _q_demo_step(self):
        agent = self._get_q_agent(silent=True)
        actions = self.env.get_feasible_actions()
        if not actions:
            self._log('[Q演示] 无可行动作，演示结束')
            self.train_status.config(text='Q 演示结束', fg='gray')
            return
        priors = self._action_prior_values(actions)
        action, index = agent.greedy_action(self.env, actions, priors)
        before = self.env.get_player_state().copy()
        before_pos = self.env.n2p[self.env.observation[-1]][:3]
        before_phi = stage_potential(self.env)
        ending = self.env.step(action, refresh_frame=True)
        after = self.env.get_player_state().copy()
        after_pos = self.env.n2p[self.env.observation[-1]][:3]
        reward_info = self._stage_reward_info(action, before, after, ending, before_phi)
        base_reward = reward_info.components.get('env_step', 0.0) if reward_info else self._base_reward(before, after, ending)
        reward = reward_info.total if reward_info else self._local_sword_reward_before_penalty(
            action, base_reward, ending, before_pos, after_pos
        )
        self._record_reward(
            self._make_reward_row(
                action,
                before,
                after,
                ending,
                before_pos,
                after_pos,
                source='Q演示',
                base_reward=base_reward,
                total_reward=reward,
                reward_components={} if reward_info is None else reward_info.components,
                stage_before=None if reward_info is None else reward_info.before.stage,
                stage_after=None if reward_info is None else reward_info.after.stage,
                phi_before=None if reward_info is None else reward_info.before.total,
                phi_after=None if reward_info is None else reward_info.after.total,
            )
        )
        self._q_demo_state['step_count'] += 1
        self._refresh_state()
        self._refresh_action_list()
        self.update_idletasks()
        if reward_info is not None and reward_info.after.stage == 'done':
            self._log(f'[Q演示] 击败队长/完成目标，HP={self.env.player.hp}')
            self.train_status.config(text='Q 演示结束（完成）', fg='green')
            return
        if ending != 'continue' or self._q_demo_state['step_count'] >= self._q_demo_state['max_steps']:
            self._log(f'[Q演示] 结束: {ending} steps={self._q_demo_state["step_count"]} hp={self.env.player.hp}')
            self.train_status.config(text='Q 演示结束', fg='gray')
            return
        self._q_after_id = self.after(self.train_speed_ms, self._q_demo_step)

    # ------------------------------------------------------------------------
    #  PPO 训练 - 生成器模式，支持动态演示
    # ------------------------------------------------------------------------
    def _start_train_generator(self):
        if self.training or self.q_training:
            messagebox.showwarning('提示', '训练正在进行中')
            return
        if not self._ensure_rl_components():
            return
        try:
            rounds = int(self.rounds_var.get())
        except ValueError:
            messagebox.showerror('错误', '回合数必须是整数')
            return
        if rounds <= 0:
            return

        save_path = self.model_path_var.get()
        gnn_encoder = MapGNNEncoder(output_dim=64, hidden_dim=128, num_layers=3)
        self.agent = PPO(emb_map_dim=64, emb_state_dim=32, emb_action_dim=32, gnn_encoder=gnn_encoder)
        if os.path.exists(save_path):
            try:
                self.agent.load(save_path)
                self._log(f'[训练] 已加载模型: {save_path}')
            except Exception as e:
                self._log(f'[训练] 加载模型失败: {e}')

        self.training = True
        self.stop_training = False
        self.train_btn.config(state='disabled')
        self.train_status.config(text='训练中...', fg='green')
        self._log(f'[训练] 开始 {rounds} 回合训练，速度间隔 {self.train_speed_ms}ms')

        # 训练状态
        self._train_state = {
            'rounds': rounds,
            'episode': 0,
            'step_count': 0,
            'max_steps': 180,
            'sword_collected': False,
            'best_hp': 0,
            'save_path': save_path,
        }
        self._train_episode_start()

    def _train_episode_start(self):
        if self.stop_training or self._train_state['episode'] >= self._train_state['rounds']:
            self._train_finish()
            return
        self.env.reset(refresh_frame=True)
        self._reset_reward_monitor(note=f'训练 Episode {self._train_state["episode"] + 1} 开始；reward 从 0 重新累计。')
        self._refresh_state()
        self._train_state['step_count'] = 0
        self._train_recent_positions = [self.env.n2p[self.env.observation[-1]][:3]]
        self._train_last_action_was_stair = False
        self._train_step_loop()

    def _train_step_loop(self):
        if self.stop_training:
            self._train_finish()
            return

        actions = self.env.get_feasible_actions()
        actions = self._filter_train_stair_loops(actions)
        if not actions:
            self._train_episode_end('stop')
            return

        action, info = self.agent.choose_action(self.env, actions)
        before = self.env.get_player_state()
        before_pos = self.env.n2p[self.env.observation[-1]][:3]

        # 执行动作并刷新画面（动态演示核心）
        ending = self.env.step(action, refresh_frame=True)
        after_pos = self.env.n2p[self.env.observation[-1]][:3]
        action_pos = self.env.n2p.get(action)
        was_stair = bool(action_pos is not None and getattr(action, 'id', '') in {'upFloor', 'downFloor'})
        self._refresh_state()
        self.update_idletasks()

        after = self.env.get_player_state()
        if ending == 'stop':
            base_reward = -9999.0
        else:
            base_reward = float(np.sum((after - before) * PPO_REWARD_RATE))
        reward = compute_sword_reward(self.env, base_reward, ending, action)
        stair_penalty = 0.0
        done = (ending != 'continue')

        # 拿到剑
        if self.env.n2p[action] == (4, 11, 11):
            self._train_state['sword_collected'] = True
            done = True
            if self.env.player.hp > self._train_state['best_hp']:
                self._train_state['best_hp'] = self.env.player.hp
            self._log(f'[Episode {self._train_state["episode"] + 1}] 拿到剑！生命: {self.env.player.hp}')

        next_step_count = self._train_state['step_count'] + 1
        timeout_penalty = 0.0
        if not done and next_step_count >= self._train_state.get('max_steps', 180):
            timeout_penalty = 500.0
            reward -= timeout_penalty
            done = True
            ending = 'timeout'

        self._record_reward(
            self._make_reward_row(
                action,
                before,
                after,
                ending,
                before_pos,
                after_pos,
                source='训练',
                base_reward=base_reward,
                total_reward=reward,
                stair_penalty=stair_penalty,
                timeout_penalty=timeout_penalty,
            )
        )

        self.agent.store_transition(info, reward, done)
        self._train_state['step_count'] = next_step_count
        self._train_last_action_was_stair = was_stair
        self._train_recent_positions.append(after_pos)
        self._train_recent_positions = self._train_recent_positions[-10:]

        if done:
            self._train_episode_end(ending)
        else:
            # 继续下一步，延迟由速度滑块控制
            self._train_after_id = self.after(self.train_speed_ms, self._train_step_loop)

    def _train_episode_end(self, ending):
        updated = self.agent.end_episode()
        ep = self._train_state['episode'] + 1
        status = 'UPDATED' if updated else f'hp={self.env.player.hp}'
        self._log(f'[Episode {ep}/{self._train_state["rounds"]}] {status} '
                  f'steps={self._train_state["step_count"]} ending={ending}')
        self.train_status.config(text=f'Episode {ep}/{self._train_state["rounds"]}')

        self._train_state['episode'] = ep
        self._train_after_id = self.after(self.train_speed_ms, self._train_episode_start)

    def _train_finish(self):
        self.training = False
        self.train_btn.config(state='normal')
        if self.agent:
            self.agent.update()
            path = self._train_state['save_path']
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            self.agent.save(path)
            self._log(f'[训练] 完成，模型已保存至: {path}')
            if self._train_state['sword_collected']:
                self._log(f'[训练] 成功拿到剑！最佳生命: {self._train_state["best_hp"]}')
            else:
                self._log('[训练] 未拿到剑，建议增加回合数')
        self.train_status.config(text='训练结束', fg='gray')

    def _stop_training(self):
        self.stop_training = True
        if self._train_after_id:
            self.after_cancel(self._train_after_id)
            self._train_after_id = None
        self._log('[训练] 已请求停止')
        self.train_status.config(text='已停止', fg='red')
        self.train_btn.config(state='normal')
        self.training = False

    # ------------------------------------------------------------------------
    #  贪婪演示 - 同样使用 after 动态演示
    # ------------------------------------------------------------------------
    def _start_demo(self):
        if self.training or self.q_training:
            messagebox.showwarning('提示', '请先停止训练')
            return
        if not self._ensure_rl_components():
            return
        save_path = self.model_path_var.get()
        if not os.path.exists(save_path):
            messagebox.showwarning('提示', f'模型不存在: {save_path}')
            return
        self.mode_var.set('demo')
        self._switch_mode()

        gnn_encoder = MapGNNEncoder(output_dim=64, hidden_dim=128, num_layers=3)
        agent = PPO(emb_map_dim=64, emb_state_dim=32, emb_action_dim=32, gnn_encoder=gnn_encoder)
        agent.load(save_path)
        self.env.reset(refresh_frame=True)
        self._reset_reward_monitor(note='贪婪演示开始；显示模型每一步对应的当前 reward。')
        self._refresh_state()
        self._log('[演示] 开始贪婪策略演示')
        self.train_status.config(text='演示中...', fg='blue')
        self._demo_agent = agent
        self._demo_recent_positions = [self.env.n2p[self.env.observation[-1]][:3]]
        self._demo_last_action_was_stair = False
        self._demo_step()

    def _demo_step(self):
        actions = self.env.get_feasible_actions()
        actions = self._filter_demo_stair_loops(actions)
        if not actions:
            self._log('[演示] 无可行动作，演示结束')
            self.train_status.config(text='演示结束', fg='gray')
            return
        action = self._demo_agent.greedy_action(self.env, actions)
        before = self.env.get_player_state().copy()
        before_pos = self.env.n2p[self.env.observation[-1]][:3]
        ending = self.env.step(action, refresh_frame=True)
        after = self.env.get_player_state().copy()
        after_pos = self.env.n2p[self.env.observation[-1]][:3]
        base_reward = self._base_reward(before, after, ending)
        reward = self._local_sword_reward_before_penalty(action, base_reward, ending, before_pos, after_pos)
        self._record_reward(
            self._make_reward_row(
                action,
                before,
                after,
                ending,
                before_pos,
                after_pos,
                source='演示',
                base_reward=base_reward,
                total_reward=reward,
            )
        )
        action_pos = self.env.n2p.get(action)
        self._demo_last_action_was_stair = bool(
            action_pos is not None and getattr(action, 'id', '') in {'upFloor', 'downFloor'}
        )
        self._demo_recent_positions.append(self.env.n2p[self.env.observation[-1]][:3])
        self._demo_recent_positions = self._demo_recent_positions[-8:]
        self._refresh_state()
        self.update_idletasks()

        if self.env.n2p[action] == (4, 11, 11):
            self._log(f'[演示] 成功拿到剑！生命: {self.env.player.hp}')
            self.train_status.config(text='演示结束（拿到剑）', fg='green')
            return
        if ending != 'continue':
            self._log(f'[演示] 结束: {ending} 生命: {self.env.player.hp}')
            self.train_status.config(text=f'演示结束 ({ending})', fg='gray')
            return
        self._train_after_id = self.after(self.train_speed_ms, self._demo_step)

    def _filter_demo_stair_loops(self, actions):
        """Avoid the common greedy-policy 2F/3F stair bounce during demos."""
        return self._filter_stair_loop_actions(
            actions,
            recent_positions=getattr(self, '_demo_recent_positions', []),
            last_action_was_stair=getattr(self, '_demo_last_action_was_stair', False),
            label='演示',
        )

    def _filter_train_stair_loops(self, actions):
        return self._filter_stair_loop_actions(
            actions,
            recent_positions=getattr(self, '_train_recent_positions', []),
            last_action_was_stair=getattr(self, '_train_last_action_was_stair', False),
            label='训练',
        )

    def _filter_stair_loop_actions(self, actions, recent_positions, last_action_was_stair, label):
        if not actions or not self.env or not self.env.observation:
            return actions
        current_pos = self.env.n2p[self.env.observation[-1]][:3]
        recent = set((recent_positions or [])[-6:])

        def is_stair(action):
            return getattr(action, 'id', '') in {'upFloor', 'downFloor'}

        def action_pos(action):
            pos = self.env.n2p.get(action)
            return None if pos is None else pos[:3]

        filtered = []
        skipped = []
        for action in actions:
            pos = action_pos(action)
            loop_like = False
            if pos is not None and is_stair(action):
                # After arriving at a paired stair, the reverse action is usually
                # the current node itself. If there are other choices, do not let
                # the policy immediately undo the previous floor transition.
                if last_action_was_stair and pos == current_pos:
                    loop_like = True
                elif pos in recent and len(recent) >= 2:
                    loop_like = True
            if loop_like:
                skipped.append(action)
            else:
                filtered.append(action)
        if filtered and skipped:
            self._set_status(f'[{label}] 已临时过滤 {len(skipped)} 个楼梯回环动作，避免反复上下楼。')
            return filtered
        return actions

    def _train_stair_loop_reward_penalty(self, action, before_pos, after_pos):
        # Kept for compatibility with old logs. Stair loops are now handled as
        # action filtering/masking, not reward shaping.
        return 0.0

    def _show_story_messages(self, messages):
        if not messages:
            return
        text = '\n\n'.join(messages)
        self._log('[剧情] ' + ' / '.join(messages))
        messagebox.showinfo('剧情事件', text)


# ============================================================================
#  主程序
# ============================================================================
if __name__ == '__main__':
    app = App()
    app.mainloop()
