# -*- coding: utf-8 -*-
"""Lightweight Magic Tower window for manual play.

This legacy window is kept only as a minimal runnable visualization. Old
automated learning and policy playback modules were removed from this copy.
"""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

from animation_environment import Mota  # noqa: E402


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("10层魔塔 - 手动可视化")
        self.geometry("1200x800")
        self.minsize(1000, 700)

        self.env = None
        self.action_list = []
        self.selected_action_idx = -1

        self._build_ui()
        self._init_env()

    def _build_ui(self):
        toolbar = tk.Frame(self, bg="#e0e0e0", height=40)
        toolbar.pack(fill=tk.X, side=tk.TOP)
        toolbar.pack_propagate(False)

        tk.Label(toolbar, text="手动游玩 / 复现窗口", bg="#e0e0e0", font=("Arial", 11, "bold")).pack(
            side=tk.LEFT, padx=10
        )
        tk.Button(toolbar, text="重置环境", command=self._reset_env).pack(side=tk.LEFT, padx=10)
        tk.Button(toolbar, text="楼层▼", command=self._floor_down).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="楼层▲", command=self._floor_up).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="连线切换", command=self._toggle_lines).pack(side=tk.LEFT, padx=2)

        main = tk.Frame(self)
        main.pack(fill=tk.BOTH, expand=True)

        left_frame = tk.Frame(main, bg="#333333")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        left_frame.rowconfigure(0, weight=1)
        left_frame.columnconfigure(0, weight=1)
        self.map_frame = left_frame

        right_frame = tk.Frame(main, width=380, bg="#f5f5f5")
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        right_frame.pack_propagate(False)

        state_frame = tk.LabelFrame(right_frame, text="角色状态", font=("Arial", 10, "bold"))
        state_frame.pack(fill=tk.X, padx=5, pady=3)
        self.state_labels = {}
        state_items = ["HP", "ATK", "DEF", "Money", "Exp", "黄钥匙", "蓝钥匙", "红钥匙", "楼层"]
        for i, name in enumerate(state_items):
            label = tk.Label(state_frame, text=f"{name}: --", font=("Consolas", 10))
            label.grid(row=i // 2, column=i % 2, sticky="w", padx=5, pady=1)
            self.state_labels[name] = label

        action_panel = tk.LabelFrame(right_frame, text="可行动作", font=("Arial", 10, "bold"))
        action_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=3)

        cols = ("序号", "坐标", "类别", "ID")
        self.action_tree = ttk.Treeview(action_panel, columns=cols, show="headings", height=14)
        for col in cols:
            self.action_tree.heading(col, text=col)
            self.action_tree.column(col, width=70 if col == "序号" else 95, anchor="center")
        self.action_tree.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
        self.action_tree.bind("<<TreeviewSelect>>", self._on_action_select)

        btn_frame = tk.Frame(action_panel)
        btn_frame.pack(fill=tk.X, padx=3, pady=3)
        tk.Button(btn_frame, text="执行选中", bg="#4CAF50", fg="white", command=self._execute_selected).pack(
            side=tk.LEFT, padx=2, fill=tk.X, expand=True
        )
        tk.Button(btn_frame, text="刷新列表", command=self._refresh_action_list).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="回退一步", command=self._back_step).pack(side=tk.LEFT, padx=2)

        self.log = scrolledtext.ScrolledText(right_frame, height=8, font=("Consolas", 9))
        self.log.pack(fill=tk.X, padx=5, pady=3)

    def _init_env(self):
        self.env = Mota(self.map_frame)
        self.env.build_env("10層魔塔")
        self.env.create_nodes()
        self.env.create_map()
        self.env.anima_frame.grid(row=0, column=0, sticky="nsew")
        self.env.build_anima_frame(bg="#333333")
        self._refresh_state()
        self._refresh_action_list()

    def _refresh_state(self):
        player = self.env.player
        floor, _y, _x = self.env.n2p[self.env.observation[-1]]
        values = {
            "HP": player.hp,
            "ATK": player.atk,
            "DEF": player.def_,
            "Money": player.money,
            "Exp": player.exp,
            "黄钥匙": player.items.get("yellowKey", 0),
            "蓝钥匙": player.items.get("blueKey", 0),
            "红钥匙": player.items.get("redKey", 0),
            "楼层": f"{floor + 1}F",
        }
        for name, value in values.items():
            self.state_labels[name].config(text=f"{name}: {value}")

    def _log(self, message: str):
        self.log.insert(tk.END, message + "\n")
        self.log.see(tk.END)

    def _reset_env(self):
        self.env.reset(refresh_frame=True)
        self._refresh_state()
        self._refresh_action_list()
        self._log("[系统] 环境已重置")

    def _floor_up(self):
        floor = self.env.anima_frame.now_floor + 1
        if floor <= self.env.anima_frame.max_floor_num:
            self.env.anima_frame.change_floor(floor)

    def _floor_down(self):
        floor = self.env.anima_frame.now_floor - 1
        if floor >= 0:
            self.env.anima_frame.change_floor(floor)

    def _toggle_lines(self):
        self.env.anima_line_visible(not self.env.anima_frame.line_visible)

    def _refresh_action_list(self):
        for item in self.action_tree.get_children():
            self.action_tree.delete(item)
        self.action_list = self.env.get_feasible_actions()
        for idx, action in enumerate(self.action_list, 1):
            position = self.env.n2p[action]
            self.action_tree.insert("", "end", values=(idx, str(position), action.class_, action.id))
        self.selected_action_idx = -1

    def _on_action_select(self, _event):
        selection = self.action_tree.selection()
        if not selection:
            return
        values = self.action_tree.item(selection[0], "values")
        self.selected_action_idx = int(values[0]) - 1
        if 0 <= self.selected_action_idx < len(self.action_list):
            self.env.anima_frame.show_cursor(self.env.n2p[self.action_list[self.selected_action_idx]])

    def _execute_selected(self):
        if self.selected_action_idx < 0 or self.selected_action_idx >= len(self.action_list):
            messagebox.showwarning("提示", "请先从列表中选择一个动作")
            return
        action = self.action_list[self.selected_action_idx]
        ending = self.env.step(action, refresh_frame=True)
        self._refresh_state()
        self._refresh_action_list()
        self._log(f"行动至 {self.env.n2p[action]} -> {ending}")
        if ending != "continue":
            messagebox.showinfo("回合结束", f"结束状态: {ending}\n剩余生命: {self.env.player.hp}")

    def _back_step(self):
        if len(self.env.observation) <= 1:
            return
        self.env.back_step(1, refresh_frame=True)
        self._refresh_state()
        self._refresh_action_list()
        self._log("[系统] 已回退一步")


if __name__ == "__main__":
    App().mainloop()
