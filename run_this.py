# -*- coding: utf-8 -*-
"""
10层魔塔可视化界面 V2
支持 PPO 训练动态演示 + 手动节点选择
"""
import os
import time
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

import numpy as np
from animation_environment import Mota
from PPO import PPO, compute_sword_reward, PPO_REWARD_RATE
from GNN import MapGNNEncoder


# ============================================================================
#  主窗口
# ============================================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('10层魔塔 - PPO 训练与可视化')
        self.geometry('1200x800')
        self.minsize(1000, 700)

        # 环境
        self.env = None
        # PPO agent
        self.agent = None
        # 训练控制
        self.training = False
        self.stop_training = False
        self.train_speed_ms = 50   # 每步间隔毫秒
        self._train_after_id = None
        # 手动模式
        self.action_list = []
        self.selected_action_idx = -1

        self._build_ui()
        self._init_env()

    # ------------------------------------------------------------------------
    #  构建界面
    # ------------------------------------------------------------------------
    def _build_ui(self):
        # 顶部工具栏
        toolbar = tk.Frame(self, bg='#e0e0e0', height=40)
        toolbar.pack(fill=tk.X, side=tk.TOP)
        toolbar.pack_propagate(False)

        tk.Label(toolbar, text='模式:', bg='#e0e0e0', font=('微软雅黑', 10)).pack(side=tk.LEFT, padx=5)
        self.mode_var = tk.StringVar(value='manual')
        tk.Radiobutton(toolbar, text='手动控制', variable=self.mode_var, value='manual',
                       bg='#e0e0e0', command=self._switch_mode).pack(side=tk.LEFT, padx=2)
        tk.Radiobutton(toolbar, text='PPO训练', variable=self.mode_var, value='train',
                       bg='#e0e0e0', command=self._switch_mode).pack(side=tk.LEFT, padx=2)
        tk.Radiobutton(toolbar, text='贪婪演示', variable=self.mode_var, value='demo',
                       bg='#e0e0e0', command=self._switch_mode).pack(side=tk.LEFT, padx=2)

        tk.Label(toolbar, text='| 速度:', bg='#e0e0e0').pack(side=tk.LEFT, padx=10)
        self.speed_scale = tk.Scale(toolbar, from_=0, to=500, orient=tk.HORIZONTAL,
                                    length=150, showvalue=True, resolution=10,
                                    command=self._on_speed_change)
        self.speed_scale.set(50)
        self.speed_scale.pack(side=tk.LEFT, padx=5)

        tk.Button(toolbar, text='重置环境', command=self._reset_env).pack(side=tk.LEFT, padx=10)
        tk.Button(toolbar, text='楼层▼', command=self._floor_down).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text='楼层▲', command=self._floor_up).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text='连线切换', command=self._toggle_lines).pack(side=tk.LEFT, padx=2)

        # 主体左右分栏
        main = tk.Frame(self)
        main.pack(fill=tk.BOTH, expand=True)

        # 左侧地图
        left_frame = tk.Frame(main, bg='#333333')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        left_frame.rowconfigure(0, weight=1)
        left_frame.columnconfigure(0, weight=1)
        self.map_frame = left_frame

        # 右侧面板
        right_frame = tk.Frame(main, width=360, bg='#f5f5f5')
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        right_frame.pack_propagate(False)

        # --- 状态区 ---
        state_frame = tk.LabelFrame(right_frame, text='角色状态', font=('微软雅黑', 10, 'bold'))
        state_frame.pack(fill=tk.X, padx=5, pady=3)
        self.state_labels = {}
        state_items = ['HP', 'ATK', 'DEF', 'MDEF', 'Money', 'Exp', '黄钥匙', '蓝钥匙', '红钥匙', '楼层']
        for i, name in enumerate(state_items):
            lbl = tk.Label(state_frame, text=f'{name}: --', font=('Consolas', 10))
            lbl.grid(row=i // 2, column=i % 2, sticky='w', padx=5, pady=1)
            self.state_labels[name] = lbl

        # --- 手动模式面板 ---
        self.manual_panel = tk.LabelFrame(right_frame, text='手动控制 - 选择可行动作',
                                          font=('微软雅黑', 10, 'bold'))
        self.manual_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=3)

        # 动作列表 Treeview
        cols = ('序号', '坐标', '类别', 'ID')
        self.action_tree = ttk.Treeview(self.manual_panel, columns=cols,
                                        show='headings', height=10)
        for c in cols:
            self.action_tree.heading(c, text=c)
            self.action_tree.column(c, width=70 if c == '序号' else 90, anchor='center')
        self.action_tree.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
        self.action_tree.bind('<<TreeviewSelect>>', self._on_action_select)

        btn_frm = tk.Frame(self.manual_panel)
        btn_frm.pack(fill=tk.X, padx=3, pady=3)
        tk.Button(btn_frm, text='执行选中', bg='#4CAF50', fg='white',
                  command=self._execute_selected).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        tk.Button(btn_frm, text='刷新列表', command=self._refresh_action_list).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frm, text='回退一步', command=self._back_step).pack(side=tk.LEFT, padx=2)

        # --- 训练/演示面板 ---
        self.train_panel = tk.LabelFrame(right_frame, text='PPO 训练',
                                         font=('微软雅黑', 10, 'bold'))
        # 默认隐藏，切换模式时显示

        tk.Label(self.train_panel, text='训练回合数:').grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.rounds_var = tk.StringVar(value='500')
        tk.Entry(self.train_panel, textvariable=self.rounds_var, width=10).grid(row=0, column=1, sticky='w', padx=5)

        tk.Label(self.train_panel, text='模型路径:').grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.model_path_var = tk.StringVar(value='model/ppo_10floor.pth')
        tk.Entry(self.train_panel, textvariable=self.model_path_var, width=28).grid(row=1, column=1, sticky='w', padx=5)

        btn_frm2 = tk.Frame(self.train_panel)
        btn_frm2.grid(row=2, column=0, columnspan=2, pady=5)
        self.train_btn = tk.Button(btn_frm2, text='开始训练', bg='#4CAF50', fg='white', width=10,
                                   command=self._start_train_generator)
        self.train_btn.pack(side=tk.LEFT, padx=3)
        tk.Button(btn_frm2, text='停止训练', bg='#f44336', fg='white', width=10,
                  command=self._stop_training).pack(side=tk.LEFT, padx=3)
        tk.Button(btn_frm2, text='贪婪演示', bg='#2196F3', fg='white', width=10,
                  command=self._start_demo).pack(side=tk.LEFT, padx=3)

        self.train_log = scrolledtext.ScrolledText(self.train_panel, height=10, width=40, font=('Consolas', 9))
        self.train_log.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky='nsew')
        self.train_panel.rowconfigure(3, weight=1)

        # 训练进度标签
        self.train_status = tk.Label(self.train_panel, text='就绪', fg='gray',
                                     font=('微软雅黑', 9, 'bold'))
        self.train_status.grid(row=4, column=0, columnspan=2, sticky='w', padx=5)

        # 默认显示手动面板
        self._switch_mode()

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
        self._refresh_state()
        self._refresh_action_list()

    def _refresh_state(self):
        p = self.env.player
        z, y, x = self.env.n2p[self.env.observation[-1]]
        values = {
            'HP': f'{p.hp}', 'ATK': f'{p.atk}', 'DEF': f'{p.def_}',
            'MDEF': f'{p.mdef}', 'Money': f'{p.money}', 'Exp': f'{p.exp}',
            '黄钥匙': f'{p.items.get("yellowKey", 0)}',
            '蓝钥匙': f'{p.items.get("blueKey", 0)}',
            '红钥匙': f'{p.items.get("redKey", 0)}',
            '楼层': f'{z + 1}F',
        }
        for name, val in values.items():
            self.state_labels[name].config(text=f'{name}: {val}')

    def _log(self, msg):
        self.train_log.insert(tk.END, msg + '\n')
        self.train_log.see(tk.END)

    # ------------------------------------------------------------------------
    #  模式切换
    # ------------------------------------------------------------------------
    def _switch_mode(self):
        mode = self.mode_var.get()
        if mode == 'manual':
            self.manual_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=3)
            self.train_panel.pack_forget()
            if self.env:
                self._refresh_action_list()
        else:
            self.manual_panel.pack_forget()
            self.train_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=3)

    def _on_speed_change(self, val):
        self.train_speed_ms = int(val)

    # ------------------------------------------------------------------------
    #  地图控制
    # ------------------------------------------------------------------------
    def _reset_env(self):
        self.env.reset(refresh_frame=True)
        self._refresh_state()
        if self.mode_var.get() == 'manual':
            self._refresh_action_list()
        self._log('[系统] 环境已重置')

    def _floor_up(self):
        f = self.env.anima_frame.now_floor + 1
        if f <= self.env.anima_frame.max_floor_num:
            self.env.anima_frame.change_floor(f)

    def _floor_down(self):
        f = self.env.anima_frame.now_floor - 1
        if f >= 0:
            self.env.anima_frame.change_floor(f)

    def _toggle_lines(self):
        vis = not self.env.anima_frame.line_visible
        self.env.anima_line_visible(vis)

    # ------------------------------------------------------------------------
    #  手动模式 - 动作列表
    # ------------------------------------------------------------------------
    def _refresh_action_list(self):
        for item in self.action_tree.get_children():
            self.action_tree.delete(item)
        self.action_list = self.env.get_feasible_actions()
        for idx, action in enumerate(self.action_list, 1):
            pos = self.env.n2p[action]
            self.action_tree.insert('', 'end', values=(idx, str(pos), action.class_, action.id))
        self.selected_action_idx = -1

    def _on_action_select(self, event):
        sel = self.action_tree.selection()
        if not sel:
            return
        values = self.action_tree.item(sel[0], 'values')
        idx = int(values[0]) - 1
        self.selected_action_idx = idx
        if 0 <= idx < len(self.action_list):
            pos = self.env.n2p[self.action_list[idx]]
            self.env.anima_frame.show_cursor(pos)

    def _execute_selected(self):
        if self.selected_action_idx < 0 or self.selected_action_idx >= len(self.action_list):
            messagebox.showwarning('提示', '请先从列表中选择一个动作')
            return
        action = self.action_list[self.selected_action_idx]
        ending = self.env.step(action, refresh_frame=True)
        self._refresh_state()
        self._refresh_action_list()
        pos = self.env.n2p[action]
        self._log(f'行动至 {pos} -> {ending}')
        if ending != 'continue':
            messagebox.showinfo('回合结束', f'结束状态: {ending}\n剩余生命: {self.env.player.hp}')

    def _back_step(self):
        if len(self.env.observation) <= 1:
            return
        self.env.back_step(1, refresh_frame=True)
        self._refresh_state()
        self._refresh_action_list()

    # ------------------------------------------------------------------------
    #  PPO 训练 - 生成器模式，支持动态演示
    # ------------------------------------------------------------------------
    def _start_train_generator(self):
        if self.training:
            messagebox.showwarning('提示', '训练正在进行中')
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
        self._refresh_state()
        self._train_state['step_count'] = 0
        self._train_step_loop()

    def _train_step_loop(self):
        if self.stop_training:
            self._train_finish()
            return

        actions = self.env.get_feasible_actions()
        if not actions:
            self._train_episode_end('stop')
            return

        action, info = self.agent.choose_action(self.env, actions)
        before = self.env.get_player_state()

        # 执行动作并刷新画面（动态演示核心）
        ending = self.env.step(action, refresh_frame=True)
        self._refresh_state()
        self.update_idletasks()

        after = self.env.get_player_state()
        if ending == 'stop':
            base_reward = -9999.0
        else:
            base_reward = float(np.sum((after - before) * PPO_REWARD_RATE))
        reward = compute_sword_reward(self.env, base_reward, ending, action)
        done = (ending != 'continue')

        # 拿到剑
        if self.env.n2p[action] == (4, 11, 11):
            self._train_state['sword_collected'] = True
            done = True
            if self.env.player.hp > self._train_state['best_hp']:
                self._train_state['best_hp'] = self.env.player.hp
            self._log(f'[Episode {self._train_state["episode"] + 1}] 拿到剑！生命: {self.env.player.hp}')

        self.agent.store_transition(info, reward, done)
        self._train_state['step_count'] += 1

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
        if self.training:
            messagebox.showwarning('提示', '请先停止训练')
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
        self._refresh_state()
        self._log('[演示] 开始贪婪策略演示')
        self.train_status.config(text='演示中...', fg='blue')
        self._demo_agent = agent
        self._demo_step()

    def _demo_step(self):
        actions = self.env.get_feasible_actions()
        if not actions:
            self._log('[演示] 无可行动作，演示结束')
            self.train_status.config(text='演示结束', fg='gray')
            return
        action = self._demo_agent.greedy_action(self.env, actions)
        ending = self.env.step(action, refresh_frame=True)
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


# ============================================================================
#  主程序
# ============================================================================
if __name__ == '__main__':
    app = App()
    app.mainloop()
