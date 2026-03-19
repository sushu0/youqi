# -*- coding: utf-8 -*-
"""
产量劈分软件 V2.0
@author: Kang
"""
import json
import os
import threading
import time
import tkinter as tk
from tkinter import (Menu, Text, StringVar, Toplevel, Label, Entry,
                     filedialog, messagebox, ttk)

import numpy as np
import pandas as pd

try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    try:
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        print("未能设置中文字体'SimHei'，图表中文可能显示异常。")
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


CONFIG_FILE = "config.json"

# 默认列名映射
DEFAULT_COLUMN_MAPPINGS = {
    "tops_well_name": "Well name", "tops_top": "Top", "tops_bottom": "Bottom", "tops_surface": "Surface",
    "result_well_name": "Well name", "result_top": "Top", "result_bottom": "Bottom",
    "prod_well_name": "井号", "prod_date": "生产年月", "prod_section": "生产井段",
    "prod_oil": "月产油量", "prod_water": "月产水量", "prod_gas": "月产气量",
    "perf_well_name": "井号", "perf_date": "射孔日期", "perf_top": "射孔段顶深", "perf_bottom": "射孔段底深",
    "log_depth": "DEPTH", "log_perm": "PERM", "log_por": "POR"
}


class DataManager:
    """数据管理器"""
    def __init__(self, logger, mappings):
        self.logger = logger
        self.mappings = mappings
        self.well_logs_path = None
        self.well_tops_data = None
        self.result_data = None
        self.production_data = None
        self.perforation_data = None
        self.wells_in_logs = []

    def load_data(self, file_paths):
        """加载所有需要的数据文件。"""
        try:
            m = self.mappings
            self.logger("开始加载地质分层数据...")
            self.well_tops_data = pd.read_excel(file_paths['tops'])
            self.well_tops_data.sort_values(by=[m['tops_well_name'], m['tops_top']], ascending=True, inplace=True)

            self.logger("开始加载解释结论数据...")
            self.result_data = pd.read_excel(file_paths['result'])
            self.result_data.sort_values(by=[m['result_well_name'], m['result_top']], ascending=True, inplace=True)

            self.logger("开始加载生产月报数据...")
            self.production_data = pd.read_excel(file_paths['production'])
            self.production_data.dropna(axis=0, subset=[m['prod_well_name'], m['prod_section']], inplace=True)
            self.production_data['Date'] = pd.to_datetime(self.production_data[m['prod_date']].astype(str) + '01', format="%Y%m%d")
            self.production_data.sort_values(by=[m['prod_well_name'], m['prod_date']], ascending=True, inplace=True)

            self.logger("开始加载射孔数据...")
            self.perforation_data = pd.read_excel(file_paths['perforation'])
            self.perforation_data['Date'] = pd.to_datetime(self.perforation_data[m['perf_date']], format="%Y%m%d", errors='coerce')
            self.perforation_data.sort_values(by=[m['perf_well_name'], m['perf_date']], ascending=True, inplace=True)

            self.logger("正在检索测井曲线文件...")
            self.well_logs_path = file_paths['logs']
            self.wells_in_logs = [f for f in os.listdir(self.well_logs_path) if f.lower().endswith('.csv')]
            if not self.wells_in_logs: return False, "测井曲线文件夹中未找到任何.csv文件。"
            self.logger(f"检索到 {len(self.wells_in_logs)} 个测井曲线文件。")

            return True, "所有数据加载完毕。"
        except KeyError as e: return False, f"列名映射错误: 在某个文件中未找到名为 '{e.args[0]}' 的列。请检查列名映射设置。"
        except FileNotFoundError as e: return False, f"文件未找到：{e.filename}"
        except Exception as e: return False, f"加载数据时发生未知错误: {e}"

    def pre_check_data(self, split_method):
        """执行数据预检查，并生成一份数据质量报告。"""
        m = self.mappings
        report = ["", "--- 数据质量预检查报告 ---"]
        prod_wells = set(self.production_data[m['prod_well_name']].astype(str).unique())
        tops_wells = set(self.well_tops_data[m['tops_well_name']].astype(str).unique())
        perf_wells = set(self.perforation_data[m['perf_well_name']].astype(str).unique())
        log_wells_no_ext = set([os.path.splitext(f)[0] for f in self.wells_in_logs])

        def check_missing(label, base_set, check_set):
            missing = base_set - check_set
            if missing: report.append(f"-> 警告: {len(missing)}口生产井缺少{label}： {', '.join(list(missing)[:3])}...")
            return not missing

        all_ok = check_missing("分层数据", prod_wells, tops_wells)
        all_ok &= check_missing("射孔数据", prod_wells, perf_wells)
        all_ok &= check_missing("测井CSV文件", prod_wells, log_wells_no_ext)

        required_logs = [m['log_perm']]
        if '孔隙体积' in split_method: required_logs.append(m['log_por'])
        
        logs_to_check = [f for f in self.wells_in_logs if os.path.splitext(f)[0] in prod_wells]
        missing_cols_wells = {col: [] for col in required_logs}
        
        for log_file in logs_to_check[:10]: # 抽查前10个文件
            try:
                df = pd.read_csv(os.path.join(self.well_logs_path, log_file), nrows=1)
                for col in required_logs:
                    if col not in df.columns: missing_cols_wells[col].append(os.path.splitext(log_file)[0])
            except Exception: pass

        for col, wells in missing_cols_wells.items():
            if wells:
                report.append(f"-> 警告: 在抽查的测井文件中，以下井缺少'{col}'列: {', '.join(wells)}")
                all_ok = False
        
        if all_ok: report.append("-> 恭喜: 主要数据类型和所需列均能匹配，未发现明显缺失。")
        report.append("--- 预检查结束 ---\n")
        self.logger("\n".join(report), "info")
        return all_ok


class CalculationEngine:
    """计算引擎"""
    def __init__(self, data_manager, logger, update_progress, update_status, mappings, split_method):
        self.dm = data_manager
        self.logger = logger
        self.update_progress = update_progress
        self.update_status = update_status
        self.mappings = mappings
        self.split_method = split_method
        self.sampling_intervals = {}
        self.diagnostics_report = []

    def run_splitting(self, perm_limit, output_path):
        """执行产量劈分的主流程。"""
        try:
            self.update_status("步骤 1/2: 匹配生产层位及计算物性参数...")
            production_data_with_layers = self._match_layers_and_calc_params(perm_limit)
            self.logger("层位匹配与物性参数计算完成。")

            self.update_status("步骤 2/2: 执行产量劈分...")
            final_result = self._split_production(production_data_with_layers)
            self.logger("产量劈分计算完成。")
            
            # 保存结果和诊断报告
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            result_file = f"{output_path}/劈分结果_{timestamp}.xlsx"
            final_result.to_excel(result_file, index=False, engine='openpyxl')
            self.logger(f"计算结果已成功保存至: {result_file}")
            
            self._save_diagnostics_report(output_path, timestamp)

            return final_result
        except Exception as e:
            self.logger(f"计算过程中发生错误: {e}", "error")
            import traceback; self.logger(traceback.format_exc(), "error")
            return None
            
    def _match_layers_and_calc_params(self, perm_limit):
        """匹配每个生产时段的生产层位，并计算这些层的相关物性参数。"""
        m = self.mappings
        prod_data = self.dm.production_data.copy()
        prod_data['split_details'] = [[] for _ in range(len(prod_data))] # 初始化新列
        wells_in_tops = set(self.dm.well_tops_data[m['tops_well_name']].astype(str))
        wells_in_perf = set(self.dm.perforation_data[m['perf_well_name']].astype(str))
        total_rows = len(prod_data)

        for index, row in prod_data.iterrows():
            well_name, prod_date = str(row[m['prod_well_name']]), row['Date']
            if index % 10 == 0: self.update_progress(int((index / total_rows) * 50))
            
            if well_name not in wells_in_tops: 
                self.diagnostics_report.append({'井号': well_name, '年月': row[m['prod_date']], '问题': '分层数据不存在'})
                continue
            if well_name not in wells_in_perf: 
                self.diagnostics_report.append({'井号': well_name, '年月': row[m['prod_date']], '问题': '射孔数据不存在'})
                continue
            
            try:
                ding, di = [float(i) for i in row[m['prod_section']].split('-')]
                well_tops = self.dm.well_tops_data[self.dm.well_tops_data[m['tops_well_name']].astype(str) == well_name]
                well_perf = self.dm.perforation_data[self.dm.perforation_data[m['perf_well_name']].astype(str) == well_name]
                perf_before_date = well_perf[well_perf['Date'] < prod_date]
                perf_in_production = perf_before_date[(perf_before_date[m['perf_bottom']] > ding) & (perf_before_date[m['perf_top']] < di)]
                
                if perf_in_production.empty:
                    self.diagnostics_report.append({'井号': well_name, '年月': row[m['prod_date']], '问题': '该生产时段内无有效射孔记录'});
                    continue

                matched_tops_info = []
                unique_surfaces = set()
                for _, perf_row in perf_in_production.iterrows():
                    tops_interval = well_tops[(well_tops[m['tops_top']] <= perf_row[m['perf_top']]) & (well_tops[m['tops_bottom']] >= perf_row[m['perf_bottom']])]
                    for _, top_row in tops_interval.iterrows():
                        surface = top_row[m['tops_surface']]
                        if surface not in unique_surfaces:
                            params = self._calculate_layer_params(well_name, top_row, perf_in_production, perm_limit, row[m['prod_date']])
                            matched_tops_info.append(params)
                            unique_surfaces.add(surface)
                
                prod_data.at[index, 'split_details'] = matched_tops_info
            except (ValueError, IndexError): 
                self.diagnostics_report.append({'井号': well_name, '年月': row[m['prod_date']], '问题': '生产井段格式错误'})
            except Exception as e: 
                self.diagnostics_report.append({'井号': well_name, '年月': row[m['prod_date']], '问题': f'未知计算错误: {e}'})
        
        return prod_data

    def _calculate_layer_params(self, well_name, top_row, perforation_data, perm_limit, prod_date_for_report):
        """计算单个小层的物性参数字典。"""
        m, surface = self.mappings, top_row[self.mappings['tops_surface']]
        well_log_file = f"{well_name}.csv"
        log_files_lower = [f.lower() for f in self.dm.wells_in_logs]
        
        result_params = {'surface': surface, 'perm': 0, 'thickness': 0, 'por': 0, 'error': None}

        try: log_file_cased = self.dm.wells_in_logs[log_files_lower.index(well_log_file.lower())]
        except ValueError: result_params['error'] = '无测井数据'; return result_params
            
        try:
            log_data = pd.read_csv(os.path.join(self.dm.well_logs_path, log_file_cased))
            if m['log_perm'] not in log_data.columns: 
                result_params['error'] = '无渗透率曲线' 
                return result_params
            if '孔隙体积' in self.split_method and m['log_por'] not in log_data.columns: 
                result_params['error'] = '无孔隙度曲线'
                return result_params

            sampling_interval = self._get_sampling_interval(well_name, log_data)
            well_result = self.dm.result_data[self.dm.result_data[m['result_well_name']].astype(str) == well_name]
            if well_result.empty: 
                result_params['error'] = '无解释结论数据'
                return result_params

            layer_top, layer_bottom = top_row[m['tops_top']], top_row[m['tops_bottom']]
            total_numerator, total_denominator, por_numerator, por_denominator = 0, 0, 0, 0

            for _, perf_row in perforation_data.iterrows():
                overlap_top, overlap_bottom = max(layer_top, perf_row[m['perf_top']]), min(layer_bottom, perf_row[m['perf_bottom']])
                if overlap_top >= overlap_bottom: 
                    continue

                relevant_results = well_result[(well_result[m['result_bottom']] > overlap_top) & (well_result[m['result_top']] < overlap_bottom)]
                for _, res_row in relevant_results.iterrows():
                    res_top, res_bottom = max(overlap_top, res_row[m['result_top']]), min(overlap_bottom, res_row[m['result_bottom']])
                    if res_top >= res_bottom: 
                        continue
                    
                    log_section = log_data[(log_data[m['log_depth']] > res_top) & (log_data[m['log_depth']] <= res_bottom)]
                    valid_perm_data = log_section[log_section[m['log_perm']] > perm_limit]
                    if valid_perm_data.empty: 
                        continue
                    
                    effective_thickness = len(valid_perm_data) * sampling_interval
                    mean_perm = valid_perm_data[m['log_perm']].mean()
                    total_numerator += mean_perm * effective_thickness
                    total_denominator += effective_thickness
                    if '孔隙体积' in self.split_method:
                        mean_por = valid_perm_data[m['log_por']].mean()
                        por_numerator += mean_por * effective_thickness
                        por_denominator += effective_thickness

            if total_denominator > 0:
                result_params.update({
                    'perm': round(total_numerator / total_denominator, 3),
                    'thickness': round(total_denominator, 3),
                    'por': round(por_numerator / por_denominator, 3) if por_denominator > 0 else 0
                })
            else: result_params['error'] = '无有效储层'
            return result_params
        except Exception as e:
            result_params['error'] = f'计算异常: {e}'; return result_params
    
    def _split_production(self, production_data):
        """根据选择的方法和计算出的物性进行产量劈分。"""
        m = self.mappings
        final_rows = []
        total_rows = len(production_data)
        prod_oil, prod_water, prod_gas = m['prod_oil'], m['prod_water'], m['prod_gas']

        for index, row in production_data.iterrows():
            self.update_progress(50 + int((index / total_rows) * 50))
            split_details = row['split_details']
            if not split_details: continue

            total_basis = 0
            for detail in split_details:
                if detail['error'] is None:
                    if self.split_method == '渗透率厚度积 (Kh)': basis = detail['perm'] * detail['thickness']
                    elif self.split_method == '仅有效厚度 (h)': basis = detail['thickness']
                    elif self.split_method == '孔隙体积 (φh)': basis = detail['por'] * detail['thickness']
                    else: basis = 0
                    detail['basis'] = basis
                    total_basis += basis
            
            if total_basis <= 0:
                self.diagnostics_report.append({'井号': row[m['prod_well_name']], '年月': row[m['prod_date']], '问题': '总劈分依据为零或负'})
                continue

            for detail in split_details:
                if detail['error'] is not None:
                    self.diagnostics_report.append({'井号': row[m['prod_well_name']], '年月': row[m['prod_date']], '层位': detail['surface'], '问题': detail['error']})
                    continue

                ratio = detail.get('basis', 0) / total_basis
                final_rows.append({
                    "井号": row[m['prod_well_name']], "生产年月": row[m['prod_date']],
                    "月产油量": row.get(prod_oil, 0), "月产水量": row.get(prod_water, 0), "月产气量": row.get(prod_gas, 0),
                    "层位": detail['surface'], "渗透率": detail['perm'], "有效厚度": detail['thickness'],
                    "孔隙度": detail['por'], "劈分依据": f"{detail['basis']:.2f}", "劈分系数": round(ratio, 4),
                    "劈分月产油量": round(row.get(prod_oil, 0) * ratio, 2),
                    "劈分月产水量": round(row.get(prod_water, 0) * ratio, 2),
                    "劈分月产气量": round(row.get(prod_gas, 0) * ratio, 2)
                })
        return pd.DataFrame(final_rows)

    def _get_sampling_interval(self, well_name, log_data):
        m = self.mappings
        if well_name in self.sampling_intervals: return self.sampling_intervals[well_name]
        interval = 0.125
        if m['log_depth'] in log_data.columns and len(log_data[m['log_depth']]) > 1:
            avg_diff = np.nanmean(np.diff(log_data[m['log_depth']]))
            if 0.01 < avg_diff < 1.0: interval = avg_diff
        self.sampling_intervals[well_name] = interval
        return interval

    def _save_diagnostics_report(self, output_path, timestamp):
        if not self.diagnostics_report:
            self.logger("未发现计算问题，不生成诊断报告。")
            return
        
        report_df = pd.DataFrame(self.diagnostics_report).drop_duplicates()
        report_file = f"{output_path}/诊断报告_{timestamp}.xlsx"
        report_df.to_excel(report_file, index=False, engine='openpyxl')
        self.logger(f"发现 {len(report_df)} 条计算问题，已生成诊断报告: {report_file}", "warning")


class ProductionSplittingApp:
    """产量劈分软件主程序"""
    def __init__(self, root):
        self.root = root
        self.root.title("产量劈分软件 V2.0")
        self.root.geometry("950x750")
        
        try: s = ttk.Style(); s.theme_use('clam')
        except tk.TclError: self.log_message("未能加载'clam'主题，使用默认主题。", "warning")

        self.file_paths = {k: StringVar() for k in ['logs', 'tops', 'result', 'production', 'perforation']}
        self.perm_limit = StringVar(value="0.1")
        self.split_method = StringVar(value='渗透率厚度积 (Kh)')
        self.column_mappings = DEFAULT_COLUMN_MAPPINGS.copy()
        self.last_output_dir = ""
        self.status_text = StringVar(value="就绪")
        self.result_df = None # 存储结果
        
        self.chart_canvas = self.dynamic_chart_canvas = None

        self._create_menu()
        self._create_widgets()
        self._display_welcome_message()
        self._load_config()
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _create_menu(self):
        menubar = Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="开始劈分", command=self.start_calculation_thread)
        file_menu.add_command(label="打开结果目录", command=self._open_result_folder, state='disabled')
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self._on_closing)
        self.file_menu = file_menu
        
        settings_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="高级设置", menu=settings_menu)
        settings_menu.add_command(label="列名映射...", command=self._open_column_mapping_window)
        settings_menu.add_command(label="重置所有设置", command=self._reset_all)

        help_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="关于...", command=self._show_about)

    def _create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10"); main_frame.pack(fill=tk.BOTH, expand=True)

        settings_frame = ttk.LabelFrame(main_frame, text="输入设置", padding="10"); settings_frame.pack(fill=tk.X, expand=False, pady=5)
        
        file_info = {
            'logs': ('测井曲线文件夹:', '(.csv文件目录)'), 'tops': ('地质分层文件:', '(.xlsx, .xls)'),
            'result': ('解释结论文件:', '(.xlsx, .xls)'), 'production': ('单井月报文件:', '(.xlsx, .xls)'),
            'perforation': ('射孔数据文件:', '(.xlsx, .xls)')
        }
        for i, (key, (label, hint)) in enumerate(file_info.items()):
            ttk.Label(settings_frame, text=label).grid(row=i, column=0, sticky='w', padx=5, pady=2)
            ttk.Entry(settings_frame, textvariable=self.file_paths[key], width=80).grid(row=i, column=1, columnspan=2, sticky='we', padx=5)
            ttk.Label(settings_frame, text=hint, foreground="gray").grid(row=i, column=3, sticky='w', padx=5)
            cmd = lambda k=key, d='folder' in label: self._select_path(k, d)
            ttk.Button(settings_frame, text="浏览...", command=cmd, width=10).grid(row=i, column=4, padx=5)
        
        param_frame = ttk.Frame(settings_frame); param_frame.grid(row=len(file_info), column=0, columnspan=5, sticky='w', pady=5)
        ttk.Label(param_frame, text="渗透率下限:").pack(side=tk.LEFT, padx=(5,0))
        ttk.Entry(param_frame, textvariable=self.perm_limit, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(param_frame, text="劈分方法:").pack(side=tk.LEFT, padx=(20,0))
        ttk.Combobox(param_frame, textvariable=self.split_method, values=['渗透率厚度积 (Kh)', '仅有效厚度 (h)', '孔隙体积 (φh)'], state='readonly', width=18).pack(side=tk.LEFT, padx=5)
        settings_frame.columnconfigure(1, weight=1)

        output_frame = ttk.Frame(main_frame); output_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        notebook = ttk.Notebook(output_frame); notebook.pack(fill=tk.BOTH, expand=True)
        self.notebook = notebook
        
        log_frame = ttk.Frame(notebook, padding="5")
        self.log_text = Text(log_frame, height=10, wrap=tk.WORD, state=tk.DISABLED, font=("Consolas", 9))
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text['yscrollcommand'] = log_scrollbar.set; log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y); self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        notebook.add(log_frame, text="运行日志")
        
        result_frame = ttk.Frame(notebook, padding="5")
        self.tree = ttk.Treeview(result_frame, show='headings')
        result_frame.columnconfigure(0, weight=1); result_frame.rowconfigure(0, weight=1); self.tree.grid(row=0, column=0, sticky='nsew')
        vsb = ttk.Scrollbar(result_frame, orient="vertical", command=self.tree.yview); hsb = ttk.Scrollbar(result_frame, orient="horizontal", command=self.tree.xview)
        vsb.grid(row=0, column=1, sticky='ns'); hsb.grid(row=1, column=0, sticky='ew'); self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        notebook.add(result_frame, text="结果预览")
        
        self.analysis_frame = ttk.Frame(notebook, padding="5")
        notebook.add(self.analysis_frame, text="图表分析")
        if MATPLOTLIB_AVAILABLE:
            chart_control_frame = ttk.Frame(self.analysis_frame); chart_control_frame.pack(fill=tk.X, side=tk.TOP, pady=(0, 5))
            ttk.Label(chart_control_frame, text="选择层位查看动态:").pack(side=tk.LEFT)
            self.layer_combo = ttk.Combobox(chart_control_frame, state='readonly'); self.layer_combo.pack(side=tk.LEFT, padx=5)
            self.layer_combo.bind("<<ComboboxSelected>>", self._on_layer_select)
            self.dynamic_chart_frame = ttk.Frame(self.analysis_frame); self.dynamic_chart_frame.pack(fill=tk.BOTH, expand=True)
        else:
            ttk.Label(self.analysis_frame, text="图表功能不可用，请安装 matplotlib 库: pip install matplotlib", foreground="red").pack(pady=20)

        control_frame = ttk.Frame(main_frame); control_frame.pack(fill=tk.X, expand=False, pady=(5,0))
        self.progress_bar = ttk.Progressbar(control_frame, orient="horizontal", mode='determinate'); self.progress_bar.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=5)
        self.start_button = ttk.Button(control_frame, text="开始劈分", command=self.start_calculation_thread, width=15); self.start_button.pack(side=tk.LEFT, padx=5)
        
        status_bar = ttk.Label(self.root, textvariable=self.status_text, relief=tk.SUNKEN, anchor='w', padding=2); status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _select_path(self, key, is_directory=False):
        path = filedialog.askdirectory() if is_directory else filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
        if path: self.file_paths[key].set(path)

    def _sort_treeview_column(self, col, reverse):
        try: data = [(float(self.tree.set(k, col)), k) for k in self.tree.get_children('')]
        except ValueError: data = [(self.tree.set(k, col), k) for k in self.tree.get_children('')]
        data.sort(reverse=reverse)
        for index, (val, k) in enumerate(data): self.tree.move(k, '', index)
        self.tree.heading(col, command=lambda _col=col: self._sort_treeview_column(_col, not reverse))

    def _display_result_preview(self, df):
        for item in self.tree.get_children(): self.tree.delete(item)
        if df is None or df.empty: return
        self.tree["columns"] = list(df.columns)
        for col in df.columns:
            self.tree.heading(col, text=col, command=lambda _col=col: self._sort_treeview_column(_col, False))
            self.tree.column(col, width=100, anchor='center')
        for _, row in df.head(200).iterrows(): self.tree.insert("", "end", values=list(row))
        self.log_message(f"结果预览加载完成，共显示 {min(len(df), 200)} 条记录。")

    def _update_analysis_chart(self, df):
        if not MATPLOTLIB_AVAILABLE or df is None or df.empty: return
        if self.chart_canvas: self.chart_canvas.get_tk_widget().destroy()
        
        try:
            analysis_data = df.groupby('层位')['劈分月产油量'].sum().sort_values(ascending=True).tail(20) # 只显示贡献最大的20个层
            
            fig = Figure(figsize=(8, 6), dpi=100)
            ax = fig.add_subplot(111)
            analysis_data.plot(kind='barh', ax=ax, title="分层累计产油量贡献图 (Top 20)")
            ax.set_xlabel("劈分累计产油量")
            ax.set_ylabel("层位")
            fig.tight_layout()

            self.chart_canvas = FigureCanvasTkAgg(fig, master=self.analysis_frame)
            self.chart_canvas.draw()
            self.chart_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self.log_message("产量贡献图表生成完毕。")
        except Exception as e:
            self.log_message(f"生成图表时出错: {e}", "error")

    def _display_welcome_message(self):
        """在日志区域显示欢迎信息和数据要求。"""
        welcome_message = (
            "欢迎使用产量劈分软件 V2.0\n\n"
            "请按以下说明准备数据文件，并选择对应路径：\n"
            "----------------------------------------------------\n"
            "1. 测井曲线: 文件夹 (内含.csv文件)\n"
            "   - 关键列: DEPTH, PERM (孔隙体积法还需: POR)\n\n"
            "2. 地质分层: Excel 文件 (.xlsx/.xls)\n"
            "   - 关键列: Well name, Top, Bottom, Surface\n\n"
            "3. 解释结论: Excel 文件 (.xlsx/.xls)\n"
            "   - 关键列: Well name, Top, Bottom\n\n"
            "4. 单井月报: Excel 文件 (.xlsx/.xls)\n"
            "   - 关键列: 井号, 生产年月, 生产井段, 月产油量等\n\n"
            "5. 射孔数据: Excel 文件 (.xlsx/.xls)\n"
            "   - 关键列: 井号, 射孔段顶深, 射孔段底深, 射孔日期\n\n"
            "----------------------------------------------------\n"
            "提示：所有关键列的名称均可在“高级设置”->“列名映射”中自定义。\n"
        )
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.insert(tk.END, welcome_message)
        self.log_text.config(state=tk.DISABLED)

    def _load_config(self):
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r') as f: config = json.load(f)
                for key, value in config.items():
                    if key in self.file_paths: self.file_paths[key].set(value)
                    elif key == 'perm_limit': self.perm_limit.set(value)
                    elif key == 'split_method': self.split_method.set(value)
                    elif key == 'column_mappings': self.column_mappings.update(value)
                self.log_message("已成功加载上次的配置。")
        except Exception as e: self.log_message(f"加载配置文件失败: {e}", "warning")

    def _save_config(self):
        config = {key: var.get() for key, var in self.file_paths.items()}
        config['perm_limit'] = self.perm_limit.get()
        config['split_method'] = self.split_method.get()
        config['column_mappings'] = self.column_mappings
        try:
            with open(CONFIG_FILE, 'w') as f: json.dump(config, f, indent=4)
        except Exception as e: messagebox.showwarning("保存配置失败", f"无法写入配置文件: {e}")

    def _on_closing(self): self._save_config(); self.root.destroy()
    
    def _reset_all(self):
        self.split_method.set('渗透率厚度积 (Kh)')
        for var in self.file_paths.values(): var.set("")
        self.perm_limit.set("0.1")
        self.column_mappings = DEFAULT_COLUMN_MAPPINGS.copy()
        self.log_text.config(state=tk.NORMAL); self.log_text.delete(1.0, tk.END); self.log_text.config(state=tk.DISABLED)
        for item in self.tree.get_children(): self.tree.delete(item)
        if self.dynamic_chart_canvas: self.dynamic_chart_canvas.get_tk_widget().destroy()
        if hasattr(self, 'layer_combo'): self.layer_combo.set(''); self.layer_combo['values'] = []
        self.log_message("所有设置已重置。")
        self._display_welcome_message()

    def _open_result_folder(self):
        if self.last_output_dir and os.path.isdir(self.last_output_dir): os.startfile(self.last_output_dir)

    def _show_about(self):
        messagebox.showinfo("关于", "产量劈分软件 V2.0\n\n作者: Kang\n\n一个用于产量劈分的软件。")
    
    def _open_column_mapping_window(self):
        ColumnMappingWindow(self.root, self.column_mappings, self.log_message)

    def log_message(self, msg, level="info"):
        self.log_text.config(state=tk.NORMAL)
        timestamp = time.strftime('%H:%M:%S'); self.log_text.insert(tk.END, f"[{timestamp}] [{level.upper():^7}] {msg}\n")
        self.log_text.see(tk.END); self.log_text.config(state=tk.DISABLED); self.root.update_idletasks()

    def _update_progress(self, value): self.progress_bar['value'] = value
    def _update_status(self, text): self.status_text.set(text); self.root.update_idletasks()
    def _toggle_controls(self, enable=True):
        state = 'normal' if enable else 'disabled'
        self.start_button.config(state=state)
        for menu in [self.file_menu, self.root.nametowidget(self.root.cget('menu')).nametowidget('高级设置')]:
             for i in range(menu.index('end') + 1):
                if menu.type(i) == 'command': menu.entryconfig(i, state=state)

    def start_calculation_thread(self):
        for key, var in self.file_paths.items():
            if not var.get(): messagebox.showerror("输入错误", f"请先选择 {key} 的路径！"); return
        try: float(self.perm_limit.get())
        except ValueError: messagebox.showerror("参数错误", "渗透率下限必须为数值！"); return
        calc_thread = threading.Thread(target=self._run_calculation, daemon=True); calc_thread.start()

    def _run_calculation(self):
        self._toggle_controls(enable=False); self.file_menu.entryconfig("打开结果目录", state='disabled'); self._clear_log()
        self.progress_bar['value'] = 0; self.log_message("="*15 + " 开始执行产量劈分 " + "="*15); start_time = time.time()
        
        self._update_status("正在加载数据...")
        dm = DataManager(self.log_message, self.column_mappings)
        paths = {k: v.get() for k, v in self.file_paths.items()}
        success, message = dm.load_data(paths)
        if not success: messagebox.showerror("数据加载失败", message); self._toggle_controls(enable=True); self._update_status("就绪"); return
        
        self._update_status("正在进行数据预检查...")
        dm.pre_check_data(self.split_method.get())
        if not messagebox.askyesno("预检查完成", "数据预检查已完成（详见日志），是否继续执行计算？"):
             self._toggle_controls(enable=True); self._update_status("已取消"); return

        engine = CalculationEngine(dm, self.log_message, self._update_progress, self._update_status, self.column_mappings, self.split_method.get())
        self.last_output_dir = os.path.dirname(paths['production'])
        self.result_df = engine.run_splitting(float(self.perm_limit.get()), self.last_output_dir)
        self.progress_bar['value'] = 100

        self._display_result_preview(self.result_df)
        self._populate_layer_combobox(self.result_df)
        end_time = time.time(); final_message = f"全部任务完成，总耗时: {end_time - start_time:.2f} 秒。"
        self.log_message(final_message, "info"); self._update_status(final_message)
        messagebox.showinfo("任务完成", "产量劈分计算已成功完成！")
        self.file_menu.entryconfig("打开结果目录", state='normal'); self._toggle_controls(enable=True)
    
    def _clear_log(self):
        self.log_text.config(state=tk.NORMAL); self.log_text.delete(1.0, tk.END); self.log_text.config(state=tk.DISABLED)

    def _on_layer_select(self, event=None):
        selected_layer = self.layer_combo.get()
        if selected_layer and self.result_df is not None:
            self._update_dynamic_chart(selected_layer)
            
    def _populate_layer_combobox(self, df):
        if df is None or df.empty: return
        layers = sorted(df['层位'].unique())
        self.layer_combo['values'] = layers
        if layers:
            self.layer_combo.set(layers[0])
            self._update_dynamic_chart(layers[0])
    
    def _update_dynamic_chart(self, layer_name):
        if not MATPLOTLIB_AVAILABLE: return
        if self.dynamic_chart_canvas: self.dynamic_chart_canvas.get_tk_widget().destroy()

        layer_df = self.result_df[self.result_df['层位'] == layer_name].copy()
        layer_df['年月'] = pd.to_datetime(layer_df['生产年月'].astype(str), format='%Y%m')
        
        dynamic_data = layer_df.groupby('年月').agg({
            '劈分月产油量': 'sum',
            '劈分月产水量': 'sum',
            '劈分月产气量': 'sum'
        }).sort_index()

        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(dynamic_data.index, dynamic_data['劈分月产油量'], label='月产油', marker='o')
        ax.plot(dynamic_data.index, dynamic_data['劈分月产水量'], label='月产水', marker='s')
        # ax.plot(dynamic_data.index, dynamic_data['劈分月产气量'], label='月产气', marker='^')
        ax.set_title(f"层位 [{layer_name}] 生产动态曲线")
        ax.set_xlabel("日期"); ax.set_ylabel("产量")
        ax.legend(); ax.grid(True)
        fig.tight_layout()

        self.dynamic_chart_canvas = FigureCanvasTkAgg(fig, master=self.dynamic_chart_frame)
        self.dynamic_chart_canvas.draw(); self.dynamic_chart_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

class ColumnMappingWindow(Toplevel):
    """列名映射设置窗口"""
    def __init__(self, parent, mappings, logger):
        super().__init__(parent)
        self.title("列名映射设置")
        self.transient(parent)
        self.grab_set()

        self.mappings = mappings
        self.logger = logger
        self.entries = {}
        
        frame = ttk.Frame(self, padding="10")
        frame.pack(expand=True, fill="both")
        
        ttk.Label(frame, text="请根据您的Excel文件，填写对应的列名。", foreground="blue").grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        row = 1
        for key, value in self.mappings.items():
            ttk.Label(frame, text=f"{key}:").grid(row=row, column=0, sticky='w', padx=5, pady=2)
            entry = ttk.Entry(frame, width=40)
            entry.insert(0, value)
            entry.grid(row=row, column=1, sticky='we', padx=5, pady=2)
            self.entries[key] = entry
            row += 1
            
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=row, column=0, columnspan=2, pady=10)
        ttk.Button(btn_frame, text="保存", command=self._save).pack(side="left", padx=10)
        ttk.Button(btn_frame, text="恢复默认", command=self._restore_defaults).pack(side="left", padx=10)
        ttk.Button(btn_frame, text="取消", command=self.destroy).pack(side="left", padx=10)
        
    def _save(self):
        for key, entry in self.entries.items():
            self.mappings[key] = entry.get()
        self.logger("列名映射已更新并保存。")
        self.destroy()
        
    def _restore_defaults(self):
        if messagebox.askyesno("确认", "确定要恢复所有列名为默认设置吗？"):
            for key, entry in self.entries.items():
                entry.delete(0, tk.END)
                entry.insert(0, DEFAULT_COLUMN_MAPPINGS[key])


if __name__ == "__main__":
    root = tk.Tk()
    app = ProductionSplittingApp(root)
    root.mainloop()
