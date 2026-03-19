# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 09:30:06 2023
@author: KANG
"""
import pandas as pd
from os import listdir
import os
import lasio
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from datetime import datetime
from astropy.convolution import convolve, Gaussian1DKernel
import math
import shutil

from chong32_formulas import calculate_permeability_from_porosity_percent

# --- Matplotlib全局配置 ---
mpl.rcParams['font.sans-serif'] = ['SimSun']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
mpl.rcParams['axes.facecolor'] = 'w'
mpl.rcParams['axes.edgecolor'] = 'k'

class WellLogProcessor:
    """
    测井数据处理与油水解释工具类。
    """
    def __init__(self):
        """
        初始化处理器。
        """
        pass

    @staticmethod
    def _gauss_filter(data_1d, pix):
        """
        对一维数据应用高斯滤波器，进行平滑处理。
        :param data_1d: (np.array) 一维数据数组。
        :param pix: (int) 高斯核的标准差，控制平滑程度。
        :return: (np.array) 平滑后的数据。
        """
        gauss_kernel = Gaussian1DKernel(pix)
        smoothed_data_gauss = convolve(data_1d, gauss_kernel)
        return smoothed_data_gauss

    @staticmethod
    def _get_all_dir_re(path):
        """
        递归获取指定路径下的所有子文件夹列表。
        :param path: (str) 起始文件夹路径。
        :return: (list) 所有子文件夹的绝对路径列表。
        """
        folder_list = []
        files_list = os.listdir(path)
        for file_name in files_list:
            abs_path = os.path.join(path, file_name)
            if os.path.isdir(abs_path):
                folder_list.append(abs_path)
                folder_list.extend(WellLogProcessor._get_all_dir_re(abs_path))
        return folder_list

    @staticmethod
    def _get_block_index(litho_data):
        """
        根据岩性数据获取分段索引。当岩性发生变化时，记录一个新段的起止位置。
        :param litho_data: (list or np.array) 岩性标识序列。
        :return: (dict) 包含各分段起止索引的字典。 e.g., {'1': [0, 50], '2': [51, 100]}
        """
        index_dict = {}
        segment_num = 0
        start_index = 0
        for i in range(len(litho_data) - 1):
            if litho_data[i] != litho_data[i + 1]:
                segment_num += 1
                index_dict[str(segment_num)] = [start_index, i]
                start_index = i + 1
        # 处理最后一段
        segment_num += 1
        index_dict[str(segment_num)] = [start_index, len(litho_data) - 1]
        return index_dict

    @staticmethod
    def _lian_xu_data_check(property_data):
        """
        检查数据中的连续块，并返回长度大于10的块的索引。
        :param property_data: (list or np.array) 待检查的属性数据。
        :return: (list) 包含连续数据块起始和结束索引的列表。
        """
        index_block = []
        count = 0
        for i in range(len(property_data) - 1):
            if property_data[i] != property_data[i + 1]:
                if i - count > 10:
                    index_block.append([count, i])
                count = i + 1
        # 检查最后一段
        if len(property_data) - 1 - count > 10:
            index_block.append([count, len(property_data) - 1])
        return index_block
    
    @staticmethod
    def _perm_caculation(por):
        """
        根据孔隙度计算渗透率（经验公式）。
        :param por: (list or np.array) 孔隙度数据（单位：%）。
        :return: (list) 计算得到的渗透率数据。
        """
        perm_series = pd.Series(por)
        return calculate_permeability_from_porosity_percent(perm_series, -999.25).tolist()

    @staticmethod
    def _fill_missing_values(df, missing_value=-999.25, text_missing=""):
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        text_cols = [col for col in df.columns if col not in numeric_cols]

        if numeric_cols:
            df.loc[:, numeric_cols] = df[numeric_cols].fillna(missing_value)
        if text_cols:
            df.loc[:, text_cols] = df[text_cols].fillna(text_missing)
        return df

    def tqjt1(self, wellhead_file, analysis_file, output_dir):
        """
        提取另一个表格数据合并到一个文件。
        该函数从一个总的井头文件和一个油品分析文件中提取数据，
        根据井号匹配井的坐标信息，并将结果按不同列拆分保存为txt文件。
        """
        xbj_data = pd.read_excel(wellhead_file)
        kfqjt_data = pd.read_excel(analysis_file)

        xbj_data["井号"] = xbj_data["井号"].astype(str)
        kfqjt_data["井号"] = kfqjt_data["井号"].astype(str)
        welld = xbj_data["井号"].values

        columns = kfqjt_data.columns[1:]
        for col in columns:
            data = kfqjt_data[["井号", col]]
            well = data["井号"].values

            res = pd.DataFrame([], columns=data.columns)
            xx = []
            yy = []
            surface = []
            type_list = []
            for w in well:
                tt = 0
                for ww in welld:
                    if w == ww:
                        tt = ww
                        break
                if tt == 0:
                    for ww in welld:
                        if w in ww:
                            tt = ww
                            break
                wd = xbj_data[xbj_data["井号"] == tt]
                if not len(wd):
                    continue
                res = pd.concat([res, data[data["井号"] == w]])
                x_val = wd["Surface X"].values[0]
                y_val = wd["Surface Y"].values[0]
                xx.append(x_val)
                yy.append(y_val)
                surface.append(col)
                type_list.append("Horizon")
            res["X"] = xx
            res["Y"] = yy
            res["Surface"] = surface
            res["Type"] = type_list
            output_path = os.path.join(output_dir, f"{col}.txt")
            res.to_csv(output_path, sep=" ", index=False)
            print(f"已生成文件: {output_path}")

    def tqjt(self, stats_file, well_list_file, output_dir):
        """
        提取井头数据。
        比较两个Excel文件中的井号列表，将在第一个文件中存在但在第二个文件中也存在的井的信息提取出来，
        同时列出在第一个文件中存在但在第二个文件中缺失的井号。
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        xbj_data = pd.read_excel(stats_file)
        kfqjt_data = pd.read_excel(well_list_file)

        no_well_list = []
        res = pd.DataFrame([], columns=xbj_data.columns)

        xbj_data["Well"] = xbj_data["Well"].astype("str")
        xbj_well = sorted(set(xbj_data["Well"]))

        kfqjt_data["Well"] = kfqjt_data["Well"].astype("str")
        kfqjt_well = sorted(set(kfqjt_data["Well"]))

        for well in xbj_well:
            if well in kfqjt_well:
                print(f"找到匹配井: {well}")
                mm = xbj_data.loc[xbj_data["Well"] == well, :]
                res = pd.concat([res, mm])
            else:
                no_well_list.append(well)

        output_path_found = os.path.join(output_dir, "有问题的井曲线.xlsx")
        res.to_excel(output_path_found, index=False)
        print(f"有问题的井曲线信息已保存至: {output_path_found}")

        ress = pd.DataFrame([])
        ress["Well"] = no_well_list
        output_path_not_found = os.path.join(output_dir, "未找到的井号.xlsx")
        ress.to_excel(output_path_not_found, index=False)
        print(f"未找到的井号列表已保存至: {output_path_not_found}")
        print("未找到的井号:", no_well_list)

    def well_files(self, stats_file, folder_path):
        """
        按井名查找文件是否存在。
        根据一个Excel文件中的井号列表，检查在指定的文件夹中是否存在对应的csv文件。
        """
        xbj_data = pd.read_excel(stats_file)
        filelist = listdir(folder_path)

        no_well_list = []

        xbj_data["Well"] = xbj_data["Well"].astype("str")
        xbj_well = sorted(set(xbj_data["Well"]))

        for well in xbj_well:
            if well + ".csv" not in filelist:
                no_well_list.append(well)

        print(f"缺失文件对应的井号数量: {len(no_well_list)}")
        print("缺失的井号:", no_well_list)

    def jf_qx_zhengli(self, input_dir, output_dir):
        """
        多个重名文件转为las后移动到不同的文件夹中。
        该函数读取一个文件夹中的所有csv文件，将其转换为LAS 2.0格式，
        并处理重名文件（通过创建新的子文件夹来存放）。
        """
        filename = listdir(input_dir)
        
        base_folder_path = os.path.join(output_dir, "1")
        if not os.path.exists(base_folder_path):
            os.makedirs(base_folder_path)

        for file in filename:
            name = file.split(".las")[0]
            
            # 读取并处理数据
            data = pd.read_csv(os.path.join(input_dir, file), encoding='gb2312')
            data[data <= -300] = -999.25

            # 创建LAS对象
            las = lasio.LASFile()
            las.well.DATE = str(datetime.today().date())
            las.well.WELL = name
            las.well.UWI = name
            las.well.CTRY = 'CHINA'
            las.well.NULL = -999.25
            for cure in data.columns:
                las.add_curve(cure, data[cure])

            # 处理重名文件
            i = 1
            current_folder_path = os.path.join(output_dir, str(i))
            while os.path.exists(os.path.join(current_folder_path, name + ".las")):
                i += 1
                current_folder_path = os.path.join(output_dir, str(i))
                if not os.path.exists(current_folder_path):
                    os.makedirs(current_folder_path)
            
            new_path = os.path.join(current_folder_path, name + ".las")
            print(f"正在写入: {new_path}")
            las.write(new_path, fmt='%.3f', version=2)

    def jf_qx_shaixuan(self, input_dir, output_dir):
        """
        筛选同名井的测井曲线。
        对于同一个井（通过文件名判断），若存在多个测井文件，此函数会比较这些文件中的曲线。
        如果曲线同名且数值完全相同，则认为是重复曲线，在后续文件中将其剔除，只保留独有的曲线。
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        filename = listdir(input_dir)
        well_path = {}

        for file in filename:
            wellname = file.split(" ")[0]
            if wellname not in well_path:
                well_path[wellname] = []
            well_path[wellname].append(file)

        for well in well_path:
            print(f"正在处理井: {well}")
            if len(well_path[well]) > 1:
                well_data = {}
                for i in range(len(well_path[well])):
                    data_path = os.path.join(input_dir, well_path[well][i])
                    data = pd.read_csv(data_path, encoding="gb2312")
                    
                    if i > 0:  # 从第二个文件开始，与之前的所有文件比较
                        unique_logs = []
                        for log in data.columns:
                            if log.upper() == "DEPTH":
                                continue
                            is_duplicate = False
                            for j in range(i):
                                prev_data = well_data[str(j)]
                                if log in prev_data.columns:
                                    is_duplicate = True
                                    # 如果数据完全一致，则认为是重复的，不保留
                                    if len(data[log]) == len(prev_data[log]) and all(data[log] == prev_data[log]):
                                        pass # 重复，跳过
                                    else:
                                        unique_logs.append(log) # 同名但数据不同，保留
                                    break # 已找到同名曲线，跳出内层循环
                            if not is_duplicate:
                                unique_logs.append(log) # 没有同名曲线，保留
                        
                        # 只保留深度曲线和独特的曲线
                        data = data[[data.columns[0]] + unique_logs]

                    well_data[str(i)] = data
                    output_file_path = os.path.join(output_dir, well_path[well][i])
                    data.to_csv(output_file_path, index=False, encoding="gb2312")

            else:  # 只有一个文件，直接复制
                shutil.copy(os.path.join(input_dir, well_path[well][0]), 
                            os.path.join(output_dir, well_path[well][0]))

    def qu_xian_chong_gou(self, input_dir, output_dir):
        """
        重构含油敏感曲线。
        通过公式 RT_Oil = log10(RT / (GR * AC)) 计算新的含油敏感曲线。
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filelist = listdir(input_dir)
        
        base_logs = ["GR_one", "AC_one"]
        rt_logs = ["RT_one", "RLLD_one", "LLD_one"]
        
        for file in filelist:
            well_name = file.split(".csv")[0]
            print(f"正在处理井: {well_name}")
            
            data = pd.read_csv(os.path.join(input_dir, file), encoding="gb2312")
            
            # 检查所需曲线是否存在
            if not all(log in data.columns for log in base_logs):
                print(f"跳过 {well_name}: 缺少 GR_one 或 AC_one 曲线。")
                continue
            
            rt_log_found = next((log for log in rt_logs if log in data.columns), None)
            if not rt_log_found:
                print(f"跳过 {well_name}: 缺少电阻率曲线。")
                continue
            
            # 计算含油敏感曲线
            # 避免除以零的错误
            denominator = data["GR_one"] * data["AC_one"]
            data['RT_Oil'] = -999.25 # 默认为无效值
            valid_mask = denominator != 0
            
            data.loc[valid_mask, 'RT_Oil'] = data.loc[valid_mask, rt_log_found] / denominator[valid_mask]
            
            # 处理计算后的无效值和无穷大值
            data.loc[data['RT_Oil'] < 0.1, 'RT_Oil'] = -999.25
            data.loc[data['RT_Oil'] == np.inf, 'RT_Oil'] = 1000 # 设为上限
            
            # 取对数
            valid_rt_oil_mask = data['RT_Oil'] != -999.25
            data.loc[valid_rt_oil_mask, 'RT_Oil'] = np.log10(data.loc[valid_rt_oil_mask, 'RT_Oil'])
            data.loc[data['RT_Oil'] < 0, 'RT_Oil'] = -999.25 # 对数后小于0的也视为无效
            
            output_path = os.path.join(output_dir, file)
            data.to_csv(output_path, encoding="gb2312", index=False)
            print(f"已生成文件: {output_path}")

    def folder_files(self, root_dir, output_excel_path, file_type=".las"):
        """
        统计多层文件夹内的文件，并生成井名列表。
        """
        files = []
        folders = []
        
        folder_list = self._get_all_dir_re(root_dir)
        # 包括根目录
        if root_dir not in folder_list:
            folder_list.insert(0, root_dir)

        for folder in folder_list:
            folder_name = os.path.basename(folder)
            
            filelist = listdir(folder)
            for file in filelist:
                if file.lower().endswith(file_type.lower()):
                    well_name = file[:-len(file_type)]
                else:
                    # 尝试按"_"分割取第一部分作为井名
                    well_name = file.split("_")[0]
                
                files.append(well_name)
                folders.append(folder_name)
                
        red = pd.DataFrame({"Well": files, "类型": folders})
        red = red.drop_duplicates(subset=["Well"], keep='first')
        red.to_excel(output_excel_path, index=False)
        print(f"井名列表已保存至: {output_excel_path}")

    def dian_zu_yan_xing(self, input_dir, output_dir, param_curve="E07L", litho_threshold=20, generate_figure=True):
        """
        利用电阻率曲线按固定阈值划分岩性。
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filelist = listdir(input_dir)
        
        red = pd.DataFrame([], columns=["WellName", "DEPTH", "Litho"])

        for file in filelist:
            wellname = file.split(".csv")[0]
            print(f"正在处理井: {wellname}")
            
            data = pd.read_csv(os.path.join(input_dir, file), encoding="gb2312")
            depth_name = data.columns[0]
            
            if param_curve not in data.columns:
                print(f"跳过 {wellname}: 缺少 {param_curve} 曲线。")
                continue
            
            # 数据预处理
            data.loc[data[param_curve] < -300, param_curve] = -999.25
            data = data.loc[data[param_curve] != -999.25, :].copy()
            
            # 岩性解释
            data["Litho"] = "泥岩"
            data.loc[data[param_curve] > litho_threshold, "Litho"] = "砂岩"
            
            # 汇总结果
            data["WellName"] = wellname
            data["DEPTH"] = data[depth_name]
            red = pd.concat([red, data[["WellName", "DEPTH", "Litho"]]])
    
            if generate_figure:
                fig = plt.figure(figsize=(8, 80), dpi=80)
                ax1 = fig.add_subplot(1, 1, 1)
                ax1.plot(data[param_curve], data['DEPTH'], color='blue', label=param_curve, linewidth=3)
                
                ax1.axvline(x=litho_threshold, ymin=0, ymax=1, c='red', linewidth=4, linestyle="--")
                
                ax1.invert_yaxis()
                ax1.xaxis.set_ticks_position('top')
                ax1.xaxis.set_label_position('top')
                ax1.set_xlabel(param_curve, fontsize=28)
                ax1.set_ylabel('DEPTH(m)', fontsize=28)
                ax1.tick_params(labelsize=28)
                ax1.grid()
                plt.suptitle(f'{param_curve} 分段结果图', fontsize=28, font='SimSun')
                
                figure_path = os.path.join(output_dir, f"{wellname}_{param_curve}_分段结果图.jpg")
                plt.savefig(figure_path, bbox_inches='tight', dpi=200)
                plt.close()

        # 保存所有井的解释结果
        output_csv_path = os.path.join(output_dir, "All_Wells_Litho_Interpretation.csv")
        red.to_csv(output_csv_path, encoding="gb2312", index=False)
        print(f"所有井的岩性解释结果已保存至: {output_csv_path}")

    def rt_cal(self, input_dir, output_dir, well_list_file=None):
        """
        统一多种电阻率曲线名称为'RTCAL'。
        该函数会按预设的曲线名列表查找电阻率曲线，找到第一个后将其重命名为 RTCAL 并保存。
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 深电阻率曲线名列表 (优先)
        resistivity_deep = [
            'E07L','EL07','RD','LLD','M2RX','R85O','E02L','E03L','E16N','E10N','E13L',
            'EL13',"E06N","EL14","E14L",'E19L','EN20','E26L','E28L','LL9','LL7',"RILD","RILM"
        ]
        # 浅电阻率或其他曲线名列表
        resistivity_shallow = [
            'RS','LLS','RM','B01_A095_M','B01_A095_M1','B01_A095M','B01_A095M1','B07_A095_M',
            'ILM','ILM1','M2R2','M2R3','M2R6','M2R9','M2R1','R40O','R60O',"RTCALQ"
        ]

        well_filter = []
        if well_list_file:
            wells_df = pd.read_excel(well_list_file)
            wells_df["Well"] = wells_df["Well"].astype("str")
            well_filter = sorted(set(wells_df["Well"]))

        filelist = [f for f in listdir(input_dir) if f.lower().endswith('.csv')]
        error_wells = []
        
        process_bar = tqdm(filelist, desc="进度", leave=True, ascii=True, ncols=80)
        for file in process_bar:
            wellname = str(file.split('.csv')[0])
            process_bar.set_description(f'处理中: {wellname}')
            
            if well_filter and wellname not in well_filter:
                continue
                
            log_data = pd.read_csv(os.path.join(input_dir, file))
            
            found = False
            # 优先在深电阻率里找
            for rlog in resistivity_deep:
                if rlog in log_data.columns:
                    log_data["RTCAL"] = log_data[rlog]
                    found = True
                    break
            # 如果没找到，在浅电阻率里找
            if not found:
                for rlog in resistivity_shallow:
                    if rlog in log_data.columns:
                        log_data["RTCAL"] = log_data[rlog]
                        # 这里可以根据需要添加校正因子，目前为1.0
                        log_data.loc[log_data["RTCAL"] != -999.25, "RTCAL"] *= 1.0 
                        found = True
                        break
            
            if found:
                output_path = os.path.join(output_dir, file)
                log_data[[log_data.columns[0], "RTCAL"]].to_csv(output_path, encoding='gb2312', index=False)
            else:
                error_wells.append(wellname)
                
        print("\n处理完成。")
        if error_wells:
            print("以下井未找到任何指定的电阻率曲线:", error_wells)

    def por_combine(self, path_list, output_dir):
        """
        合并来自不同计算方法的孔隙度结果。
        以第一个路径下的文件为基础，用后续路径文件中同一口井的孔隙度数据（POR_PRE）填补基础文件中的缺失值。
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 获取所有不重复的文件名
        all_files = set()
        path_files = []
        for path in path_list:
            files = [f for f in listdir(path) if f.lower().endswith('.csv')]
            path_files.append(files)
            all_files.update(files)

        process_bar = tqdm(sorted(list(all_files)), desc="合并孔隙度", leave=True, ascii=True, ncols=80)
        for file in process_bar:
            wellname = str(file.split('.csv')[0])
            process_bar.set_description(f'处理中: {wellname}')

            # 找到包含该文件的第一个路径作为基础数据
            base_data = None
            start_index = -1
            for i, files in enumerate(path_files):
                if file in files:
                    base_data = pd.read_csv(os.path.join(path_list[i], file))
                    start_index = i
                    break
            
            if base_data is None:
                continue

            # 用后续路径的数据进行填充
            for i in range(start_index + 1, len(path_list)):
                if file in path_files[i]:
                    secondary_data = pd.read_csv(os.path.join(path_list[i], file))
                    if "POR_PRE" in secondary_data.columns:
                        # 确保对齐
                        if len(base_data) == len(secondary_data):
                            fill_mask = base_data["POR_PRE"] == -999.25
                            base_data.loc[fill_mask, "POR_PRE"] = secondary_data.loc[fill_mask, "POR_PRE"]
                        else:
                            print(f"警告：{wellname} 在不同路径下的文件深度点数不一致，跳过合并。")

            output_path = os.path.join(output_dir, file)
            base_data.to_csv(output_path, encoding='gb2312', index=False)
            
        print("\n孔隙度合并完成。")

    def vsh_combine(self, path_list, output_dir):
        """
        合并来自不同计算方法的泥质含量(VSH)结果。
        逻辑同 `por_combine`。
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        all_files = set()
        path_files = []
        for path in path_list:
            files = [f for f in listdir(path) if f.lower().endswith('.csv')]
            path_files.append(files)
            all_files.update(files)

        process_bar = tqdm(sorted(list(all_files)), desc="合并泥质含量", leave=True, ascii=True, ncols=80)
        for file in process_bar:
            wellname = str(file.split('.csv')[0])
            process_bar.set_description(f'处理中: {wellname}')

            base_data = None
            start_index = -1
            for i, files in enumerate(path_files):
                if file in files:
                    base_data = pd.read_csv(os.path.join(path_list[i], file))
                    start_index = i
                    break
            
            if base_data is None:
                continue

            for i in range(start_index + 1, len(path_list)):
                if file in path_files[i]:
                    secondary_data = pd.read_csv(os.path.join(path_list[i], file))
                    if "VSH" in secondary_data.columns and len(base_data) == len(secondary_data):
                        fill_mask = base_data["VSH"] == -999.25
                        base_data.loc[fill_mask, "VSH"] = secondary_data.loc[fill_mask, "VSH"]

            output_path = os.path.join(output_dir, file)
            base_data.to_csv(output_path, encoding='gb2312', index=False)

        print("\n泥质含量合并完成。")

    def por_vsh_perm_sw_combine_logs(self, calculated_param_dir, original_logs_dir, output_dir, param_to_merge="VSH"):
        """
        将计算出的参数（如VSH, POR等）与原始测井曲线文件合并。
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logs_files = listdir(original_logs_dir)
        cal_files = [f for f in listdir(calculated_param_dir) if f.lower().endswith('.csv')]
        
        error_log = []
        process_bar = tqdm(cal_files, desc=f"合并 {param_to_merge}", leave=True, ascii=True, ncols=80)
        for file in process_bar:
            wellname = str(file.split('.csv')[0])
            process_bar.set_description(f'处理中: {wellname}')
            
            cal_data_path = os.path.join(calculated_param_dir, file)
            cal_data = pd.read_csv(cal_data_path, encoding='gb2312')

            if file in logs_files:
                log_data_path = os.path.join(original_logs_dir, file)
                log_data = pd.read_csv(log_data_path, encoding='gb2312')
                
                # 深度对齐检查
                if len(log_data) == len(cal_data) and \
                   log_data.columns[0] == cal_data.columns[0] and \
                   log_data.iloc[0, 0] == cal_data.iloc[0, 0] and \
                   log_data.iloc[-1, 0] == cal_data.iloc[-1, 0]:
                    
                    if param_to_merge in cal_data.columns:
                        log_data[param_to_merge] = cal_data[param_to_merge]
                        output_path = os.path.join(output_dir, file)
                        log_data.to_csv(output_path, encoding='gb2312', index=False)
                    else:
                        error_log.append([wellname, f"计算结果文件中缺少'{param_to_merge}'列"])
                else:
                    error_log.append([wellname, "深度不匹配或点数不一致"])
            else:
                error_log.append([wellname, "在原始测井文件夹中未找到对应文件"])
        
        print("\n合并完成。")
        if error_log:
            print("出现以下错误:")
            for error in error_log:
                print(f"  - 井: {error[0]}, 原因: {error[1]}")

    def cal_aa(self, input_dir, output_dir, step=10, buffer=20):
        """
        为每口井计算一条全局的、等间距的'AA'曲线，通常用于标记或作为全局参考。
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logs_files = [f for f in listdir(input_dir) if f.lower().endswith('.csv')]
        
        process_bar = tqdm(logs_files, desc="计算AA曲线", leave=True, ascii=True, ncols=80)
        for file in process_bar:
            well = str(file.split('.csv')[0])
            process_bar.set_description(f'处理中: {well}')
            
            data = pd.read_csv(os.path.join(input_dir, file), encoding='gb2312')
            if data.empty:
                continue

            start_depth = data.iloc[0, 0]
            end_depth = data.iloc[-1, 0]
            
            # 创建新的深度轴
            start_md = int(start_depth - buffer)
            end_md = int(end_depth + buffer)
            num_points = int((end_md - start_md) / step)
            
            sample_depths = np.linspace(start_md, start_md + num_points * step, num=num_points, endpoint=False)
            sample_depths = [round(i, 3) for i in sample_depths]
            
            res = pd.DataFrame(sample_depths, columns=["DEPT"])
            res["AA"] = 0 # AA曲线值设为0
            
            output_path = os.path.join(output_dir, f"{well}.csv")
            res.to_csv(output_path, encoding='gb2312', index=False)
        print("\nAA曲线计算完成。")

    def jsjl_sand(self, input_dir, output_dir, sand_log="Sand36", rt_log="RTCAL", vsh_log="VSH", por_log="POR_PRE", sw_log="SW_PRE", target_log="Oil33", percent_for_mean=0.5):
        """
        根据砂体段的测井响应，综合判断解释结论（油层、水层、干层等）。
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filelist = [f for f in listdir(input_dir) if f.lower().endswith('.csv')]
        
        process_bar = tqdm(filelist, desc="砂体解释结论", leave=True, ascii=True, ncols=80)
        for file in process_bar:
            wellname = str(file.split('.csv')[0])
            process_bar.set_description(f'处理中: {wellname}')
            
            if os.path.exists(os.path.join(output_dir, file)):
                continue
            
            log_data = pd.read_csv(os.path.join(input_dir, file), encoding='gb2312')
            log_data = self._fill_missing_values(log_data)
            dept_col = log_data.columns[0]
            
            if sand_log not in log_data.columns or rt_log not in log_data.columns:
                continue

            if "OIL_PRE" in log_data.columns:
                res = log_data.copy()
                res[target_log] = -999.25
                valid_mask = res["OIL_PRE"] != -999.25
                res.loc[valid_mask, target_log] = 4
                res.loc[valid_mask & (res["OIL_PRE"] == 1), target_log] = 1
                if "JC_PRE" in res.columns:
                    res.loc[valid_mask & (res["JC_PRE"] == 1), target_log] = 4

                output_csv_path = os.path.join(output_dir, file)
                res.to_csv(output_csv_path, encoding='gb2312', index=False)

                las_path = os.path.join(output_dir, f"{wellname}.las")
                las = lasio.LASFile()
                las.well.WELL = wellname
                las.well.NULL = -999.25
                las.add_curve(dept_col, res[dept_col].values)
                for curve in [target_log, "POR_PRE", "PERM_PRE", "SW_PRE", "SO_PRE", "OIL_PRE", "JC_PRE"]:
                    if curve in res.columns:
                        las.add_curve(curve, res[curve].values)
                las.write(las_path, fmt='%.3f', version=2)
                continue

            # 砂体分段
            litho = log_data[sand_log].values
            index_dict = self._get_block_index(litho)
            
            res = log_data.copy()
            res[target_log] = log_data[sand_log].values
            
            # 初始化孔渗饱曲线
            res["POR_PRE"] = res[por_log] if por_log in res.columns else -999.25
            if "PERM_PRE" not in res.columns and "PERM_PRE1" in res.columns:
                res["PERM_PRE"] = res["PERM_PRE1"]
            if sw_log in res.columns:
                res["SW_PRE"] = res[sw_log]

            for key in index_dict:
                index = index_dict[key]
                litho_block = log_data.loc[index[0]:index[1], sand_log]
                
                # 只处理砂体段 (假设非0值为砂体)
                if litho_block.mode().iloc[0] == 0 or litho_block.mode().iloc[0] == -999.25:
                    continue
                
                interpretation_code = 0
                # 根据岩性代码确定解释结论代码的基准
                if litho_block.mode().iloc[0] == 2: # 假设2是某种特殊层
                    interpretation_code = 4 # 干层
                else:
                    rt_block = log_data.loc[index[0]:index[1], rt_log]
                    vsh_block = log_data.loc[index[0]:index[1], vsh_log]
                    por_block = log_data.loc[index[0]:index[1], por_log]
                    
                    # 使用部分数据点计算均值，减少极值影响
                    points_count = len(rt_block) if len(rt_block) < 20 else int(len(rt_block) * percent_for_mean)
                    rt_mean = np.mean(sorted(rt_block.values, reverse=True)[:points_count + 1])
                    vsh_mean = np.mean(sorted(vsh_block.values, reverse=False)[:points_count + 1])
                    
                    # 主要解释逻辑
                    if vsh_mean <= 0.43:
                        if rt_mean >= 17: interpretation_code = 1 # 油层
                        elif 14 <= rt_mean < 17: interpretation_code = 2 # 油水同层
                        elif 11 <= rt_mean < 14: interpretation_code = 3 # 含油水层
                        else: interpretation_code = 5 # 水层
                    else:
                        interpretation_code = 4 # 泥质含量高，判为干层
                    
                    # SP曲线辅助判断
                    if rt_mean < 11 and vsh_mean <= 0.43 and "SP_b_one" in log_data.columns:
                        sp_block = log_data.loc[index[0]:index[1], "SP_b_one"].values
                        sp_block_valid = sp_block[sp_block != -999.25]
                        if len(sp_block_valid) > 0:
                            sp_points_count = len(sp_block_valid) if len(sp_block_valid) < 20 else int(len(sp_block_valid) * percent_for_mean)
                            sp_mean = np.mean(sorted(sp_block_valid, reverse=False)[:sp_points_count+1])
                            if sp_mean > 0.8: interpretation_code = 4
                        elif vsh_mean > 0.25:
                            interpretation_code = 4

                    # 孔隙度和饱和度修正
                    if por_block.mean() < 0.16: interpretation_code = 4 # 低孔干层
                    if sw_log in log_data.columns:
                        sw_block = log_data.loc[index[0]:index[1], sw_log]
                        if sw_block.mean() > 0.7 and interpretation_code != 4:
                            interpretation_code = 5 # 高饱和度水层

                res.loc[index[0]:index[1], target_log] = interpretation_code

                # 处理干层的孔隙度
                if interpretation_code == 4 and por_block.mean() > 0.16:
                    res.loc[index[0]:index[1], "POR_PRE"] = por_block.values * 0.16 / por_block.mean()
            
            # 更新干层和水层的饱和度
            if "SW_PRE" in res.columns:
                res.loc[res[target_log] == 4, "SW_PRE"] = 1 # 干层饱和度为1
                res.loc[res[target_log] == 5, "SW_PRE"] = 1 # 水层饱和度为1
            
            # 根据更新后的孔隙度重新计算渗透率
            res.loc[(res['POR_PRE'] < 0) & (res['POR_PRE'] != -999.25), "POR_PRE"] = 0
            por_for_perm = res["POR_PRE"].values * 100
            perm_values = self._perm_caculation(por_for_perm)
            if len(perm_values) > 0:
                res['PERM_PRE'] = perm_values
                res.loc[(res['PERM_PRE'] < 0.01) & (res['PERM_PRE'] != -999.25), "PERM_PRE"] = 0.01

            # 泥岩段的物性
            res.loc[res[sand_log] == 0, "POR_PRE"] = 0
            res.loc[res[sand_log] == 0, "PERM_PRE"] = 0.01
            if "SW_PRE" in res.columns:
                res.loc[res[sand_log] == 0, "SW_PRE"] = 1

            # 输出CSV
            output_csv_path = os.path.join(output_dir, file)
            res.to_csv(output_csv_path, encoding='gb2312', index=False)
            
            # 输出LAS
            las_path = os.path.join(output_dir, f"{wellname}.las")
            las = lasio.LASFile()
            las.well.WELL = wellname
            las.well.NULL = -999.25
            las.add_curve(dept_col, res[dept_col].values)
            for curve in [target_log, "POR_PRE", "PERM_PRE", "SW_PRE"]:
                if curve in res.columns:
                    las.add_curve(curve, res[curve].values)
            las.write(las_path, fmt='%.3f', version=2)

    def del_jsjl(self, interpretation_file, tops_file, output_excel_path):
        """
        根据分层数据，删除非目的层的解释结论。
        """
        tops_data = pd.read_excel(tops_file)
        tops_data["Well"] = tops_data["Well"].astype("str")
        
        interp_data = pd.read_excel(interpretation_file)
        interp_data["Well"] = interp_data["Well"].astype("str")
        wells = sorted(set(interp_data["Well"]))
        
        final_res = pd.DataFrame([], columns=interp_data.columns)
        
        process_bar = tqdm(wells, desc="删除非目的层解释", leave=True, ascii=True, ncols=80)
        for wellname in process_bar:
            process_bar.set_description(f'处理中: {wellname}')
            well_tops = tops_data.loc[tops_data["Well"] == wellname, :]
            
            if not well_tops.empty:
                top_depth = well_tops["MD"].min()
                bottom_depth = well_tops["MD"].max()
            else:
                # 如果没有分层数据，则保留该井所有解释
                top_depth, bottom_depth = -99999, 99999
            
            well_data = interp_data.loc[interp_data["Well"] == wellname, :]
            # 保留顶底深在目的层段内的解释结论
            well_data_filtered = well_data.loc[(well_data["底深"] >= top_depth) & (well_data["顶深"] <= bottom_depth)]
            
            final_res = pd.concat([final_res, well_data_filtered])
            
        final_res.to_excel(output_excel_path, index=False)
        print(f"处理完成，结果已保存至: {output_excel_path}")

    def logs_chongfutiaoxuan(self, input_dir, output_dir):
        """
        合并同一口井的多个测井文件。
        当一口井被拆分成多个文件存储时（如不同测段），此函数将它们合并成一个文件。
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        files = listdir(input_dir)
        well_map = {}
        for file in files:
            # 提取井名的逻辑，可能需要根据实际情况调整
            parts = file.split(".csv")[0].split(" ")
            if "K" in parts[0] or "V" in parts[0] and len(parts) > 1:
                wellname = f"{parts[0]} {parts[1]}"
            else:
                wellname = parts[0]
            
            if wellname not in well_map:
                well_map[wellname] = []
            well_map[wellname].append(file)
            
        process_bar = tqdm(well_map.keys(), desc="合并同名测井文件", leave=True, ascii=True, ncols=80)
        for well in process_bar:
            process_bar.set_description(f'处理中: {well}')
            
            well_files = well_map[well]
            well_data_dict = {}
            top_list, bot_list, step_list = [], [], []

            for i, file in enumerate(well_files):
                data = pd.read_csv(os.path.join(input_dir, file), encoding="gb2312")
                if data.empty: continue
                
                well_data_dict[str(i)] = data
                depth_col = data.columns[0]
                top = data[depth_col].iloc[0]
                bot = data[depth_col].iloc[-1]
                top_list.append(top)
                bot_list.append(bot)
                if len(data) > 1:
                    step = round((bot - top) / (len(data) - 1), 3)
                    step_list.append(step)

            if not top_list: continue

            # 创建新的统一深度轴
            overall_top = min(top_list)
            overall_bot = max(bot_list)
            # 优先使用最常见的采样间隔
            step = pd.Series(step_list).mode().iloc[0] if step_list else 0.1 

            num_points = int((overall_bot - overall_top) / step) + 1
            new_depths = np.linspace(overall_top, overall_bot, num=num_points)
            
            res_df = pd.DataFrame({"DEPT": new_depths})

            # 逐个文件合并
            for data_key, data in well_data_dict.items():
                depth_col = data.columns[0]
                # 使用插值方法将数据映射到新深度轴上
                for col in data.columns[1:]:
                    # 处理重名曲线
                    new_col_name = col
                    k = 1
                    while new_col_name in res_df.columns:
                        new_col_name = f"{col}_{k}"
                        k += 1

                    res_df[new_col_name] = np.interp(res_df["DEPT"], data[depth_col], data[col], left=-999.25, right=-999.25)
            
            res_df = self._fill_missing_values(res_df)
            output_path = os.path.join(output_dir, f"{well}.csv")
            res_df.to_csv(output_path, encoding='gb2312', index=False)
            
        print("\n同名文件合并完成。")

    def gr_rt_sand(self, input_dir, output_dir, sand_base_log="Sand33", gr_log="GR_b_one", sand_rt_log="Sand_RT", min_layer_thickness=0.5):
        """
        利用GR与SP(或RT)的差异确定砂体顶底的干层。
        在已识别的砂岩段(Sand33)内部，根据另一条曲线(Sand_RT)的响应，
        将砂岩段的顶部和底部的非储层部分（如泥质夹层）识别为干层。
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        filelist = [f for f in listdir(input_dir) if f.lower().endswith('.csv')]
        
        process_bar = tqdm(filelist, desc="顶底干层识别", leave=True, ascii=True, ncols=80)
        for file in process_bar:
            wellname = str(file.split('.csv')[0])
            process_bar.set_description(f'处理中: {wellname}')
            
            log_data = pd.read_csv(os.path.join(input_dir, file), encoding='gb2312')
            dept_col = log_data.columns[0]
            
            if sand_base_log not in log_data.columns or gr_log not in log_data.columns or sand_rt_log not in log_data.columns:
                continue

            log_data["Sand34"] = log_data[sand_base_log] # 新的解释结论列
            
            # 预处理：在无效GR处或非砂岩段，将Sand_RT置为0
            log_data.loc[log_data[gr_log] == -999.25, sand_rt_log] = 0
            log_data.loc[log_data["Sand34"] == 0, sand_rt_log] = 0
            
            # 分段处理
            litho = log_data["Sand34"].values
            index_dict = self._get_block_index(litho)
            
            step = abs(log_data[dept_col].iloc[1] - log_data[dept_col].iloc[0])
            min_points = int(min_layer_thickness / step)
            
            for key in index_dict:
                index = index_dict[key]
                
                # 只处理砂岩段
                if log_data.loc[index[0], "Sand34"] == 0: continue
                
                block_rt = log_data.loc[index[0]:index[1], sand_rt_log].values
                
                # 从顶部向下查找第一个储层点
                first_reservoir_idx = -1
                for i, v in enumerate(block_rt):
                    if v == 1:
                        first_reservoir_idx = i
                        break
                
                # 如果顶部存在超过最小厚度的非储层，标记为干层(编码2)
                if first_reservoir_idx != -1 and first_reservoir_idx >= min_points:
                    log_data.loc[index[0]:index[0] + first_reservoir_idx - 1, "Sand34"] = 2
                
                # 从底部向上查找第一个储层点
                last_reservoir_idx = -1
                for i in range(len(block_rt) - 1, -1, -1):
                    if block_rt[i] == 1:
                        last_reservoir_idx = i
                        break

                # 如果底部存在超过最小厚度的非储层，标记为干层
                if last_reservoir_idx != -1 and (len(block_rt) - 1 - last_reservoir_idx) >= min_points:
                    log_data.loc[index[0] + last_reservoir_idx + 1:index[1], "Sand34"] = 2
            
            # 保存为LAS
            las = lasio.LASFile()
            las.well.WELL = wellname
            las.well.NULL = -999.25
            las.add_curve(dept_col, log_data[dept_col].values)
            las.add_curve("Sand34", log_data["Sand34"].values)
            output_las_path = os.path.join(output_dir, f"{wellname}.las")
            las.write(output_las_path, fmt='%.3f', version=2)
            
        process_bar.close()

    def sand_rt(self, input_dir, output_dir, rt_log="RTCAL", rt_threshold=1.0, filter_scale=4):
        """
        根据电阻率曲线识别砂岩和泥岩。
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filelist = [f for f in listdir(input_dir) if f.lower().endswith('.csv')]
        target_log = "Sand_RT"
        
        process_bar = tqdm(filelist, desc="电阻岩性识别", leave=True, ascii=True, ncols=80)
        for file in process_bar:
            wellname = str(file.split('.csv')[0])
            process_bar.set_description(f'处理中: {wellname}')
            
            log_data = pd.read_csv(os.path.join(input_dir, file), encoding='gb2312')
            log_data = self._fill_missing_values(log_data)
            depth_col = log_data.columns[0]
            
            if rt_log not in log_data.columns:
                continue
            
            # 复制一份用于处理，避免修改原数据
            rt_data = log_data[rt_log].copy()
            
            # 预处理
            rt_data[rt_data <= 0] = -999.25
            valid_mask = rt_data != -999.25
            rt_data[valid_mask] = np.log10(rt_data[valid_mask]) # 取对数
            
            # 平滑
            rt_data_smoothed = self._gauss_filter(rt_data.values, filter_scale)
            
            # 岩性解释 (0=泥岩, 1=砂岩)
            log_data[target_log] = 0
            log_data.loc[rt_data_smoothed > rt_threshold, target_log] = 1
            
            # 输出LAS
            output_df = log_data[[depth_col, target_log]]
            las = lasio.LASFile()
            las.well.WELL = wellname
            las.well.NULL = -999.25
            for curve in output_df.columns:
                las.add_curve(curve, output_df[curve].values)
            output_las_path = os.path.join(output_dir, f"{wellname}.las")
            las.write(output_las_path, fmt='%.3f', version=2)
            
        process_bar.close()

def _run_local_default_pipeline():
    from local_project_paths import INTERPRET_DIR, OUTPUT_ROOT
    from run_local_pipeline import main as run_local_main

    print("=============================================")
    print("      使用项目内默认路径执行重32一键流程")
    print("=============================================")
    print(f"项目目录: {OUTPUT_ROOT.parent}")
    print(f"输出目录: {OUTPUT_ROOT}")
    print("---------------------------------------------")

    run_local_main()

    print("---------------------------------------------")
    print(f"一键流程完成，最终解释目录: {INTERPRET_DIR}")


if __name__ == '__main__':
    _run_local_default_pipeline()
    raise SystemExit()

if False:  # Legacy path examples kept only as reference.
    processor = WellLogProcessor()

    print("=============================================")
    print("      测井数据处理及油水解释工具")
    print("=============================================")
    print("---------------------------------------------")

    # --- 1. 数据提取与合并 ---
    # # 提取数据并与井头坐标合并
    # processor.tqjt1(
    #     wellhead_file="G:/PM油田项目/pm_wellhead.xlsx",
    #     analysis_file="G:/PM油田项目/粘度数据/OIL ANALYSIS BANKERS WELLS.xlsx",
    #     output_dir="G:/PM油田项目/粘度数据/输出/"
    # )

    # # 对比两个井列表，提取共有井信息及差异井列表
    # processor.tqjt(
    #     stats_file="H:/PM油田项目/01测井数据/09水平井数据/水平井数据/Logs_statistics.xlsx",
    #     well_list_file="H:/PM油田项目/01测井数据/06petrel合并曲线后导出/工区所有井列表.xlsx",
    #     output_dir="./输出结果/井号对比/"
    # )

    # --- 2. 文件管理与格式转换 ---
    # # 检查一个井列表中的井，其对应的文件是否存在于指定文件夹
    # processor.well_files(
    #     stats_file="H:/PM油田项目/01测井数据/03曲线系列较全的老井0301补充/Logs_statistics.xlsx",
    #     folder_path="H:/PM油田项目/01测井数据/02测井数据处理/05GR处理/01GR异常处理/"
    # )

    # # 筛选并合并同名井的测井曲线，去除重复曲线
    # processor.jf_qx_shaixuan(
    #     input_dir="I:/张治恒拷贝/最新补充测井曲线0322/文件夹文件拆分后/csv/",
    #     output_dir="I:/张治恒拷贝/最新补充测井曲线0322/文件夹文件拆分后/曲线名称合并整理后/"
    # )
    
    # # 统计多层文件夹下的所有文件并生成井名列表Excel
    # processor.folder_files(
    #     root_dir="C:/Users/Administrator/Desktop/骨架剖面未处理井曲线-查找GR-DT资料齐全井资料包曲线/",
    #     output_excel_path="C:/Users/Administrator/Desktop/所有井名列表.xlsx",
    #     file_type=".las"
    # )

    # # 合并同一口井的多个测井文件
    # processor.logs_chongfutiaoxuan(
    #     input_dir="H:/PM油田项目/01测井数据/洲际补充资料0301/12 LAS - Open Hole LAS Files原始曲线/05有问题的井/",
    #     output_dir="H:/PM油田项目/01测井数据/洲际补充资料0301/12 LAS - Open Hole LAS Files原始曲线/合并后/"
    # )

    # --- 3. 曲线计算与岩性解释 ---
    # # 计算含油敏感曲线 RT_Oil
    # processor.qu_xian_chong_gou(
    #     input_dir="G:/新木项目/测井曲线/03归一化/",
    #     output_dir="G:/新木项目/测井曲线/05含油敏感线计算/csv/"
    # )


    # # 统一多种电阻率曲线名为 RTCAL
    # processor.rt_cal(
    #     input_dir="C:/Users/Administrator/Desktop/成果数据_前北区骨架井_20240417(1)/csv/",
    #     output_dir="C:/Users/Administrator/Desktop/成果数据_前北区骨架井_20240417(1)/RTCAL计算/",
    #     well_list_file='H:/PM油田项目/01测井数据/06petrel合并曲线后导出/02原始csv整合辽河后.xlsx' # 可选，用于筛选井
    # )

    # # 根据GR或SP或电阻率识别砂泥岩
    # processor.sand_rt(
    #     input_dir="H:/PM油田项目/01测井数据/06petrel合并曲线后导出/02原始csv/",
    #     output_dir="H:/PM油田项目/01测井数据/06petrel合并曲线后导出/21新建文件夹/04RTCAL岩性识别/",
    #     rt_log="RTCAL",
    #     rt_threshold=1.0,
    #     filter_scale=4
    # )

    # --- 4. 多源数据合并与处理 ---
    # # 合并多种方法计算的孔隙度结果
    # processor.por_combine(
    #     path_list=[
    #         "G:/PM油田项目/01测井数据/06petrel合并曲线后导出/24声波修正孔渗重新计算/04孔隙度计算/03偏移后/",
    #         "G:/PM油田项目/01测井数据/06petrel合并曲线后导出/08参数预测/02孔隙度计算/02利用密度计算/03偏移后/",
    #         "G:/PM油田项目/01测井数据/06petrel合并曲线后导出/08参数预测/02孔隙度计算/03利用中子计算/03偏移后/"
    #     ],
    #     output_dir="G:/PM油田项目/01测井数据/06petrel合并曲线后导出/24声波修正孔渗重新计算/04孔隙度计算/04合并后/"
    # )
    
    # # 合并多种方法计算的泥质含量结果
    # processor.vsh_combine(
    #     path_list=[
    #         "H:/PM油田项目/01测井数据/06petrel合并曲线后导出/08参数预测/01泥质含量计算/新建文件夹/01GR计算/",
    #         "H:/PM油田项目/01测井数据/06petrel合并曲线后导出/08参数预测/01泥质含量计算/新建文件夹/02SP_RT计算/"
    #     ],
    #     output_dir="H:/PM油田项目/01测井数据/06petrel合并曲线后导出/08参数预测/01泥质含量计算/新建文件夹/03合并后/"
    # )
    
    # # 将计算出的VSH参数合并回原始测井文件
    # processor.por_vsh_perm_sw_combine_logs(
    #     calculated_param_dir="H:/PM油田项目/01测井数据/06petrel合并曲线后导出/04参数预测/01泥质含量计算/05各类型计算结果合并/csv/",
    #     original_logs_dir="H:/PM油田项目/01测井数据/06petrel合并曲线后导出/02原始csv/",
    #     output_dir="H:/PM油田项目/01测井数据/06petrel合并曲线后导出/04参数预测/01泥质含量计算/06泥质含量与测井曲线合并后/",
    #     param_to_merge="VSH"
    # )

    # --- 5. 高级解释与结论生成 ---
    # # 综合解释砂体结论（油层、水层、干层等）
    # processor.jsjl_sand(
    #     input_dir="G:/PM油田项目/01测井数据/06petrel合并曲线后导出/24声波修正孔渗重新计算/08整合后peterl导出/02csv/",
    #     output_dir="G:/PM油田项目/01测井数据/06petrel合并曲线后导出/24声波修正孔渗重新计算/10解释结论/",
    #     sand_log="Sand36",
    #     rt_log="RTCAL",
    #     vsh_log="VSH",
    #     por_log="POR_PRE",
    #     sw_log="SW_PRE",
    #     target_log="Oil33"
    # )
    
    # # 利用GR和RT差异识别砂体顶底干层
    # processor.gr_rt_sand(
    #     input_dir="H:/PM油田项目/01测井数据/06petrel合并曲线后导出/02原始csv/",
    #     output_dir="H:/PM油田项目/01测井数据/06petrel合并曲线后导出/深度归位修正/",
    #     min_layer_thickness=0.5
    # )

    # # 根据分层数据，剔除非目的层的解释结论
    # processor.del_jsjl(
    #     interpretation_file="H:/PM油田项目/01测井数据/06petrel合并曲线后导出/16第二批井解释/10最终导出/砂体段测井解释.xlsx",
    #     tops_file="C:/Users/Administrator/Desktop/分层0110.xlsx",
    #     output_excel_path="H:/PM油田项目/01测井数据/06petrel合并曲线后导出/16第二批井解释/10最终导出/解释结论.xlsx"
    # )
    
    print("---------------------------------------------")
