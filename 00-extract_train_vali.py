# -*- coding:utf-8 -*-
import time
import numpy as np
import utils
import os

# 时间范围:筛选验证集
time_range = [[2016, 5], [2016, 6], [2016, 7]]

# 数据保存目录
data_dir = "./train_vali_data/"
utils.safe_mkdir(data_dir)

# vte特征数据
vte_data = []
vte_code_list = []

feature_index = list(range(11))
# 雌孕激素
feature_index.append(11)
# 机械通气
feature_index.append(19)

# padua分级
padua_rank = []
# padua评分
padua_score = []
# vte 发生
vte_label = []
# 入院时间
admit_time = []
patient_index = 0
# 科室编码
office_code_dict = {}

# 按月份进行统计
year_month_num = {}
nontime_num = {}
nontime_vte_num = 0

# 按特征编码统计每种特征组合情况下vte non-vte数量
vte_code_dict = {}

padua_tp = 0
padua_fp = 0
with open("./vte_dataset.txt", 'r') as df:
    for i, perl in enumerate(df):
        perl = perl.strip()
        if i == 0 or len(perl) == 0: continue

        per_sp = [x.strip() for x in perl.split("\t")]
        # 获得特征数据
        per_feature = []
        per_code = ""
        for j in feature_index:
            per_feature.append(int(per_sp[j]))
            per_code += per_sp[j] + ";"
        if not per_code in vte_code_dict:
            vte_code_dict[per_code] = {1: 0, 0: 0}

        # 每个样本的code, padua rank, score, label
        vte_data.append(per_feature)
        vte_code_list.append(per_code)
        padua_rank.append(int(per_sp[23]))
        padua_score.append(float(per_sp[22]))
        vte_label.append(int(per_sp[26]))

        if not per_code in office_code_dict:
            office_code_dict[per_code] = {}
        per_office = int(per_sp[27])
        if not per_office in office_code_dict[per_code]:
            office_code_dict[per_code][per_office] = 0
        office_code_dict[per_code][per_office] += 1

        # 统计每种特征编码下vte和non-vte人数
        vte_code_dict[per_code][int(per_sp[26])] += 1
        if padua_rank[-1] == 1:
            if vte_label[-1] == 1:
                padua_tp += 1
            elif vte_label[-1] == 0:
                padua_fp += 1

        # 每个样本的时间
        if per_sp[29] != "None":
            # admit_time.append(time.strptime(per_sp[29], "%Y/%m/%d"))
            # 按年月统计
            per_year, per_month, per_day = [int(x.strip()) for x in per_sp[29].split("/")]
            admit_time.append([per_year, per_month, per_day])

            if not per_year in year_month_num:
                year_month_num[per_year] = {}
            if not per_month in year_month_num[per_year]:
                year_month_num[per_year][per_month] = 0
            year_month_num[per_year][per_month] += 1
        else:
            nontime_num[patient_index] = int(per_sp[26])
            admit_time.append("None")
            if vte_label[-1] == 1:
                nontime_vte_num += 1

        patient_index += 1

# 数据进行基本统计
print("Data Num: {}".format(len(vte_label)))
print("VTE Num: {}".format(np.sum(vte_label)))
print("Year Month Num: ")
print(year_month_num)
print("Non Time Num: {}, VTE Num: {}".format(len(nontime_num), nontime_vte_num))

vte_code_stat_list = []
for perk in vte_code_dict:
    per_vte = vte_code_dict[perk][1]
    per_nvte = vte_code_dict[perk][0]
    per_ratio = per_vte * 1.0 / (per_nvte + per_vte)

    per_office_dict = office_code_dict[perk]
    pero_list = list(per_office_dict.items())
    pero_list.sort(key = lambda x: x[1], reverse  = True)

    pero_str = ""
    for pero in pero_list:
        pero_str += str(pero[0]) + "_" + str(pero[1]) + "/"

    vte_code_stat_list.append([perk, per_vte, per_nvte, per_ratio, pero_str])
vte_code_stat_list.sort(key = lambda x: (x[3], x[1], - x[2]), reverse = True)

all_pos = 230
all_neg = 3054
padua_sen = padua_tp * 1.0 / all_pos
padua_spec = 1 - padua_fp * 1.0 / all_neg
print("Padua Sen: {:.4f}, Spec: {:.4f}".format(padua_sen, padua_spec))

# 保存数据
def save_data(save_dir, prefix, per_data, per_code, per_y, per_prank):
    path_prefix = save_dir + "/" + prefix
    np.savetxt(path_prefix + "_data.txt", per_data, "%d")
    np.savetxt(path_prefix + "_code.txt", per_code, fmt = "%s")
    np.savetxt(path_prefix + "_y.txt", per_y, fmt = "%d")
    np.savetxt(path_prefix + "_padua_rank.txt", per_prank, fmt = "%d")

# 按时间划分数据集
for per_range in time_range:
    per_year, per_month = per_range
    print("Year: {}, Month: {}".format(per_year, per_month))
    per_train_data = []
    per_train_prank = []
    per_vali_data = []
    per_train_code = []
    per_vali_code = []
    per_train_y = []
    per_vali_y = []
    per_vali_prank = []

    for i, perd in enumerate(vte_data):
        per_code = vte_code_list[i]

        per_time = admit_time[i]
        #print(per_time)
        if per_time == "None":
            per_train_data.append(perd)
            per_train_code.append(per_code)
            per_train_y.append(vte_label[i])
            per_train_prank.append(padua_rank[i])
        else:
            if per_time[0] == per_year and per_time[1] == per_month:
                per_vali_data.append(perd)
                per_vali_code.append(per_code)
                per_vali_y.append(vte_label[i])
                per_vali_prank.append(padua_rank[i])
            else:
                per_train_data.append(perd)
                per_train_code.append(per_code)
                per_train_y.append(vte_label[i])
                per_train_prank.append(padua_rank[i])

    # 保存结果
    save_dir = data_dir + "/{}_{}/".format(per_year, per_month)
    utils.safe_mkdir(save_dir)

    save_data(save_dir, "train", per_train_data, per_train_code, per_train_y, per_train_prank)
    save_data(save_dir, "vali", per_vali_data, per_vali_code, per_vali_y, per_vali_prank)

'''
cul_pos = 0
cul_neg = 0
for per in vte_code_stat_list:
    per_pos = per[1]
    per_neg = per[2]

    cul_pos += per_pos
    cul_neg += per_neg

    per_sen = cul_pos * 1.0 / all_pos
    per_spec = 1 - cul_neg * 1.0/ all_neg
    per.append(per_sen)
    per.append(per_spec)
# vte_code_stat_list.sort(key = lambda x: x[0])
has_vte_code_num = 0
has_vte_code_1num = 0
has_vte_code_0num = 0
for per in vte_code_stat_list:
    if per[1] != 0:
        has_vte_code_num += 1
        has_vte_code_1num += per[1]
        has_vte_code_0num += per[2]
print("All Code Type Num: {}".format(len(vte_code_stat_list)))
print("Has VTE code Type Num: {}".format(has_vte_code_num))
print("Has VTE code 1-Num: {}, 0-Num: {}".format(has_vte_code_1num, has_vte_code_0num))
with open("vte_code_stat.txt", 'w') as cf:
    cf.write("\t".join(["code", "vte", "non-vte", "vte/non-vte", "office_distribution", "sen", "spec"]) + "\n")
    for per in vte_code_stat_list:
        # print(per)
        for j, perv in enumerate(per):

            if j == len(per) - 1:
                cf.write("{:.4f}\n".format(perv))
            elif j == 4:
                cf.write("{}\t".format(perv))
            elif j > 2:
                cf.write("{:.4f}\t".format(perv))
            else:
                cf.write("{}\t".format(perv))
'''
