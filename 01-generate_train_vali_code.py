#!/usr/bin/env python
# coding=utf-8
import numpy as np

time_range = [[2016, 5], [2016, 6], [2016, 7]]
data_dir = "./train_vali_data/"

def readData(data_path, prefix):
    return np.loadtxt(data_path + "/" + prefix + "_data.txt", dtype = np.integer), np.loadtxt(data_path + "/" + prefix + "_code.txt", dtype = str), np.loadtxt(data_path + "/" + prefix + "_y.txt", dtype = np.integer), np.loadtxt(data_path + "/" + prefix + "_padua_rank.txt", dtype = np.integer)

sen_thres = 0.9
spec_trhes = 0.7

def computePopProb(pnum, nnum, vprob):
    prob = np.exp(pnum * np.log(vprob) + nnum * np.log(1 - vprob))
    return prob

for per_time in time_range:
    vte_code_dict = {}
    per_dir = data_dir + "/{}_{}/".format(per_time[0], per_time[1])
    print("Time: {}".format(per_time))
    padua_tp = 0
    all_pos = 0
    all_neg = 0
    padua_fp = 0

    train_data, train_code, train_y, train_prank = readData(per_dir, "train")
    # 统计code的vte, non-vte频率
    for i, pcode in enumerate(train_code):
        pery = train_y[i]
        if not pcode in vte_code_dict:
            vte_code_dict[pcode] = {0: 0, 1: 0}

        if pery == 0:
            all_neg += 1
            if train_prank[i] == 1:
                padua_fp += 1
        else:
            all_pos += 1
            if train_prank[i] == 1:
                padua_tp += 1


        vte_code_dict[pcode][pery] += 1


    vte_code_stat_list = []
    for perk in vte_code_dict:
        per_vte = vte_code_dict[perk][1]
        per_nvte = vte_code_dict[perk][0]
        per_ratio = per_vte * 1.0 / (per_nvte + per_vte)

        vte_code_stat_list.append([perk, per_vte, per_nvte, per_ratio])
    vte_code_stat_list.sort(key = lambda x: (x[3], x[1], - x[2]), reverse = True)

    padua_sen = padua_tp * 1.0 / all_pos
    padua_spec = 1 - padua_fp * 1.0 / all_neg
    print("Padua Sen: {}/{} = {:.4f}, Spec: {}/{} = {:.4f}".format(padua_tp, all_pos, padua_sen, all_neg - padua_fp, all_neg, padua_spec))

    vte_ratio = 78 / 3115# all_pos * 1.0 / (all_pos + all_neg)
    # print("vte ratio: {}/{} = {:.4f}".format(all_pos, all_pos + all_neg, vte_ratio))

    #计算每个code敏感度特异度
    cul_pos = 0
    cul_neg = 0

    code_pos = 0
    code_neg = 0
    for per in vte_code_stat_list:
        per_pos = per[1]
        per_neg = per[2]

        cul_pos += per_pos
        cul_neg += per_neg

        per_sen = cul_pos * 1.0 / all_pos
        per_spec = 1 - cul_neg * 1.0/ all_neg
        per.append(per_sen)
        per.append(per_spec)

        # 计算子人群概率
        per_prob = computePopProb(per_pos, per_neg, vte_ratio)

        per_y = 0
        if per_sen <= sen_thres and per_spec >= spec_trhes:
        # if per_spec >= spec_trhes:
            per_y = 1
            code_pos += 1
        else:
            code_neg += 1

        per.append(per_y)
        per.append(per_prob)

        # 根据prob进行标记
        if per_prob < 0.05:
            per.append(1)
        else:
            per.append(0)

    with open(per_dir + "/train_code_data.txt", 'w') as tf:
        print("Code Sample: Pos: {} Neg: {}".format(code_pos, code_neg))
        for per in vte_code_stat_list:
            tf.write("{}\t{}\t{}\t{}\t{:.3f}\t{:.3f}\t{}\t{:.4f}\t{}\n".format(per[0], per[1], per[2], per[3], per[4], per[5], per[6], per[7], per[8]))
