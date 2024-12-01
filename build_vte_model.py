# -*- coding:utf-8 -*-
import utils
import os
import numpy as np
import joblib
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.linear_model import BayesianRidge, LogisticRegression
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.neural_network import MLPRegressor

data_dir = "./train_vali_data/2016_5_6/"
# 模型训练x
train_x = []
# 模型数据字典
train_code_dict = {}
# 模型label
train_y = []
train_yr = []
train_yr_thres = 1
# 是否开启概率过滤
prob_flag = True
with open(data_dir + "/train_code_data.txt", 'r') as rf:
    for perl in rf:
        perl = perl.strip()
        if len(perl) == 0: continue

        per_code, per_vte, per_nvte, per_ratio, per_sen, per_spec, per_label, per_prob, per_prob_label = [x.strip() for x in perl.split("\t", 8)]
        # 用于查询数据已经被包含的字典

        if (not prob_flag) or (prob_flag and float(per_prob) < 0.05):
            train_code_dict[per_code] = [int(per_vte), int(per_nvte), float(per_ratio), int(per_label)]
            # 训练数据
            per_data = [int(x) for x in per_code.split(";")[:-1]]
            train_x.append(per_data)
            # label
            train_y.append(int(per_label))
            if int(per_label) == 1:
                if float(per_ratio) < train_yr_thres:
                    train_yr_thres = float(per_ratio)
            # vte ratio
            train_yr.append(float(per_ratio))

train_x = np.array(train_x, dtype = np.integer)
train_y = np.array(train_y, dtype = np.integer)
train_yr = np.array(train_yr)
print("Y ratio Thres: {}".format(train_yr_thres))
# 构建决策树
clf = tree.DecisionTreeClassifier()
clf.fit(train_x, train_y)
pred_y = clf.predict(train_x)
train_acc = accuracy_score(train_y, pred_y)
print("Train Acc: {:.4f}".format(train_acc))

# 读入患者级别数据
#train_xf = np.loadtxt('./train_vali_data/{}/train_data.txt'.format(date_str), dtype = np.integer)
#train_yf = np.loadtxt('./train_vali_data/{}/train_y.txt'.format(date_str), dtype = np.integer)
#
## 构建患者级别模型
#sub_index = np.array(utils.equalSample(train_yf))
#txf_sub = train_xf[sub_index]
#tyf_sub = train_yf[sub_index]
#print("Sub Data Pos: {}, Num: {}".format(np.sum(tyf_sub), len(sub_index)))
#clfs_flist = [SVC(), RandomForestClassifier(), GradientBoostingClassifier(), LogisticRegression(), XGBClassifier()]
#clfs_f = []
#clfs_fn = ["SVM", "RF", "GBDT", "LR", "XGBoost"]
#for aml in clfs_flist:
#    aml_scores = cross_validate(aml, txf_sub, tyf_sub, scoring = 'roc_auc', cv = 5, return_estimator = True)
#    # print(aml_scores['test_score'])
#    atest_score = aml_scores['test_score']
#    cindex = np.argmax(atest_score)
#    aml = aml_scores['estimator'][cindex]
#    clfs_f.append(aml)

# 未知数据的预测模型
print("Train Num: {}, Pos: {}".format(len(train_x), np.sum(train_y)))
# clf2 = RandomForestRegressor(n_estimators = 50)
clf2 = MLPRegressor((20, ))
# sqrt放大极小概率阈值
train_yr = np.sqrt(train_yr)
# clf2.fit(train_x, train_yr)

# 交叉验证选择最大值
clf2_scores = cross_validate(clf2, train_x, train_yr, scoring = 'roc_auc', cv = 5, return_estimator = True)
atest_score = clf2_scores['test_score']
cindex = np.argmax(atest_score)
clf2 = clf2_scores['estimator'][cindex]

# 保存模型
joblib.dump(clf2, data_dir + '/model_save/clf2.joblib')
joblib.dump(clf, data_dir + '/model_save/clf1.joblib')

# 读入验证数据
vali_x = np.loadtxt(data_dir + "/vali_data.txt", dtype = np.integer)
vali_y = np.loadtxt(data_dir + "/vali_y.txt", dtype = np.integer)
vali_padua = np.loadtxt(data_dir + "/vali_padua_rank.txt", dtype = np.integer)
print("Vali Data: {}, VTE: {}".format(vali_x.shape, np.sum(vali_y)))

# padua在验证集上敏感度和特异度
padua_sen, padua_spec, tp, pos, tn, neg = utils.computeMetric(vali_padua, vali_y)
print("Vali Padua Sen: {}/{}={:.4f}, Spec: {}/{}={:.4f}".format(tp, pos, padua_sen, tn, neg,  padua_spec))

# 患者级别模型结果
#for k, aml in enumerate(clfs_f):
#    aml_preds = aml.predict(vali_x)
#
#    aml_sen, aml_spec, atp, apos, atn, aneg = utils.computeMetric(aml_preds, vali_y)
#    print("Vali Patient: {} Sen: {}/{}={:.4f}, Spec: {}/{}={:.4f}".format(clfs_fn[k], atp, apos, aml_sen, atn, aneg,  aml_spec))

# 根据数据是否已经出现分配到两个模型中
tp = 0
fp = 0
all_neg = 0
all_pos = 0

def getDataCode(x):
    xc = ""
    for perx in x:
        xc += str(perx) + ";"

    return xc
pred_y_list = []
vali_yr_thres = 0.5
cover_num = 0
pos_cover_num = 0
pred_result = []
pred_result_full = []
for i in range(len(vali_y)):
    per_x = vali_x[i,:]
    per_y = vali_y[i]
    per_code = getDataCode(per_x)
    pres = [per_y]
    # 数据已经出现过
    per_cover = 0
    if per_code in train_code_dict:
        per_origin_label = []
        pred_y = clf.predict(per_x.reshape(1, -1))[0]
        pres.append(pred_y)
        per_origin_label = train_code_dict[per_code]
        # print(per_orgin_label)
        cover_num += 1
        per_cover = 1
        if per_y == 1:
            per_res = [per_code, pred_y, per_y, per_cover]
            per_res.extend(per_origin_label)
            pred_result.append(per_res)
            pos_cover_num += 1
    else:
        pred_y = clf2.predict(per_x.reshape(1, -1))[0]
        if pred_y >= vali_yr_thres:
            pred_y = 1
        else:
            pred_y = 0

        pres.append(pred_y)
        if per_y == 1:
            per_res = [per_code, pred_y, per_y, per_cover]
            per_origin_label = [-1, -1, -1, -1]
            per_res.extend(per_origin_label)
            pred_result.append(per_res)

    pres.append(vali_padua[i])
    pred_result_full.append(pres)

    pred_y_list.append(pred_y)

model_sen, model_spec, tp, pos, tn, neg = utils.computeMetric(np.array(pred_y_list), vali_y)
print("Data Cover Rate: {}/{}={:.4f}, Pos Cover Rate: {}/{}={:.4f}".format(cover_num, len(vali_y), cover_num * 1.0 / len(vali_y), pos_cover_num, pos, pos_cover_num * 1.0 / pos))
print("Vali Model Sen: {}/{}={:.4f}, Spec: {}/{}={:.4f}".format(tp, pos, model_sen, tn, neg,  model_spec))

np.savetxt(data_dir+'/pred_label_merge.txt', pred_result_full, fmt="%d")

#with open(data_dir + "/pred_result_ratio.txt", 'w') as df:
#    df.write("\t".join(["code", "pred_label", "real_label", "is_covered", "vte_num", "nvte_num", "ratio", "label"]) + "\n")
#    for per_row in pred_result:
#        for i, perv in enumerate(per_row):
#            df.write("{}".format(perv))
#            if i != len(per_row) - 1:
#                df.write("\t")
#            else:
#                df.write("\n")
