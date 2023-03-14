import os
import numpy as np
import pandas as pd
import traceback
from kmodes.kmodes import KModes
from sklearn.cluster import KMeans
from loguru import logger

FEATURE_COLUMN = ["RESOURCE", "MGR_ID", "ROLE_ROLLUP_1", "ROLE_ROLLUP_2", "ROLE_DEPTNAME", "ROLE_TITLE", "ROLE_FAMILY_DESC", "ROLE_FAMILY", "ROLE_CODE"]
LABEL_COLUMN = ["ACTION"]

param_K = 15 # 聚类中心数量
param_TP = 0.5 # attribute filter tp
param_TN = 0.5 # attribute filter tn
param_RP = 0.99 # relation condition rp
param_RN = 0.99 # relation condition rn
param_TRAIN_NUM = 1000 # 训练数据的数量
param_Debug = False # debug输出中间结果

logger.add("../exp/NUM={}K={}_TP={}_TN={}_RP={}_RN={}_exp.log".format(param_TRAIN_NUM, param_K, param_TP, param_TN, param_RP, param_RN))


def match_pf(pf, item):
    # 判断item是否满足所有 filter
    for f in pf:
        try: 
            # 对于正过滤器 item[feature] |= filter(feature, val) 
            # 对于负过滤器 item[feature] |= filter(feature, !val) 
            if (f[2]==1 and item.get(f[0]).any()==f[1]) or (f[2]==-1 and item.get(f[0]).any()!=f[1]):
                continue
            else:
                return 0
            
        except Exception as e:
            logger.error("{} {} {}".format(f[0], f[1], f[2]))
            logger.error(item)
            logger.error(e)

    return 1

def match_pr(pr, item):
    # 判断item是否满足所有 relation
    for r in pr:
        try:
            # 对于正条件 item[feature_a, feature_b] |= relation(feature_a, feature_b) 
            # 对于负条件 item[feature_a, feature_b] |= relation(feature_a, !feature_b) 
            if (r[2]==1 and item.get(r[0]).any()==item.get(r[1]).any()) or (r[2]==-1 and item.get(r[0]).any()!=item.get(r[1]).any()):
                continue
            else:
                return 0

        except Exception as e:
            logger.error("{} {} {}".format(r[0], r[1], r[2]))
            logger.error(item)
            logger.error(e)

    return 1

def decide(p, item):
    match_filter_flag = match_pf(p['F'], item)
    match_relation_flag = match_pr(p['R'], item)

    if param_Debug and match_filter_flag:
        logger.warning("Filter match")
    if param_Debug and match_relation_flag:
        logger.warning("Relation match")

    if match_filter_flag and match_relation_flag:
        return 1 
    
    return 0

def test_decide():
    p1 = {"F":set([('MGR_ID',2), ('ROLE_ROLLUP_1',3)]), "R":set([('MGR_ID','ROLE_ROLLUP_1')]), "OP":set([1,2])}
    items = pd.DataFrame({'MGR_ID':[2, 3], 'ROLE_ROLLUP_1':[2, 3]})
    item = items[0:1]
    logger.info(decide(p1, item))

def getPredict(p, dataset):
    # 功能：针对指定的单条ABAC Role, 测试结果
    preds = []
    for i in range(len(dataset)):
        preds.append(decide(p, dataset[i:i+1]))
    preds_data = pd.concat((pd.DataFrame({"preds": preds}), dataset), axis=1)
    FNs = preds_data.query('preds=={} and ACTION=={}'.format(1, 0))
    FPs = preds_data.query('preds=={} and ACTION=={}'.format(0, 1))
    TNs = preds_data.query('preds=={} and ACTION=={}'.format(0, 0))
    TPs = preds_data.query('preds=={} and ACTION=={}'.format(1, 1))
    return FNs, FPs, TNs, TPs

def getPredictV2(Pi, dataset):
    # 功能：针对指定的ABAC Policy, 测试dataset中所有request的授权结果
    # 原理：对于单个request, 存在规则p属于策略Pi, 使得item|=p, 则允许该request的请求
    preds = []
    for i in range(len(dataset)):
        allow = 0
        for j in Pi.keys():
            allow = allow | decide(Pi[j], dataset[i:i+1]) # 因为这边用的是 |, 因此decide全为0 
        preds.append(1 if allow else 0)

    preds_data = pd.concat((pd.DataFrame({"preds": preds}), dataset), axis=1)  
    FNs = preds_data.query('preds=={} and ACTION=={}'.format(1, 0))
    FPs = preds_data.query('preds=={} and ACTION=={}'.format(0, 1)) # 0 1
    TNs = preds_data.query('preds=={} and ACTION=={}'.format(0, 0)) # 0 0 也就是说这玩意全部预测为0了
    TPs = preds_data.query('preds=={} and ACTION=={}'.format(1, 1)) 

    Pos = preds_data.query('ACTION=={}'.format(1)) 
    Neg = preds_data.query('ACTION=={}'.format(0)) 
    logger.info("pos: {} neg:{}".format(len(Pos), len(Neg)))

    return FNs, FPs, TNs, TPs

def WSC(p):
    # 功能： 计算单条策略的加权结构复杂度
    wsc = 0.5 * len(p["F"]) + 0.5 * len(p["R"])
    return wsc

def pSub(p, i_sub):
    p_sub = p.copy()
    if i_sub in p_sub.keys():
        del p_sub[i_sub]
    return p_sub

def jacardSim(s1, s2):
    interact = s1 & s2
    union = s1 | s2
    sim = len(interact) / len(union)
    return sim

def similarity(p1, p2):
    # 功能：计算两个策略之间的相似度
    # input: p {"F":set(), "R":set(), "OP":set()}
    # interact = len(p1['F'] & p2['F']) + len(p1['R'] & p2['R']) + len(p1['OP'] & p2['OP'])
    # union = len(p1['F'] | p2['F']) + len(p1['R'] | p2['R']) + len(p1['OP'] | p2['OP'])
    interact = len(p1['F'] & p2['F']) + len(p1['R'] & p2['R']) 
    union = len(p1['F'] | p2['F']) + len(p1['R'] | p2['R'])

    sim = interact / union if union>0 else 0
    return interact, union, sim

def getSimilarRules(p1, p2):
    p = {}
    k = 0
    for i in p1.keys():
        for j in p2.keys():
            if similarity(p1[i], p2[j]) > 0.5:
                p[i] 
    return p

def freq_filter(val, feature, df_ci):
    if param_Debug:
        logger.info("freq_filter start feature-val {} {}".format(feature, val))

    value_counts = df_ci[feature].value_counts()
    if val not in value_counts.keys():
        return 0
    return value_counts[val]

def freq_relation(feature_a, feature_b, df_ci, topic=None):
    if param_Debug:
        logger.info("{} freq_relation start feature {} {}".format(topic, feature_a, feature_b))
    res = df_ci.apply(lambda x: x[feature_a] == x[feature_b], axis=1)
    freq = res.sum() / len(df_ci) if len(df_ci)>0 else 0
    return freq

class AmazonData(object):
    def __init__(self, data_path="../data/amazon", num=None):
        try:
            self.train_set = pd.read_csv(os.path.join(data_path, "train.csv"))[0:num] if num is not None else pd.read_csv(os.path.join(data_path, "train.csv"))
            self.test_set = pd.read_csv(os.path.join(data_path, "test.csv"))
            self.train_x, self.train_y = self.train_set[FEATURE_COLUMN], self.train_set[LABEL_COLUMN]
            self.test_x = self.test_set[FEATURE_COLUMN]
        except:
            logger.error(traceback.format_exc())
        
        logger.info(self.train_set.head(5))
    
    def data_static(self): 
        static_res = {}
        for key in FEATURE_COLUMN:
            static_res[key] = len(self.train_set[key].value_counts())

        for key in LABEL_COLUMN:
            value_counts = self.train_set[key].value_counts()
            static_res[key] = len(value_counts)
            logger.info(value_counts)

        logger.info(static_res)
        return static_res

    def get_data(self):
        return np.array(self.train_x), np.array(self.train_y).ravel()
    

class AutomaticPolicyExtraction(object):
    def __init__(self, dataset, cluster=2):
        self.cluster = cluster
        self.dataset = dataset.train_set
        train_data, _= data.get_data()
        label, cost = self.cluster_data(train_data, cluster, algo="kmodes")
        self.dataset = pd.concat((pd.DataFrame({"cluster_k":label}), self.dataset), axis=1)

        C_dict = {}
        C_dict_freq = {}
        C_dict_hist = {}
        for i in range(self.cluster):
            C_dict[i] = self.dataset.query('cluster_k=={}'.format(i))
            C_dict_freq[i] = {}
            C_dict_hist[i] = len(C_dict[i])
            for feature in FEATURE_COLUMN:
                C_dict_freq[i][feature] = C_dict[i][feature].value_counts().to_dict()
               
        C_dict_freq[9999] = {}
        C_dict_hist[9999] = len(self.dataset)
        for feature in FEATURE_COLUMN:
            C_dict_freq[9999][feature] = self.dataset[feature].value_counts().to_dict()
        self.C_dict = C_dict
        self.C_dict_freq = C_dict_freq
        self.C_dict_hist = C_dict_hist

        self.freq_relations_L = {}
        for feature_a in FEATURE_COLUMN:
            for feature_b in FEATURE_COLUMN:
                if feature_a==feature_b:
                    continue
                self.freq_relations_L[(feature_a, feature_b)] = freq_relation(feature_a, feature_b, self.dataset, topic="ALL")

        V = {}
        for feature in FEATURE_COLUMN:
            V[feature] = set(self.dataset[feature].values)
        self.V = V
        self.A = FEATURE_COLUMN
        self.L = self.dataset

        self.P_list = []
        self.P_wsc_max = None 
        self.P_wsc_max_score = 1

    def freq_filter(self, val, feature, cluster_k):
        value_counts = self.C_dict_freq[cluster_k][feature]
        cluster_k_hist = self.C_dict_hist[cluster_k]
        if val not in value_counts.keys():
            return 0
        return value_counts[val] / cluster_k_hist

    def policy_rules_extraction(self):
        P_dict = {}
        for i in range(self.cluster):
            C_i = self.C_dict[i]
            F_res = self.extract_attribute_filters(i, self.A, self.V, self.L)
            R_res = self.extract_relations(i, C_i, self.A, self.L)
            P_dict[i] = {"F": F_res, "R": R_res}
            logger.info("F_res: {}".format(F_res))
            logger.info("R_res: {}".format(R_res))
        return P_dict

    def cluster_data(self, data, k, algo="kmeans"):
        logger.info("cluster_data start")
        if algo=="kmeans":
            clf = KMeans(n_clusters=k)
            clf.fit(data)
            cost = clf.inertia_
            label = self.labels_
            logger.info("k: {}, cost:{}".format(k, cost))
        elif algo=="kmodes":
            clf = KModes(n_clusters=k)
            clf.fit_predict(data)
            cost = clf.cost_
            label = clf.labels_
            logger.info("k: {}, cost:{}".format(k, cost))
        else:
            logger.info("algo {} Not Imp".format(algo))

        logger.info("cluster_data ended")
        return label, cost

    def parameter_tuning(self, data, s_min=2, s_max=20, algo="kmeans", plot=False, save=False):
        if algo=="kmeans":
            for k in range(s_min, s_max):
                clf = KMeans(n_clusters=k)
                clf.fit(data)
                score = clf.inertia_
                if k == s_min:
                    min_score = score
                if score < min_score:
                    min_cost_k = k
                    min_score = score
                logger.info("k: {} score: {}".format(k, score))
            logger.info("best k: {}, best score:{}".format(min_cost_k, min_score))
            self.cluster_num = min_cost_k
        elif algo=="kmodes":
            for k in range(s_min, s_max):
                clf = KModes(n_clusters=k)
                clf.fit_predict(data)
                score = clf.cost_
                if k == s_min:
                    min_score = score
                if score < min_score:
                    min_cost_k = k
                    min_score = score
                logger.info("k: {} score: {}".format(k, score))
            logger.info("best k: {}, best score:{}".format(min_cost_k, min_score))
            self.cluster_num = min_cost_k

    def extract_attribute_filters(self, cluster_i, A, V, L, threshold_p=param_TP, threshold_n=param_TN):
        logger.info("cluster {} extract_attribute_filters start".format(cluster_i))
        F_res = set()
        for feature_a in A:
            for value_j in V[feature_a]:
                freq_a_b_c = self.freq_filter(value_j, feature_a, cluster_i)
                freq_a_b_l = self.freq_filter(value_j, feature_a, 9999)

                if freq_a_b_c - freq_a_b_l>threshold_p:
                    logger.warning("pos filter =  freq_a_b_c :{} freq_a_b_l:{}".format(freq_a_b_c, freq_a_b_l))
                    F_res.add((feature_a, value_j, 1))
                if freq_a_b_l - freq_a_b_c >threshold_n:
                    logger.warning("neg filter =  freq_a_b_c :{} freq_a_b_l:{}".format(freq_a_b_c, freq_a_b_l))
                    F_res.add((feature_a, value_j, -1))
        
        logger.info("cluster {} extract_attribute_filters ended".format(cluster_i))
        return F_res

    def extract_relations(self, cluster, C_i, A, L, threshold_p=param_RP, threshold_n=param_RN):
        logger.info("cluster {} extract_relations start".format(cluster))
        R_res = set()
        for feature_a in A:
            for feature_b in A:
                if feature_a==feature_b:
                    continue
                freq_a_b_c = freq_relation(feature_a, feature_b, C_i, topic=cluster)
                freq_a_b_l = self.freq_relations_L[(feature_a, feature_b)]

                if freq_a_b_c - freq_a_b_l >threshold_p:
                    logger.warning("pos relation =  freq_a_b_c :{} freq_a_b_l:{}".format(freq_a_b_c, freq_a_b_l))
                    R_res.add((feature_a, feature_b, 1))
                if freq_a_b_l - freq_a_b_c>threshold_n:
                    logger.warning("neg relation = freq_a_b_c :{} freq_a_b_l:{}".format(freq_a_b_l, freq_a_b_c))
                    R_res.add((feature_a, feature_b, -1))

        logger.info("cluster {} extract_relations ended".format(cluster))
        return R_res

    def rule_pruning(self, p):
        logger.info("rule pruning start")
        q = self.calcQuality(p)
        exlude_p_list = []
        for i in range(self.cluster):
            for j in range(i+1, self.cluster):
                # 如果该策略已被裁剪掉了，则跳过
                if i in exlude_p_list or j in exlude_p_list:
                    continue
                interact, union, sim = similarity(p[i], p[j])
                logger.warning("policy {}, {} interact:{}, union:{} similarity:{}".format(i, j, interact, union, sim))
                if sim>0.5:
                    logger.info("rule pruning {} {}".format(i, j))
                    p_i_temp = pSub(p, i)
                    p_j_temp = pSub(p, j)
                    q_i = self.calcQuality(p_i_temp)
                    q_j = self.calcQuality(p_j_temp)
                    if q_i >= q and q_i >= q_j:
                        p = p_i_temp
                        exlude_p_list.append(i)
                    if q_j >= q and q_j >= q_i:
                        p = p_j_temp
                        exlude_p_list.append(j)
        logger.info("rule pruning ended")
        return p
    
    def refine_policy(self, p):
        # TODO: 待实现
        logger.info("refine plolicy start")
        # FNs, FPs, _, _ = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        # for i in p.keys():
        #     FNs_t, FPs_t, _, _ = getPredict(p[i], self.dataset)
        #     FNs = pd.concat((FNs, FNs_t), axis=1)
        #     FPs = pd.concat((FPs, FPs_t), axis=1)
        
        # p_fn = self.policy_rules_extraction()
        # p_fp = self.policy_rules_extraction()

        # R_s = getSimilarRules(p_fn, p)
        # for p_i in p_fn:
        #     if len(R_s)==0:
        #         p = p | p_i
        #     else:
        #         for p_j in R_s:
        #             p_j = p_j & p_i 

        # R_s = getSimilarRules(p_fn, p)
        # for p_i in p_fp:
        #     if len(R_s)==0:
        #         p = p | p_i
        #     else:
        #         for p_j in R_s:
        #             p_j = p_j & p_i 
        logger.info("refine plolicy ended")

    def calcQuality(self, p, cluster="all"):
        alpha = 0.5
        if cluster=="all":
            FNs, FPs, TNs, TPs = getPredictV2(p, self.dataset)
            fn, fp, tn, tp = len(FNs), len(FPs), len(TNs), len(TPs)
            logger.info("fn fp tn tp {} {} {} {}".format(fn, fp, tn, tp))
            precision = tp / (tp + fp) if (tp + fp) >0 else 0
            recall = tp / (tp + fn) if (tp + fn) >0 else 0
            acc = (tp + tn) / (fn + fp + tp + tn)
            f_score = 2 * (precision * recall) / (precision + recall) if (precision + recall)>0 else 0

            wsc = 0
            for key in p.keys():
                wsc += WSC(p[key])
            
            if self.P_wsc_max==None or wsc > self.P_wsc_max_score:
                self.P_wsc_max = p 
                self.P_wsc_max_score = wsc 
            
            delta_wsc = (self.P_wsc_max_score - wsc + 1) / self.P_wsc_max_score 
            Q = 1/((alpha/f_score) + (1-alpha)/delta_wsc) if f_score>0 else delta_wsc
            logger.warning("Q {} f1-score {} precision {} recall {} wsc {} delta_wsc {}".format(Q, f_score, precision, recall, wsc, delta_wsc))
            return  Q
        else:
            FNs, FPs, TNs, TPs = getPredict(p, self.dataset)
            fn, fp, tn, tp = len(FNs), len(FPs), len(TNs), len(TPs)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            acc = (tp + tn) / (fn + fp + tp + tn)
            f_score = 2 * (precision * recall) / (precision + recall)
            wsc = WSC(p)

            if self.P_wsc_max==None or wsc > self.P_wsc_max_score:
                self.P_wsc_max = p 
                self.P_wsc_max_score = wsc 
            
            delta_wsc = (self.P_wsc_max_score - wsc + 1) / self.P_wsc_max_score 
            Q = 1/((alpha/f_score) + (1-alpha)/delta_wsc)
            return  Q

if __name__ =='__main__':
    # test_decide()
    data = AmazonData(num=param_TRAIN_NUM)
    data.data_static()

    model = AutomaticPolicyExtraction(data, param_K)
    p = model.policy_rules_extraction()
    p = model.rule_pruning(p)
    p = model.refine_policy(p)
