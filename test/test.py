# def parameter_tuning(self, data, s_min=2, s_max=20, algo="kmeans", plot=False, save=False):
#         if algo=="kmeans":
#             for k in range(s_min, s_max):
#                 clf = KMeans(n_clusters=k)
#                 clf.fit(data)
#                 score = clf.inertia_
#                 if k == s_min:
#                     min_score = score
#                 if score < min_score:
#                     min_cost_k = k
#                     min_score = score
#                 logger.info("k: {} score: {}".format(k, score))
#             logger.info("best k: {}, best score:{}".format(min_cost_k, min_score))
#             self.cluster_num = min_cost_k
#         elif algo=="kmodes":
#             for k in range(s_min, s_max):
#                 clf = KModes(n_clusters=k)
#                 clf.fit_predict(data)
#                 score = clf.cost_
#                 if k == s_min:
#                     min_score = score
#                 if score < min_score:
#                     min_cost_k = k
#                     min_score = score
#                 logger.info("k: {} score: {}".format(k, score))
#             logger.info("best k: {}, best score:{}".format(min_cost_k, min_score))
#             self.cluster_num = min_cost_k