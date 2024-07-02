from sklearn.cluster import KMeans
import time
import numpy as np
from util import *
from pprint import pprint

def initial_bundles_with_kmeans(K, dist_mat, all_orders, num_clusters):
    pickups = dist_mat[:K, :K]
    delvs = dist_mat[K:, K:]
    pickup_to_delvs = dist_mat[:K, K:]
    delv_to_pickups = dist_mat[K:, :K]
    # (pickup_to_delvs + delv_to_pickups) / 2
    # pickup_to_delvs, delv_to_pickups

    alpha, beta, gamma = 1, 1, 1
    sum_mat = alpha * pickups + beta * delvs + gamma * (pickup_to_delvs + delv_to_pickups) / 2
    num_clusters = 8

    cluster = KMeans(n_clusters=num_clusters)
    # cluster = DBSCAN(eps=0.5, min_samples=2)
    # cluster = AgglomerativeClustering(n_clusters=8)
    labels = cluster.fit_predict(sum_mat)
    
    all_bundles = [[] for _ in range(num_clusters)]
    for i in range(num_clusters):
        order_indices = np.where(labels == i)[0].tolist()
        # orders which is in i-th cluster(bundle)
        for ord_idx in order_indices:
            all_bundles[i].append(all_orders[ord_idx])


    return all_bundles

def calculate_bundle_dist(dist_mat, bundle_orders):
    # 새로운 번들의 비용 계산 로직
    # 거리와 기타 요인들을 고려한 비용 계산
    shop_seq = [order.id for order in bundle_orders]
    dlv_seq = [order.id for order in bundle_orders]

    dist = 0
    for i in range(len(shop_seq) - 1):
        dist += dist_mat[shop_seq[i], shop_seq[i + 1]]
    for j in range(len(dlv_seq) - 1):
        dist += dist_mat[dlv_seq[j], dlv_seq[j + 1]]
    return dist
        
def algorithm(K, all_orders, all_riders, dist_mat, timelimit=60):
    start_time = time.time()

    for r in all_riders:
        r.T = np.round(dist_mat / r.speed + r.service_time)

    # A solution is a list of bundles
    solution = []

    # ------------- Custom algorithm code starts from here --------------#

    num_clusters = 8  # 라이더 수만큼 군집 생성
    initial_bundles = initial_bundles_with_kmeans(K, dist_mat, all_orders, num_clusters)
    
    bike_rider = None
    for r in all_riders:  # TODO: 
        if r.type == 'BIKE':
            bike_rider = r

    all_bundles = []
    for bundle_orders in initial_bundles:
        new_bundle = Bundle(
            all_orders=all_orders,
            rider=bike_rider,
            shop_seq=[order.id for order in bundle_orders],
            dlv_seq=[order.id for order in bundle_orders],
            total_volume=sum(order.volume for order in bundle_orders),
            total_dist=calculate_bundle_dist(dist_mat, bundle_orders)
        )
        all_bundles.append(new_bundle)
        for r in all_riders:  # TODO:
            if r.type == 'BIKE':
                r.available_number -= 1
    # pprint(all_bundles)
    

    best_obj = sum((bundle.cost for bundle in all_bundles)) / K
    print(f'Best obj = {best_obj}')

    # for bundle in all_bundles:
    #     new_rider = get_cheaper_available_riders(all_riders, bundle.rider)
    #     if new_rider is not None:
    #         old_rider = bundle.rider
    #         if try_bundle_rider_changing(all_orders, bundle, new_rider):
    #             old_rider.available_number += 1
    #             new_rider.available_number -= 1

    #         if time.time() - start_time > timelimit:
    #             break

    #     cur_obj = sum((bundle.cost for bundle in all_bundles)) / K
    #     if cur_obj < best_obj:
    #         best_obj = cur_obj
    #         print(f'Best obj = {best_obj}')

    # Solution is a list of bundle information
    solution = [
        # rider type, shop_seq, dlv_seq
        [bundle.rider.type, bundle.shop_seq, bundle.dlv_seq]
        for bundle in all_bundles
    ]

    # ------------- End of custom algorithm code--------------#

    return solution
