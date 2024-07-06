import numpy as np
from sklearn.cluster import KMeans
from util import *
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

def sum_matrix(K, dist_mat, alpha=1, beta=1, gamma=1, ret_others=False):
    pickups = dist_mat[:K, :K]
    delvs = dist_mat[K:, K:]
    pickup_to_delvs = dist_mat[:K, K:]
    delv_to_pickups = dist_mat[K:, :K]
    sum_mat = alpha * pickups + beta * delvs + gamma * (pickup_to_delvs + delv_to_pickups) / 2
    if ret_others:
        return sum_mat, pickups, delvs, pickup_to_delvs, delv_to_pickups
    else:
        return sum_mat

def make_visit_sequence_shop(bundle, shop_dist_mat):
    order_ids = [order.id for order in bundle]
    shop_dist_mat = shop_dist_mat[order_ids, :][:, order_ids]
    tsp_tour = ortools_inference(shop_dist_mat)
    sequence = [order_ids[i] for i in tsp_tour]
    return sequence
        
def make_visit_sequence_delv(bundle, delv_dist_mat):
    order_ids = [order.id for order in bundle]
    delv_dist_mat = delv_dist_mat[order_ids, :][:, order_ids]
    tsp_tour = ortools_inference(delv_dist_mat)
    sequence = [order_ids[i] for i in tsp_tour]
    return sequence

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

def initial_bundles_with_kmeans(K, dist_mat, all_orders, num_clusters):
    sum_mat = sum_matrix(K, dist_mat)
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

def try_bundle_splitting(all_orders, bundle, dist_mat, K):
    if bundle.total_volume > bundle.rider.capa:
        sum_mat, pickups, delvs, _, _ = sum_matrix(K, dist_mat, ret_others=True)
        
        rider = bundle.rider
        order_ids = [order_id for order_id in bundle.dlv_seq]
        mat = sum_mat[order_ids, :][:, order_ids]

        num_clusters = 2
        cluster = KMeans(n_clusters=num_clusters)
        labels = cluster.fit_predict(mat)

        bundle1, bundle2 = [], []
        for i, order_id in enumerate(bundle.dlv_seq):
            order = all_orders[order_id]
            if labels[i] == 0:
                bundle1.append(order)
            else:
                bundle2.append(order)
    
        new_bundle1 = Bundle(all_orders, rider,
                            make_visit_sequence_shop(bundle1, pickups),
                            make_visit_sequence_delv(bundle1, delvs),
                            sum(order.volume for order in bundle1),
                            calculate_bundle_dist(dist_mat, bundle1))
    
        new_bundle2 = Bundle(all_orders, rider,
                            make_visit_sequence_shop(bundle2, pickups),
                            make_visit_sequence_delv(bundle2, delvs),
                            sum(order.volume for order in bundle2),
                            calculate_bundle_dist(dist_mat, bundle2))

        new_bundles1 = try_bundle_splitting(all_orders, new_bundle1, dist_mat, K)
        new_bundles2 = try_bundle_splitting(all_orders, new_bundle2, dist_mat, K)
        return new_bundles1 + new_bundles2
    else:
        return [bundle]
           
    

def ortools_inference(dist_mat):
    data = {
        'distance_matrix': dist_mat.astype(int),
        'num_vehicles': 1,
        'depot': 0,
    }
    manager = pywrapcp.RoutingIndexManager(
        len(data['distance_matrix']), data['num_vehicles'], data['depot']
    )
    # Set up routing model
    routing = pywrapcp.RoutingModel(manager)
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    # Set up search parameter first solution strategy
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    ## First Solution Strategy
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    ## Local Search Metaheuristic
    # search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    ## Time Limit
    # search_parameters.time_limit.seconds = int(self.model_config['time_limit'])
    # Solve!
    solution = routing.SolveWithParameters(search_parameters)

    # Get tour
    index = routing.Start(0)
    route_distance = 0
    tour = [0]
    while not routing.IsEnd(index):
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
        tour.append(index)
    del tour[-1]
    return tour