from sklearn.cluster import KMeans
import time
import numpy as np
from util import *
from pprint import pprint
import gurobipy as gp
from gurobipy import GRB
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

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
    
    
def algorithm(K, all_orders, all_riders, dist_mat, timelimit=60):
    start_time = time.time()

    for r in all_riders:
        r.T = np.round(dist_mat / r.speed + r.service_time)

    # A solution is a list of bundles
    solution = []

    # ------------- Custom algorithm code starts from here --------------#

    num_clusters = 8  # 라이더 수만큼 군집 생성
    initial_bundles = initial_bundles_with_kmeans(K, dist_mat, all_orders, num_clusters)
    
    # Assign car rider to all bundles as default
    car_rider = None
    for r in all_riders:
        if r.type == 'CAR':
            car_rider = r

    all_bundles = []
    for bundle_orders in initial_bundles:
        new_bundle = Bundle(
            all_orders=all_orders,
            rider=car_rider,
            shop_seq=make_visit_sequence_shop(bundle_orders, dist_mat[:K, :K]),  # TODO: implement!
            dlv_seq=make_visit_sequence_delv(bundle_orders, dist_mat[K:, K:]),
            # shop_seq=[order.id for order in bundle_orders],
            # dlv_seq=[order.id for order in bundle_orders],
            total_volume=sum(order.volume for order in bundle_orders),
            total_dist=calculate_bundle_dist(dist_mat, bundle_orders)
        )
        all_bundles.append(new_bundle)
        for r in all_riders:
            if r.type == 'CAR':
                r.available_number -= 1
    # pprint(all_bundles)
    

    best_obj = sum((bundle.cost for bundle in all_bundles)) / K
    print(f'Best obj = {best_obj}')

    for bundle in all_bundles:
        new_rider = get_cheaper_available_riders(all_riders, bundle.rider)
        if new_rider is not None:
            old_rider = bundle.rider
            if try_bundle_rider_changing(all_orders, bundle, new_rider):
                old_rider.available_number += 1
                new_rider.available_number -= 1

            if time.time() - start_time > timelimit:
                break

        cur_obj = sum((bundle.cost for bundle in all_bundles)) / K
        if cur_obj < best_obj:
            best_obj = cur_obj
            print(f'Best obj = {best_obj}')

    # Solution is a list of bundle information
    solution = [
        # rider type, shop_seq, dlv_seq
        [bundle.rider.type, bundle.shop_seq, bundle.dlv_seq]
        for bundle in all_bundles
    ]

    # ------------- End of custom algorithm code--------------#

    return solution
