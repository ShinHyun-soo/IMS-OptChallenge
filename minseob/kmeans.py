import time
import numpy as np
from util import *
from my_util import *

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
            shop_seq=make_visit_sequence_shop(bundle_orders, dist_mat[:K, :K]),
            dlv_seq=make_visit_sequence_delv(bundle_orders, dist_mat[K:, K:]),
            # shop_seq=[order.id for order in bundle_orders],
            # dlv_seq=[order.id for order in bundle_orders],
            total_volume=sum(order.volume for order in bundle_orders),
            total_dist=calculate_bundle_dist(dist_mat, bundle_orders)
        )
        new_bundles = try_bundle_splitting(all_orders, new_bundle, dist_mat, K)
        if len(new_bundles) == 1:
            new_bundle = new_bundles[0]
            all_bundles.append(new_bundle)
        else:
            all_bundles.extend(new_bundles)
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
