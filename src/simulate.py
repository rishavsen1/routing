import osmnx as ox
import networkx as nx
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Set a non-interactive backend
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool

from Vehicles import Vehicle


def plot_graph_with_nodes(graph, node_list, node_color="red"):
    # Plot the graph
    fig, ax = ox.plot_graph(graph, show=False, close=False)

    # Highlight the nodes
    for node in node_list:
        if node in graph.nodes:
            point = graph.nodes[node]
            ax.scatter(point["x"], point["y"], c=node_color, s=100)
        else:
            print(f"Node {node} not found in the graph.")

    plt.show()


def get_processed_network(area_name, filepath):
    graph = ox.graph_from_place(area_name, network_type="drive")
    graph = ox.add_edge_speeds(graph)
    road_network = ox.utils_graph.get_largest_component(graph, strongly=True)

    fig, ax = ox.plot_graph(road_network, show=False, close=True)
    fig.savefig(filepath)

    return graph, road_network


def get_nearest_node(graph, lat, lon):
    return ox.distance.nearest_nodes(graph, lon, lat)


def calculate_shortest_path(graph, origin_lat, origin_lon, destination_lat, destination_lon):
    origin_node = get_nearest_node(graph, origin_lat, origin_lon)
    destination_node = get_nearest_node(graph, destination_lat, destination_lon)
    shortest_path = None
    try:
        shortest_path = nx.shortest_path(graph, origin_node, destination_node, weight="length")
    except:
        return None
    return shortest_path


# def assign_vehicles(graph, road_network, od_pairs):
#     vehicles = []
#     vehicle_id = 0
#     for idx, (origin_lat, origin_lon, destination_lat, destination_lon, departure_time, num_vehicles) in enumerate(
#         od_pairs
#     ):
#         num_vehicles = int(num_vehicles)
#         for _ in range(num_vehicles):
#             vehicle = Vehicle(
#                 vehicle_id, "car", (origin_lat, origin_lon), (destination_lat, destination_lon), departure_time
#             )
#             vehicles.append(vehicle)
#             vehicle_id += 1
#             path = calculate_shortest_path(graph, origin_lat, origin_lon, destination_lat, destination_lon)
#             if path:
#                 for node in path:
#                     vehicle.add_to_route(node)
#         print(f"Trip {idx} complete")
#     return vehicles


def process_od_pair(args):
    graph, od_pair, vehicle_id = args
    origin_lat, origin_lon, destination_lat, destination_lon, departure_time, num_vehicles = od_pair
    vehicles = []
    for _ in range(int(num_vehicles)):
        vehicle = Vehicle(
            vehicle_id, "car", (origin_lat, origin_lon), (destination_lat, destination_lon), departure_time
        )
        vehicle_id += 1
        path = calculate_shortest_path(graph, origin_lat, origin_lon, destination_lat, destination_lon)
        if path:
            for node in path:
                vehicle.add_to_route(node)
        vehicles.append(vehicle)
    return vehicles


def assign_vehicles_parallel(graph, road_network, od_pairs, num_processes=4):
    vehicle_id = 0
    all_vehicles = []
    with Pool(num_processes) as pool:
        results = pool.map(process_od_pair, [(graph, od_pair, vehicle_id + i) for i, od_pair in enumerate(od_pairs)])
        for vehicle_list in results:
            all_vehicles.extend(vehicle_list)
            vehicle_id += len(vehicle_list)
    return all_vehicles


def simulate_traffic(road_network, vehicles, total_simulation_time):
    vehicles_iterator = vehicles.copy()
    for current_time in range(total_simulation_time):
        for vehicle in vehicles_iterator:
            if vehicle.departure_time <= current_time:
                distance = 0
                travel_time = 0
                for i in range(len(vehicle.route) - 1):
                    start, end = vehicle.route[i], vehicle.route[i + 1]
                    # Assuming each edge between 'start' and 'end' might have different data
                    for _, _, edge_data in road_network.edges([start, end], data=True):
                        edge_length = edge_data["length"]  # Length in meters
                        speed_limit = edge_data.get(
                            "speed_kph", 20 * 3.6
                        )  # Default speed limit (convert from m/s to km/h if needed)
                        edge_time = edge_length / (speed_limit / 3.6)  # Convert speed to m/s for calculation
                        distance += edge_length
                        travel_time += edge_time
                vehicle.update_travel_stats(distance, travel_time)
                vehicles_iterator.remove(vehicle)


def analyze_vehicle_data(vehicles):
    for vehicle in vehicles:
        print(
            f"Vehicle {vehicle.vehicle_id} of type {vehicle.vehicle_type} traveled {vehicle.travel_distance} meters in {vehicle.travel_time} seconds at {vehicle.departure_time}."
        )
    vehicle_data = [
        {
            "Vehicle ID": vehicle.vehicle_id,
            "Type": vehicle.vehicle_type,
            "Travel Distance (meters)": vehicle.travel_distance,
            "Travel Time (seconds)": vehicle.travel_time,
            "Departure Time": vehicle.departure_time,
        }
        for vehicle in vehicles
    ]

    # Convert the list of dictionaries into a DataFrame
    vehicle_df = pd.DataFrame(vehicle_data)
    return vehicle_df


def plot_and_save_vehicle_route(vehicle, road_network):
    fig, ax = plt.subplots()
    ox.plot_graph_route(road_network, vehicle.route, ax=ax, show=False, close=True)
    plt.savefig(f"./tmp/car_{vehicle.vehicle_id}.png")
    plt.close(fig)


def visualize_network(road_network, vehicles):
    with Pool(processes=os.cpu_count()) as pool:
        pool.starmap(plot_and_save_vehicle_route, [(vehicle, road_network) for vehicle in vehicles])

    routes = [vehicle.route for vehicle in vehicles]
    fig, ax = ox.plot_graph_routes(road_network, routes, show=False, close=True)
    plt.savefig(f"./tmp/all_cars.png")
    # plt.close(fig)


def load_od_pairs(path):
    od_pairs_df = pd.read_csv(path)[:5]
    od_pairs_df["num_vehicles"] = 1
    od_pairs_df = od_pairs_df.groupby(
        ["origin_loc_lat", "origin_loc_lon", "dest_loc_lat", "dest_loc_lon", "pickup_time_0_secs"]
    ).sum()
    od_pairs_df = od_pairs_df.sort_values("num_vehicles").reset_index()
    od_pairs = od_pairs_df[
        ["origin_loc_lat", "origin_loc_lon", "dest_loc_lat", "dest_loc_lon", "pickup_time_0_secs", "num_vehicles"]
    ].values.tolist()
    return od_pairs


def main():
    area_name = "Hamilton County, Tennessee, USA"
    os.makedirs("./tmp", exist_ok=True)
    graph_save_filepath = f"./tmp/graph_{area_name}.png"
    graph, road_network = get_processed_network(area_name, graph_save_filepath)

    od_path = "/home/rishav/Programs/routing/data/lodes_2021-01-05.csv"
    od_pairs = load_od_pairs(od_path)

    total_simulation_time = 24 * 60 * 60
    vehicles = assign_vehicles_parallel(graph, road_network, od_pairs, num_processes=os.cpu_count())
    simulate_traffic(road_network, vehicles, total_simulation_time)
    vehicle_df = analyze_vehicle_data(vehicles)
    visualize_network(road_network, vehicles)

    vehicle_df.to_csv("./tmp/vehicles.csv")


if __name__ == "__main__":
    main()
