import osmnx as ox
import networkx as nx
import pandas as pd
import matplotlib
from tqdm import tqdm
import math
import numpy as np
from shapely.geometry import LineString
from datetime import datetime, time
import argparse

pd.options.mode.chained_assignment = None  # default='warn'

# matplotlib.use("Agg")  # Set a non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from multiprocessing import Pool

from Buses import Bus
from Cars import Car


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
    plt.close()

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
#     for idx, (origin_lat, origin_lon, destination_lat, destination_lon, departure_time, num_cars) in enumerate(
#         od_pairs
#     ):
#         num_cars = int(num_cars)
#         for _ in range(num_cars):
#             vehicle = Bus(
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

    origin_lat, origin_lon, destination_lat, destination_lon, departure_time, num_cars = od_pair
    vehicles = []
    for _ in range(int(num_cars)):
        vehicle = Car(
            f"car_{vehicle_id}",
            "car",
            (origin_lat, origin_lon),
            (destination_lat, destination_lon),
            int(departure_time),
            5,
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
    for vehicle in vehicles:
        if vehicle.departure_time <= total_simulation_time:
            distance = 0
            travel_time = 0
            travel_times = []
            current_time = int(vehicle.departure_time)
            vehicle.travel_time = 0
            vehicle.travel_distance = 0
            for i in range(len(vehicle.route) - 1):
                if current_time < 86400:
                    start, end = vehicle.route[i], vehicle.route[i + 1]
                    edge_data = list(road_network.edges([start, end], data=True))
                    if edge_data:
                        edge_data = list(road_network.edges([start, end], data=True))[0][2]
                        edge_length = edge_data["length"]  # Length in meters
                        edge_data["vehicle_count"][current_time] += 1

                        # Calculate congestion and adjust speed (speeds converted to m/s)
                        vehicle_count = edge_data["vehicle_count"][current_time]
                        congestion_level = calculate_congestion_level(
                            vehicle_count, edge_length, vehicle.length, lanes=2
                        )
                        adjusted_speed = adjust_speed_for_congestion(
                            edge_data.get("speed_kph", 20) / 3.6, congestion_level
                        )
                        edge_time = edge_length / adjusted_speed
                        travel_times.append(edge_time)

                        # If the vehicle will finish this segment in this timestep, decrement the vehicle count
                        new_time = round(current_time + edge_time)
                        if new_time < total_simulation_time:
                            edge_data["vehicle_count"][new_time] -= 1

                        current_time = new_time
                        vehicle.travel_time += edge_time
                        vehicle.travel_distance += edge_length

                # else:
                #     print(f"Edge between {start} - {end} not found")
                #     # vehicle.route.remove(start)
                #     # vehicles.remove(vehicle)
                #     flag = False
                #     break

            if len(vehicle.route) - 1 == len(travel_times):
                vehicle.update_travel_stats(distance, travel_time)
                travel_times_rounded = [int(round(time)) for time in travel_times]
                update_road_usage(
                    road_network, vehicle.route, vehicle.departure_time, travel_times_rounded, vehicle.type
                )
            else:
                vehicles.remove(vehicle)
                print("Mismatch in route and travel times lengths")
    return travel_times


def analyze_vehicle_data(vehicles):
    vehicle_data = [
        {
            "Vehicle ID": vehicle.id,
            "Type": vehicle.type,
            "Travel Distance (meters)": vehicle.travel_distance,
            "Travel Time (seconds)": vehicle.travel_time,
            "Departure Time": vehicle.departure_time,
        }
        for vehicle in vehicles
    ]

    vehicle_df = pd.DataFrame(vehicle_data)
    return vehicle_df


def plot_and_save_vehicle_route(vehicle, road_network, output_path):
    fig, ax = plt.subplots()
    ox.plot_graph_route(road_network, vehicle.route, ax=ax, show=False, close=True)
    plt.savefig(f"{output_path}/car_{vehicle.vehicle_id}.png")
    plt.close(fig)


def calculate_segment_travel_times(vehicle, road_network):
    travel_times = []
    for i in range(len(vehicle.route) - 1):
        start, end = vehicle.route[i], vehicle.route[i + 1]
        try:
            edge_data = road_network[start][end][0]  # Assuming single key
            edge_length = edge_data["length"]  # Length in meters
            speed_limit = edge_data.get("speed_kph", 20) * 1000 / 3600  # Convert to m/s
            edge_time = edge_length / speed_limit
            travel_times.append(edge_time)
        except KeyError:
            print(f"KeyError for edge: {start}-{end}. Edge might not exist in the graph.")
    assert len(vehicle.route) - 1 == len(travel_times), "Mismatch in route and travel times lengths"
    return travel_times


def visualize_network(road_network, vehicles, output_path):
    # with Pool(processes=os.cpu_count()) as pool:
    #     pool.starmap(plot_and_save_vehicle_route, [(vehicle, road_network) for vehicle in vehicles])

    routes = [vehicle.route for vehicle in vehicles]
    fig, ax = ox.plot_graph_routes(road_network, routes, show=False, close=True)
    plt.savefig(f"{output_path}/all_cars.png")
    plt.close(fig)


def initialize_road_usage(road_network, total_simulation_time):
    all_times = {time: 0 for time in range(total_simulation_time)}
    for u, v, key in road_network.edges(keys=True):
        road_network[u][v][key]["vehicle_count"] = all_times


def update_road_usage(road_network, vehicle_route, departure_time, travel_times, vehicle_type):
    current_time = departure_time
    for i in range(len(vehicle_route) - 1):
        start, end = vehicle_route[i], vehicle_route[i + 1]
        try:
            road_network[start][end][0]["vehicle_count"][current_time] += 1
        except KeyError:
            print(f"KeyError for edge: {start}-{end}. Edge might not exist in the graph. type: {vehicle_type}")
        current_time += int(travel_times[i])  # Increment time by the travel time for this segment


def adjust_speed_for_congestion(speed_limit, congestion_level):
    """
    Adjusts the speed of vehicles based on the congestion level of the road segment.

    :param speed_limit: The speed limit of the road segment (in km/h or m/s).
    :param congestion_level: The congestion level on the road segment (a value between 0 and 1).
    :return: The adjusted speed for the vehicles on that road segment.
    """
    # Simple linear congestion model:
    # If congestion is 0, speed is at the speed limit.
    # If congestion is 1 (maximum), speed is reduced to a certain percentage of the speed limit.
    # Feel free to adjust the factor or use a more complex model based on actual traffic data.
    MINIMUM_SPEED_FACTOR = 0.3  # At maximum congestion, speed doesn't fall below 30% of the speed limit.
    adjusted_speed = speed_limit * max(MINIMUM_SPEED_FACTOR, 1 - congestion_level)

    return adjusted_speed


# def adjust_speed_for_congestion(speed_limit, congestion_level):
#     # Simple model: linear decrease in speed with congestion
#     # At congestion level 1, speed is reduced to 50% of the speed limit
#     # Feel free to use a more sophisticated model
#     return speed_limit * (1 - 0.5 * congestion_level)


def calculate_congestion_level(vehicle_count, road_length, vehicle_length, lanes):
    lane_capacity = road_length / vehicle_length * lanes  # simplistic capacity model
    congestion_level = min(1, vehicle_count / lane_capacity)  # congestion level, capped at 1
    return congestion_level


def calculate_congestion_for_edge(edge_data, total_simulation_time, lanes=2):
    u, v, key, data = edge_data
    # car avg length = 5m, bus avg length = 12m, micro-transit avg length = 9m
    osm_capacity = data.get("capacity", None)
    avg_length_of_vehicle = 5 * 0.9 + 12 * 0.05 + 9 * 0.05
    capacity = data.get("length") * lanes / avg_length_of_vehicle
    if osm_capacity:
        capacity = osm_capacity

    if capacity > 0:
        vehicle_counts = np.array([data["vehicle_count"][time_step] for time_step in range(total_simulation_time)])
        congestion_levels = vehicle_counts / capacity
        avg_congestion = np.mean(congestion_levels)
    else:
        avg_congestion = 0.5

    return (u, v), avg_congestion


def calculate_congestion(road_network, total_simulation_time):
    with Pool(processes=os.cpu_count()) as pool:
        edge_data_list = list(road_network.edges(keys=True, data=True))
        results = pool.starmap(
            calculate_congestion_for_edge, [(edge_data, total_simulation_time) for edge_data in edge_data_list]
        )

    congestion_data = dict(results)
    return congestion_data


def assign_edge_colors(road_network, congestion_data):
    cmap = plt.colormaps["RdYlGn_r"]  # Use reversed Red-Yellow-Green colormap
    norm = mcolors.Normalize(vmin=0, vmax=1)

    for u, v, key in road_network.edges(keys=True):
        congestion_level = congestion_data.get((u, v), 0)
        color = cmap(norm(congestion_level))
        road_network[u][v][key]["color"] = color


def get_edge_colors(road_network):
    edge_colors = []
    for u, v, key, data in road_network.edges(keys=True, data=True):
        edge_color = data.get("color", "gray")  # Default color if not set
        edge_colors.append(edge_color)
    return edge_colors


def visualize_congestion(road_network, congestion_data):
    assign_edge_colors(road_network, congestion_data)
    edge_colors = get_edge_colors(road_network)
    fig, ax = ox.plot_graph(road_network, edge_color=edge_colors, node_size=0)
    plt.show()

    return fig


def load_od_pairs(od_pairs_df):
    od_pairs_df.loc[:, "num_cars"] = 1
    od_pairs_df = od_pairs_df.groupby(
        ["origin_loc_lat", "origin_loc_lon", "dest_loc_lat", "dest_loc_lon", "pickup_time_0_secs"]
    ).sum()
    od_pairs_df = od_pairs_df.sort_values("num_cars").reset_index()
    od_pairs = od_pairs_df[
        ["origin_loc_lat", "origin_loc_lon", "dest_loc_lat", "dest_loc_lon", "pickup_time_0_secs", "num_cars"]
    ].values.tolist()
    return od_pairs


def parse_shapes_to_linestrings(shapes_df):
    # Group the shapes by shape_id and create a LineString for each unique shape
    grouped = shapes_df.groupby("shape_id")
    linestrings = {shape_id: LineString(group[["shape_pt_lon", "shape_pt_lat"]].values) for shape_id, group in grouped}
    return linestrings


def map_single_linestring_to_osm(graph, linestring_data):
    shape_id, linestring = linestring_data
    points = list(linestring.coords)
    osm_nodes = [ox.distance.nearest_nodes(graph, lon, lat) for lon, lat in points]

    # Validate and connect nodes
    connected_nodes = []
    for i in range(len(osm_nodes) - 1):
        start_node = osm_nodes[i]
        end_node = osm_nodes[i + 1]
        if start_node == end_node or graph.has_edge(start_node, end_node):
            connected_nodes.append(start_node)
        else:
            # Find the shortest path between start_node and end_node
            try:
                path = nx.shortest_path(graph, start_node, end_node, weight="length")
                connected_nodes.extend(path[:-1])  # Exclude the end_node to avoid duplicates
            except nx.NetworkXNoPath:
                print(f"No path between {start_node} and {end_node}")
    connected_nodes.append(osm_nodes[-1])  # Add the last node
    return shape_id, connected_nodes


def parallel_map_linestring_to_osm(graph, linestrings):
    osm_paths = {}
    # graph = (graph,)  # Tuple containing the graph
    linestrings_data = list(linestrings.items())

    with Pool(processes=int(os.cpu_count() / 2)) as pool:
        # Map linestrings to OSM nodes in parallel
        results = pool.starmap(
            map_single_linestring_to_osm, [(graph, linestring_data) for linestring_data in linestrings_data]
        )

    # Convert results to dictionary
    for shape_id, osm_nodes in results:
        osm_paths[shape_id] = osm_nodes

    return osm_paths


def load_gtfs(trips_df, path):
    stop_times_df = pd.read_csv(f"{path}/stop_times.txt")
    shapes_df = pd.read_csv(f"{path}/shapes.txt")

    trips = trips_df["trip_id"].to_list()
    stop_times_df = stop_times_df[stop_times_df["trip_id"].isin(trips)]

    stop_times_df_copy = stop_times_df.copy()
    stop_times_df_copy["arrival_time"] = pd.to_timedelta(stop_times_df_copy["arrival_time"])
    stop_times_df_copy["arrival_time_secs"] = stop_times_df_copy["arrival_time"].dt.total_seconds()
    stop_times_df_sorted = stop_times_df_copy.sort_values(by=["trip_id", "arrival_time"])

    trip_travel_times = (
        stop_times_df_sorted.groupby("trip_id")
        .agg(start_time=("arrival_time_secs", "first"), end_time=("arrival_time_secs", "last"))
        .reset_index()
    )
    trip_travel_times["trip_gtfs_duration"] = trip_travel_times["end_time"] - trip_travel_times["start_time"]

    return (
        trip_travel_times,
        stop_times_df,
        shapes_df,
    )


def create_buses_from_gtfs(trips_df, stop_times_df, shapes_df, osm_paths):
    buses = []

    # Iterate over trips
    for _, trip in trips_df.iterrows():
        trip_id = trip["trip_id"]
        shape_id = trip["shape_id"]

        trip_times = stop_times_df[stop_times_df["trip_id"] == trip_id]
        trip_times = trip_times[trip_times["arrival_time"] < "24:00:00"]
        trip_times = trip_times[trip_times["departure_time"] < "24:00:00"]
        trip_times["arrival_time"] = pd.to_datetime(trip_times["arrival_time"], format="%H:%M:%S")
        trip_times["departure_time"] = pd.to_datetime(trip_times["departure_time"], format="%H:%M:%S")

        start_time = trip_times["arrival_time"].min()
        end_time = trip_times["departure_time"].max()

        # Convert to seconds
        start_time_in_seconds = (start_time.hour * 3600) + (start_time.minute * 60) + start_time.second
        end_time_in_seconds = (end_time.hour * 3600) + (end_time.minute * 60) + end_time.second
        length = 12  # in meters
        bus = Bus(f"bus_{trip_id}", "bus", start_time_in_seconds, end_time_in_seconds, shape_id, length)

        bus_route_nodes = osm_paths.get(shape_id, [])
        for node in bus_route_nodes:
            bus.add_to_route(node)
        buses.append(bus)

    return buses


def estimating_bus_micro_usage(gtfs_path, od_with_tt_path, income_path, output_path, use_bus, use_micro=False):
    income_df = pd.read_csv(income_path, index_col=0).drop(["geometry"], axis=1)
    income_df["estimate"] = income_df["estimate"].astype(float)

    od_df = pd.read_csv(od_with_tt_path, index_col=0, low_memory=False)
    od_df = od_df[od_df["transit_time"] > od_df["drive_time"]]
    od_df = od_df[od_df["walk_time"] > od_df["drive_time"]]
    od_df = od_df[od_df["walk_time"] >= od_df["transit_time"]]

    data = pd.merge(od_df, income_df, left_on="h_geocode", right_on="GEOID", how="left")
    data = data.dropna(subset="estimate")

    max_income = income_df["estimate"].max()
    data["income_ratio"] = data["estimate"] / max_income

    data["normalized_drive_time"] = data["drive_time"] / data["drive_time"].max()
    data["normalized_transit_time"] = data["transit_time"] / data["transit_time"].max()
    data["transit_time_ratio"] = data["drive_time"] / data["transit_time"]

    high_penalty = 10
    data["normalized_transit_time"].fillna(high_penalty, inplace=True)

    income_weight = 0.5
    transit_time_ratio_weight = 0.5

    data["bus_usage_probability"] = (
        income_weight * data["income_ratio"] + transit_time_ratio_weight * data["transit_time_ratio"]
    )

    data["bus_usage_probability"] = (data["bus_usage_probability"] - data["bus_usage_probability"].min()) / (
        data["bus_usage_probability"].max() - data["bus_usage_probability"].min()
    )

    od_count_per_cbg = od_df.groupby("h_geocode").size().reset_index(name="total_od_count")
    data = pd.merge(data, od_count_per_cbg, on="h_geocode", how="left")
    data["estimated_bus_users"] = data["bus_usage_probability"] * data["total_od_count"]

    trips_df = pd.read_csv(f"{gtfs_path}/trips.txt")
    stop_times_df = pd.read_csv(f"{gtfs_path}/stop_times.txt")
    stop_times_df = stop_times_df[stop_times_df["departure_time"] < "24:00:00"]

    stop_times_df["arrival_time"] = pd.to_datetime(stop_times_df["arrival_time"])
    stop_times_df["departure_time"] = pd.to_datetime(stop_times_df["departure_time"])
    stop_times_df["arrival_time"] = stop_times_df["arrival_time"].dt.time
    stop_times_df["departure_time"] = stop_times_df["departure_time"].dt.time

    min_od_time = int(min(od_df.pickup_time_0_secs))
    min_hour = int(min_od_time / 3600)
    min_minute = int((min_od_time - min_hour * 3600) / 60)

    max_od_time = int(max(od_df.pickup_time_0_secs)) + 60
    max_hour = int(max_od_time / 3600)
    max_minute = int((max_od_time - max_hour * 3600) / 60)

    start_time = time(hour=min_hour, minute=min_minute)
    end_time = time(hour=max_hour, minute=max_minute)

    relevant_stop_times = stop_times_df[
        (stop_times_df["arrival_time"] >= start_time) & (stop_times_df["departure_time"] <= end_time)
    ]

    relevant_trips = trips_df.loc[trips_df["trip_id"].isin(list(relevant_stop_times.trip_id.unique()))]
    relevant_trips.to_csv(f"{output_path}/relevant_bus_trips.csv")

    num_buses = relevant_stop_times["trip_id"].nunique()
    route_buses = relevant_stop_times.merge(trips_df, on="trip_id", how="left")
    num_buses_per_route = route_buses.groupby("route_id")["trip_id"].nunique().reset_index(name="num_buses")

    if use_micro:
        total_ods = len(data)
        target_micro_transit_users = 300
        estimated_percentage_of_disabled_who_use_micro_transit = 0.3

        required_disabled_percentage = target_micro_transit_users / (
            total_ods * estimated_percentage_of_disabled_who_use_micro_transit
        )
        np.random.seed(42)
        data["is_disabled"] = np.random.rand(total_ods) < required_disabled_percentage

    data = pd.merge(data, num_buses_per_route, left_on="route1", right_on="route_id", how="left")
    data["num_buses"].fillna(0, inplace=True)
    data["bus_capacity"] = 32 * 0.75
    data["max_bus_capacity"] = data["bus_capacity"] * data["num_buses"]
    data["random_number"] = np.random.rand(len(data))

    if use_bus:
        data["choice"] = np.where(data["random_number"] < data["bus_usage_probability"], "Bus", "Car")

    if use_micro:
        # data["choice"] = np.where(data["use__transit_eligible"], "Micro-Transit", data["choice"])
        data["micro_transit_eligible"] = np.where(data["is_disabled"], np.random.rand(len(data)) < 0.3, False)
        data["choice"] = np.where(data["micro_transit_eligible"], "Micro-Transit", data["choice"])
        data[data["choice"] == "Micro-Transit"].to_csv(f"{output_path}/micro-transit_users.csv")

    if use_bus or use_micro:
        bus_user_counts = data.groupby(["route_id", "choice"]).size().unstack(fill_value=0)

        for route_id, row in bus_user_counts.iterrows():
            if row["Bus"] > data.loc[data["route_id"] == route_id, "max_bus_capacity"].values[0]:
                excess = row["Bus"] - data.loc[data["route_id"] == route_id, "max_bus_capacity"].values[0]
                bus_users = data[(data["route_id"] == route_id) & (data["choice"] == "Bus")]
                drop_indices = bus_users.sample(n=int(excess)).index
                data.loc[drop_indices, "choice"] = "Car"

        data[data["choice"] == "Bus"].to_csv(f"{output_path}/bus_users.csv")
        data[data["choice"] == "Car"].to_csv(f"{output_path}/car_users.csv")

    else:
        data["choice"] = "Car"
        data.to_csv(f"{output_path}/car_users.csv")

    return data, relevant_trips


def process_micro_bus_route(args):
    graph, route_data, micro_bus_id = args
    micro_bus_route = []
    previous_lat, previous_lon = None, None

    for _, row in route_data.iterrows():
        current_lat, current_lon = row["origin_loc_lat"], row["origin_loc_lon"]

        if previous_lat is not None and previous_lon is not None:
            # Calculate shortest path from previous stop to current stop
            path = calculate_shortest_path(graph, previous_lat, previous_lon, current_lat, current_lon)
            if path:
                # If path is not None, add the path to the micro_bus_route
                micro_bus_route.extend(path[:-1])  # Exclude the last node to avoid duplication

        # Get the node for the current stop and add it to the route
        current_node = ox.distance.nearest_nodes(graph, current_lon, current_lat)
        micro_bus_route.append(current_node)
        previous_lat, previous_lon = (
            current_lat,
            current_lon,
        )  # Set the current stop as the previous stop for the next iteration

    return micro_bus_id, micro_bus_route


def parallel_process_micro_bus_route(graph, manifest_df, df_requests_path):
    df_requests = pd.read_csv(df_requests_path)
    merged_data = pd.merge(manifest_df, df_requests, left_on="booking_id", right_on="booking_id", how="left")
    sorted_data = merged_data.sort_values(by=["run_id", "scheduled_time"])

    # Process each micro-bus route in parallel
    micro_bus_routes_data = [
        (graph, route_data, micro_bus_id) for micro_bus_id, route_data in sorted_data.groupby("run_id")
    ]

    with Pool(processes=int(os.cpu_count() / 2)) as pool:
        results = pool.map(process_micro_bus_route, micro_bus_routes_data)

    # Convert results to a dictionary {micro_bus_id: micro_bus_route}
    micro_bus_routes = dict(results)
    return micro_bus_routes


def create_micro_transit_buses(manifest_df, micro_bus_routes):
    micro_transit_buses = []

    # Iterate over micro-bus routes
    for micro_bus_id, micro_bus_route in micro_bus_routes.items():
        # Extract relevant information from manifest_df
        micro_bus_manifest = manifest_df[manifest_df["run_id"] == micro_bus_id]
        start_time = micro_bus_manifest["scheduled_time"].min()
        end_time = micro_bus_manifest["scheduled_time"].max()
        length = 9  # Length for micro-transit is 9 meters

        # Initialize micro-transit bus
        micro_bus = Bus(f"micro_bus_{micro_bus_id}", "micro-transit", start_time, end_time, micro_bus_id, length)

        # Add the route to the micro-transit bus
        for node in micro_bus_route:
            micro_bus.add_to_route(node)

        micro_transit_buses.append(micro_bus)

    return micro_transit_buses


def remove_consecutive_duplicates(osm_paths):
    for key in osm_paths:
        current_list = osm_paths[key]
        new_list = [
            current_list[i] for i in range(len(current_list)) if i == 0 or current_list[i] != current_list[i - 1]
        ]
        osm_paths[key] = new_list
    return osm_paths


def calculate_adjustment(row, bus_trip_adjustment_dict, mean_adjustment, bus_gtfs_tt=None):
    adjustments = []
    for i in range(1, 7):
        trip_key = f"trip{i}"
        if pd.notnull(row[trip_key]):
            trip_id_str = str(int(row[trip_key]))
            if (trip_id_str in bus_gtfs_tt) and (trip_id_str in bus_trip_adjustment_dict):
                gtfs_tt = bus_gtfs_tt[trip_id_str]
                tt_to_gtfs_tt = row["transit_time"] / gtfs_tt
                adjustments.append(bus_trip_adjustment_dict[trip_id_str] / tt_to_gtfs_tt)
            else:
                adjustments.append(mean_adjustment)

    avg_adjustment = np.mean(adjustments) if adjustments else 1
    return row["transit_time"] * avg_adjustment


def main():
    total_simulation_time = 24 * 60 * 60
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(description="Process transit options.")

    parser.add_argument("-t", "--using-transit", dest="USING_TRANSIT", action="store_true", help="Flag to use transit")
    parser.add_argument(
        "-m",
        "--using-micro-transit",
        dest="USING_MICRO_TRANSIT",
        action="store_true",
        help="Flag to use micro transit",
    )

    parser.set_defaults(USING_TRANSIT=False, USING_MICRO_TRANSIT=False)

    args = parser.parse_args()

    USING_TRANSIT = args.USING_TRANSIT
    USING_MIRCO_TRANSIT = args.USING_MICRO_TRANSIT
    area_name = "Hamilton County, Tennessee, USA"

    output_path = "./output/" + datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    os.makedirs(f"{output_path}", exist_ok=True)
    graph_save_filepath = f"{output_path}/graph_{area_name}.png"
    graph, road_network = get_processed_network(area_name, graph_save_filepath)
    # graph_hash = add_graph_to_dict(graph)

    od_path = "/home/rishav/Programs/routing/data/lodes_2021-01-05.csv"
    gtfs_path = "/home/rishav/Programs/routing/data/CARTA_GTFS2024"

    # use the
    od_with_tt_path = "/home/rishav/Programs/routing/outputs/lodes_2021-01-05_transit_drive2.csv"
    income_path = "/home/rishav/Programs/routing/data/income_tn_all.csv"
    manifest_path = "/home/rishav/Programs/routing/data/micro_manifests/manifest_df.csv"  # from transit-webapp/notebooks/offline_optimizer.ipynb (pass the micro-transit users csv)
    df_requests_path = "/home/rishav/Programs/routing/data/micro_manifests/df_requests.csv"

    ods, relevant_trips = estimating_bus_micro_usage(
        gtfs_path,
        od_with_tt_path,
        income_path,
        output_path,
        use_bus=USING_TRANSIT,
        use_micro=USING_MIRCO_TRANSIT,
    )

    trip_travel_times, stop_times_df, shapes_df = load_gtfs(relevant_trips, gtfs_path)
    trip_travel_times["trip_id"] = trip_travel_times["trip_id"].astype(str)
    linestrings = parse_shapes_to_linestrings(shapes_df)
    osm_paths = parallel_map_linestring_to_osm(graph, linestrings)
    osm_paths = remove_consecutive_duplicates(osm_paths)
    osm_df = pd.DataFrame(list(osm_paths.items()), columns=["shapes", "osm_nodes"])
    osm_df.to_csv(f"{output_path}/shape_osm_nodes.csv")

    if not USING_MIRCO_TRANSIT:
        ods_using_car = ods[ods["choice"] == "Car"]
    else:
        ods_using_car = ods[ods["choice"] != "Micro-Transit"]

    od_pairs = load_od_pairs(ods_using_car)

    initialize_road_usage(road_network, total_simulation_time)
    vehicles = assign_vehicles_parallel(graph, road_network, od_pairs, num_processes=os.cpu_count())

    if USING_TRANSIT:
        buses = create_buses_from_gtfs(relevant_trips, stop_times_df, shapes_df, osm_paths)
        vehicles.extend(buses)

    if USING_MIRCO_TRANSIT:
        manifest_df = pd.read_csv(manifest_path)
        micro_bus_routes = parallel_process_micro_bus_route(graph, manifest_df, df_requests_path)
        micro_bus_routes = remove_consecutive_duplicates(micro_bus_routes)
        osm_df = pd.DataFrame(list(osm_paths.items()), columns=["shapes", "osm_nodes"])
        osm_df.to_csv(f"{output_path}/micro_transit_nodes.csv")

        micro_transit_buses = create_micro_transit_buses(manifest_df, micro_bus_routes)
        vehicles.extend(micro_transit_buses)

        micro_trip_travel_times = (
            manifest_df.groupby("run_id")
            .agg(start_time=("scheduled_time", "first"), end_time=("scheduled_time", "last"))
            .reset_index()
        )

    # if USING_TRANSIT and USING_MIRCO_TRANSIT:

    #     ods = estimating_bus_micro_usage(
    #         gtfs_path,
    #         od_with_tt_path,
    #         income_path,
    #         output_path,
    #         use_bus=USING_MIRCO_TRANSIT,
    #         use_micro=USING_MIRCO_TRANSIT,
    #     )
    #     ods_using_car = ods[ods["choice"] == "Car"]

    #     od_pairs = load_od_pairs(ods_using_car)

    #     trips_df, stop_times_df, shapes_df = load_gtfs(gtfs_path)
    #     linestrings = parse_shapes_to_linestrings(shapes_df)
    #     osm_paths = parallel_map_linestring_to_osm(graph, linestrings)
    #     buses = create_buses_from_gtfs(trips_df, stop_times_df, shapes_df, osm_paths)

    #     manifest_df = pd.read_csv(manifest_path)
    #     micro_bus_routes = parallel_process_micro_bus_route(graph, manifest_df, df_requests_path)
    #     micro_transit_buses = create_micro_transit_buses(manifest_df, micro_bus_routes)

    #     initialize_road_usage(road_network, total_simulation_time)
    #     vehicles = assign_vehicles_parallel(graph, road_network, od_pairs, num_processes=os.cpu_count())
    #     vehicles.extend(buses)
    #     vehicles.extend(micro_transit_buses)

    # elif USING_TRANSIT:
    #     ods = estimating_bus_micro_usage(
    #         gtfs_path,
    #         od_with_tt_path,
    #         income_path,
    #         output_path,
    #         use_bus=USING_MIRCO_TRANSIT,
    #         use_micro=USING_MIRCO_TRANSIT,
    #     )
    #     ods_using_car = ods[ods["choice"] == "Car"]

    #     od_pairs = load_od_pairs(ods_using_car)

    #     trips_df, stop_times_df, shapes_df = load_gtfs(gtfs_path)
    #     linestrings = parse_shapes_to_linestrings(shapes_df)
    #     osm_paths = parallel_map_linestring_to_osm(graph, linestrings)
    #     buses = create_buses_from_gtfs(trips_df, stop_times_df, shapes_df, osm_paths)

    #     initialize_road_usage(road_network, total_simulation_time)
    #     vehicles = assign_vehicles_parallel(graph, road_network, od_pairs, num_processes=os.cpu_count())
    #     vehicles.extend(buses)

    # elif USING_MIRCO_TRANSIT:
    #     ods = estimating_bus_micro_usage(
    #         gtfs_path,
    #         od_with_tt_path,
    #         income_path,
    #         output_path,
    #         use_bus=USING_MIRCO_TRANSIT,
    #         use_micro=USING_MIRCO_TRANSIT,
    #     )
    #     ods_using_car = ods[ods["choice"] != "Micro-Transit"]

    #     od_pairs = load_od_pairs(ods_using_car)

    #     manifest_df = pd.read_csv(manifest_path)
    #     micro_bus_routes = parallel_process_micro_bus_route(graph, manifest_df, df_requests_path)
    #     micro_transit_buses = create_micro_transit_buses(manifest_df, micro_bus_routes)

    #     initialize_road_usage(road_network, total_simulation_time)
    #     vehicles = assign_vehicles_parallel(graph, road_network, od_pairs, num_processes=os.cpu_count())
    #     vehicles.extend(micro_transit_buses)

    # else:
    #     ods = estimating_bus_micro_usage(
    #         gtfs_path,
    #         od_with_tt_path,
    #         income_path,
    #         output_path,
    #         use_bus=USING_MIRCO_TRANSIT,
    #         use_micro=USING_MIRCO_TRANSIT,
    #     )
    #     ods_using_car = ods

    #     od_pairs = load_od_pairs(ods_using_car)

    #     initialize_road_usage(road_network, total_simulation_time)
    #     vehicles = assign_vehicles_parallel(graph, road_network, od_pairs, num_processes=os.cpu_count())

    travel_times = simulate_traffic(road_network, vehicles, total_simulation_time)
    vehicle_df = analyze_vehicle_data(vehicles)
    vehicle_df["trip_id"] = vehicle_df["Vehicle ID"].apply(lambda x: x.split("_")[1])

    trip_travel_times["trip_id"] = trip_travel_times["trip_id"].astype(int).astype(str)
    merged_trip_travel_times = trip_travel_times.merge(vehicle_df[vehicle_df["Type"] == "bus"])
    merged_trip_travel_times["adjustment_ratio"] = (
        merged_trip_travel_times["Travel Time (seconds)"] / merged_trip_travel_times["trip_gtfs_duration"]
    )

    bus_gtfs_tt = merged_trip_travel_times.set_index("trip_id")["trip_gtfs_duration"].to_dict()

    if USING_TRANSIT:
        vehicle_df["trip_id"] = vehicle_df["Vehicle ID"].apply(lambda x: x.split("_")[1])
        vehicle_df = vehicle_df.merge(trip_travel_times, on="trip_id", how="left")
        vehicle_df["trip_gtfs_duration"] = vehicle_df["trip_gtfs_duration"].fillna(vehicle_df["Travel Time (seconds)"])
        vehicle_df["adjustment_ratio"] = vehicle_df["trip_gtfs_duration"] / vehicle_df["Travel Time (seconds)"]
        vehicle_df.to_csv(f"{output_path}/vehicles.csv")

        bus_trip_adjustment_dict = merged_trip_travel_times.set_index("trip_id")["adjustment_ratio"].to_dict()
        mean_adjustment = np.array(list(bus_trip_adjustment_dict.values())).mean()
        ods["transit_time"] = ods.apply(
            lambda row: calculate_adjustment(row, bus_trip_adjustment_dict, mean_adjustment, bus_gtfs_tt), axis=1
        )

        ods[ods["choice"] == "Bus"].to_csv(f"{output_path}/bus_users.csv")

    if USING_MIRCO_TRANSIT:
        micro_trip_travel_times = micro_trip_travel_times.merge(vehicle_df[vehicle_df["Type"] == "micro-transit"])
        micro_trip_travel_times["adjustment_ratio"] = (
            micro_trip_travel_times["Travel Time (seconds)"] / micro_trip_travel_times["trip_gtfs_duration"]
        )
        micro_trip_adjustment_dict = merged_trip_travel_times.set_index("trip_id")["adjustment_ratio"].to_dict()
        mean_adjustment = np.array(list(micro_trip_adjustment_dict.values())).mean()
        ods["transit_time"] = ods.apply(
            lambda row: calculate_adjustment(row, micro_trip_adjustment_dict, mean_adjustment), axis=1
        )

        ods[ods["choice"] == "Micro-Transit"].to_csv(f"{output_path}/micro_transit_users.csv")

    # for vehicle in vehicles:
    #     travel_times = calculate_segment_travel_times(vehicle, road_network)
    #     update_road_usage(road_network, vehicle.route, vehicle.departure_time, travel_times)

    congestion_data = calculate_congestion(road_network, total_simulation_time)
    fig = visualize_congestion(road_network, congestion_data)
    fig.savefig(f"{output_path}/graph_{area_name}_congestion.png", dpi=300)
    fig.savefig(f"{output_path}/graph_{area_name}_congestion.svg")

    congestion_df = pd.DataFrame.from_dict(congestion_data, orient="index", columns=["congestion_value"])
    congestion_df.reset_index(inplace=True)
    congestion_df[["start", "end"]] = pd.DataFrame(congestion_df["index"].tolist(), index=congestion_df.index)
    congestion_df.drop("index", axis=1, inplace=True)
    congestion_df.columns = ["congestion_value", "start", "end"]
    congestion_df.to_csv(f"{output_path}/congestion.csv")

    vehicles_data = []
    for vehicle in vehicles:
        vehicle_info = {
            "Vehicle ID": vehicle.id,
            "Type": vehicle.type,
            "Route": "->".join(map(str, vehicle.route)),  # Join the route node IDs with '->'
            "Distance Travelled": vehicle.travel_distance,
        }
        vehicles_data.append(vehicle_info)
    vehicles_df = pd.DataFrame(vehicles_data)
    vehicles_df.to_csv(f"{output_path}/vehicles_routes.csv", index=False)


if __name__ == "__main__":
    main()
