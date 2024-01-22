import pandas as pd
from gtfs_functions import Feed

from DataStructures.Bus import BusTrip

# trips = pd.read_csv("data/CARTA_GTFS2024/trips.txt")
gtfs_path = "data/CARTA_GTFS2024.zip"


feed = Feed(gtfs_path)
trips = feed.trips
stops = feed.stops
stop_times = feed.stop_times

for 