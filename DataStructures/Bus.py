import pandas as pd
import math


class BusTrip:
    def __init__(self, trip_id, stop_data, capacity):
        self.trip_id = trip_id
        self.capacity = capacity
        self.occupancy_count = 0
        self.stops = stop_data
        self.boardings = {stop: 0 for stop in self.stops}
        self.alightings = {stop: 0 for stop in self.stops}
        self.occupancy = {stop: 0 for stop in self.stops}

    def update_occupancy(self, stop_id, boarding, alighting):
        self.occupancy = max(0, self.occupancy + boarding - alighting)
        if self.occupancy > self.capacity:
            print(f"Warning: Overcapacity at stop {stop_id}")
            self.occupancy = self.capacity

    def process_trip(self):
        for _, stop in self.stops.iterrows():
            # Simulate or retrieve the number of boarding and alighting passengers
            boarding, alighting = self.get_passenger_count(stop["stop_id"])
            self.update_occupancy(boarding, alighting, stop["stop_id"])

    def recalc_occupancy(self):
        occupancy_count = sum(self.passenger_stops)
        return occupancy_count

    def passengers_boarding(self, stop_id, count):
        occupancy_count = self.occupancy_count[stop_id]
        excess = (occupancy_count + count) - self.capacity
        can_accomodate = count - excess
        if can_accomodate > 0:
            print(f"Trip: {self.trip_id}, Excess: {excess}")

            self.occupancy_count += can_accomodate
            self.passenger_stops[stop_id] += can_accomodate
        self.occupancy_count = occupancy_count
        self.boardings[stop_id] += can_accomodate
        self.occupancy[stop_id] = occupancy_count

    def get_passenger_count(self, stop_id):
        return (self.occupancy[stop_id], self.boardings[stop_id], self.alightings[stop_id])
