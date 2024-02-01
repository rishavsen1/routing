class Bus:
    def __init__(self, trip_id, type, departure_time, end_time, shape_id, length, passengers=0):
        self.id = trip_id
        self.type = type
        self.departure_time = departure_time
        self.end_time = end_time
        self.shape_id = shape_id
        self.length = length
        self.travel_distance = 0
        self.travel_time = 0
        self.passengers = 0
        self.route = []

    def add_to_route(self, node):
        self.route.append(node)

    def update_travel_stats(self, distance, time):
        self.travel_distance += distance
        self.travel_time += time
