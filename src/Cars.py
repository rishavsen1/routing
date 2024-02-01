class Car:
    def __init__(self, id, type, origin, destination, departure_time, length):
        self.id = id
        self.type = type
        self.origin = origin
        self.destination = destination
        self.departure_time = departure_time
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
