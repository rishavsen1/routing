import aiohttp
import asyncio
from tqdm.asyncio import tqdm
import pandas as pd
import os


async def traveltimeTransit(latHome, lonHome, latWork, lonWork, time, semaphore):
    async with semaphore:
        request_url = f"http://localhost:8080/otp/routers/default/plan?fromPlace={latHome},{lonHome}&toPlace={latWork},{lonWork}&mode=WALK,TRANSIT&date=10-09-2023&time={time}&maxWalkDistance=750"

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(request_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        res_dict = {}
                        if "plan" in data.keys() and "itineraries" in data["plan"].keys():
                            res_dict.update({"transit_time": int(data["plan"]["itineraries"][0]["duration"])})
                            res_dict.update({"transfers": int(data["plan"]["itineraries"][0]["transfers"])})
                            legs = pd.DataFrame(data["plan"]["itineraries"][0]["legs"])
                            # df.loc[i, 'trips'] = np.array(legs['tripId'].dropna().to_list())
                            # df.loc[i, 'trips'] = df.apply(lambda r: tuple(legs['tripId'].dropna()), axis=1).apply(np.array)
                            agency_split = legs["agencyId"].dropna().to_list()
                            trip_split = legs["tripId"].dropna().to_list()
                            mode_split = legs["mode"].dropna().to_list()
                            route_split = legs["route"][legs["route"] != ""].to_list()

                            res_dict.update({"mode": mode_split})
                            # res_dict.update({'agency1': agency_split[0]})
                            # res_dict.update({'agency2': None})
                            # res_dict.update({'agency3': None})
                            # if len(agency_split) > 1:
                            #     res_dict.update({'agency2': agency_split[1]})
                            #     if len(agency_split) > 2:
                            #         res_dict.update({'agency3': agency_split[2]})

                            (
                                res_dict["trip1"],
                                res_dict["trip2"],
                                res_dict["trip3"],
                                res_dict["trip4"],
                                res_dict["trip5"],
                                res_dict["trip6"],
                            ) = (None,) * 6
                            for i, trip in enumerate(trip_split, start=1):
                                res_dict[f"trip{i}"] = trip.split(":")[1]

                            (
                                res_dict["route1"],
                                res_dict["route2"],
                                res_dict["route3"],
                                res_dict["route4"],
                                res_dict["route5"],
                                res_dict["route6"],
                            ) = (None,) * 6
                            for i, route in enumerate(route_split, start=1):
                                res_dict[f"route{i}"] = route

                            res_dict.update({"bus_capacity_upto_2hours": len(data["plan"]["itineraries"]) * 35})

                        else:
                            return False

                        return res_dict
                    return False
            except Exception as e:
                print(f"Error during API call: {e}")
                return False


async def traveltimeTransit(latHome, lonHome, latWork, lonWork, time, semaphore):
    async with semaphore:
        request_url = f"http://localhost:8080/otp/routers/default/plan?fromPlace={latHome},{lonHome}&toPlace={latWork},{lonWork}&mode=DRIVE&date=10-09-2023&time={time}&maxWalkDistance=750"

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(request_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        res_dict = {}
                        if "plan" in data.keys() and "itineraries" in data["plan"].keys():
                            res_dict.update({"transit_time": int(data["plan"]["itineraries"][0]["duration"])})
                            res_dict.update({"transfers": int(data["plan"]["itineraries"][0]["transfers"])})
                            legs = pd.DataFrame(data["plan"]["itineraries"][0]["legs"])
                            mode_split = legs["mode"].dropna().to_list()
                            res_dict.update({"mode": mode_split})

                        else:
                            return False

                        return res_dict
                    return False
            except Exception as e:
                print(f"Error during API call: {e}")
                return False


async def process_rows_transit(dataframe, semaphore):
    loop = asyncio.get_event_loop()
    tasks = []
    for index, row in dataframe.iterrows():
        task = loop.create_task(
            traveltimeTransit(
                row["origin_loc_lat"],
                row["origin_loc_lon"],
                row["dest_loc_lat"],
                row["dest_loc_lon"],
                row["pickup_time_0_str"],
                semaphore,
            )
        )
        tasks.append(task)

    results = []
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        result = await f
        results.append(result)

    return results


max_workers = os.cpu_count() - 4


async def get_transit_time(trips_df):
    semaphore = asyncio.Semaphore(max_workers)  # Define semaphore
    results = await process_rows_transit(trips_df, semaphore)
    for i, res in enumerate(results):
        if res is False:
            # Set the relevant dictionary values to None
            trips_df.loc[
                i,
                [
                    "transit_time",
                    "transfers",
                    "mode",
                    "trip1",
                    "trip2",
                    "trip3",
                    "trip4",
                    "trip5",
                    "trip6",
                    "route1",
                    "route2",
                    "route3",
                    "route4",
                    "route5",
                    "route6",
                    "bus_capacity_upto_2hours",
                ],
            ] = [None] * 16
        else:
            # Update DataFrame with the results
            trips_df.loc[i, res.keys()] = res.values()

    return trips_df


async def get_drive_time(trips_df):
    semaphore = asyncio.Semaphore(max_workers)  # Define semaphore
    results = await process_rows_transit(trips_df, semaphore)
    for i, res in enumerate(results):
        if res is False:
            # Set the relevant dictionary values to None
            trips_df.loc[
                i,
                [
                    "transit_time",
                    "transfers",
                    "mode",
                ],
            ] = [None] * 16
        else:
            # Update DataFrame with the results
            trips_df.loc[i, res.keys()] = res.values()

    return trips_df


os.makedirs("./outputs", exist_ok=True)

OD_NAME = "lodes_2021-01-05"
trips = pd.read_csv(f"data/{OD_NAME}.csv")

trips = asyncio.run(get_transit_time(trips))
trips.to_csv(f"./outputs/{OD_NAME}_transit.csv")

trips = asyncio.run(get_drive_time(trips))
trips.to_csv(f"./outputs/{OD_NAME}_drive.csv")

# await get_transit_time(trips)  # Use this in an environment with an existing event loop (like Jupyter Notebook)
