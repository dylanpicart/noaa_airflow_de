import os
import logging
import graphene
import requests
from flask import Flask, jsonify
from flask_cors import CORS
from flask_graphql import GraphQLView
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, concat_ws, mean
from graphql import GraphQLError

# Initialize logging
logging.basicConfig(
    filename='/usr/local/hadoop/logs/noaa_graphql_server.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s'
)

# Ensure NOAA token is loaded from environment
NOAA_TOKEN = os.getenv("NOAA_TOKEN")
if not NOAA_TOKEN:
    logging.warning("NOAA_TOKEN not set. Please export it to the environment.")

# Initialize Spark session with Hadoop integration
try:
    spark = SparkSession.builder \
        .appName("NOAA GraphQL Server") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.hadoop.home.dir", os.getenv("HADOOP_HOME")) \
        .config("spark.ui.port", "4045") \
        .getOrCreate()
    logging.info("SparkSession initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing SparkSession: {e}")
    raise

sc = spark.sparkContext

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Define GraphQL Data Types
class PMNData(graphene.ObjectType):
    time = graphene.DateTime()
    altitude = graphene.Float()
    latitude = graphene.Float()
    longitude = graphene.Float()
    fluorescence = graphene.Float()

class BuoyData(graphene.ObjectType):
    timestamp = graphene.DateTime()
    wave_height = graphene.Float()
    sst = graphene.Float()
    station = graphene.String()

class ClimateData(graphene.ObjectType):
    date = graphene.DateTime()
    value = graphene.Float()
    datatype = graphene.String()

# Define GraphQL Query Resolvers
class Query(graphene.ObjectType):
    get_pmn_data = graphene.List(PMNData)
    get_buoy_data = graphene.List(BuoyData, station_id=graphene.String(required=True))
    get_climate_data = graphene.List(
        ClimateData,
        dataset_id=graphene.String(required=True),
        location_id=graphene.String(required=True),
        start_date=graphene.DateTime(required=True),
        end_date=graphene.DateTime(required=True)
    )

    def resolve_get_pmn_data(self, info):
        url = (
            "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMWcflhmday.json?"
            "fluorescence%5B(2024-08-16T12:00:00Z):1:(2024-09-15T12:00:00Z)%5D"
            "%5B(0.0):1:(0.0)%5D%5B(32):1:(49)%5D%5B(235):1:(243)%5D"
        )
        return fetch_and_transform_pmn_data(url, ["time", "altitude", "latitude", "longitude", "fluorescence"])

    def resolve_get_buoy_data(self, info, station_id):
        url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.txt"
        return fetch_and_transform_buoy_data(url)

    def resolve_get_climate_data(self, info, dataset_id, location_id, start_date, end_date):
        """
        Resolver to fetch Climate data from NOAA API.
        """
        return fetch_and_transform_climate_data(dataset_id, location_id, start_date, end_date)


# Helper Functions
def fetch_and_transform_climate_data(dataset_id, location_id, start_date, end_date):
    """
    Fetch and transform Climate data from NOAA API.
    """
    headers = {"token": NOAA_TOKEN}
    url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"

    params = {
        "datasetid": dataset_id,
        "locationid": location_id,
        "startdate": start_date.isoformat(),  # Convert to ISO format
        "enddate": end_date.isoformat(),      # Convert to ISO format
        "limit": 1000
    }

    try:
        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            logging.error(f"Failed to fetch climate data: {response.status_code} - {response.text}")
            raise GraphQLError("Failed to fetch climate data.")

        data = response.json().get("results", [])

        return [
            ClimateData(
                date=item["date"],
                datatype=item["datatype"],
                value=item["value"]
            )
            for item in data
        ]
    except Exception as e:
        logging.error(f"Error fetching climate data: {str(e)}")
        raise GraphQLError("Error fetching climate data.")


def fetch_and_transform_pmn_data(url, columns):
    response = requests.get(url)
    if response.status_code != 200:
        logging.error(f"Failed to fetch PMN data: {response.status_code} - {response.text}")
        return []

    data = response.json()["table"]["rows"]
    rdd = sc.parallelize(data)
    df = spark.createDataFrame(rdd, schema=columns)

    df_clean = df.withColumn("time", to_timestamp("time")) \
                 .withColumn("fluorescence", col("fluorescence").cast("float")) \
                 .filter(col("latitude").isNotNull() & col("longitude").isNotNull())

    mean_value = df_clean.select(mean(col("fluorescence"))).collect()[0][0]
    df_imputed = df_clean.fillna({"fluorescence": mean_value}).repartition(20)

    return df_imputed.collect()

def fetch_and_transform_buoy_data(url):
    response = requests.get(url)
    lines = response.text.splitlines()[2:]
    rdd = sc.parallelize([line.split() for line in lines])
    schema = ["year", "month", "day", "hour", "minute", "wave_height", "sst", "station"]
    df = spark.createDataFrame(rdd, schema=schema)

    df_clean = df.withColumn(
        "timestamp", to_timestamp(concat_ws(" ", "year", "month", "day", "hour", "minute"), "yyyy MM dd HH mm")
    ).filter(col("wave_height").isNotNull())

    return df_clean.collect()

# GraphQL schema setup
schema = graphene.Schema(query=Query)

app.add_url_rule(
    "/graphql", view_func=GraphQLView.as_view("graphql", schema=schema, graphiql=True)
)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
