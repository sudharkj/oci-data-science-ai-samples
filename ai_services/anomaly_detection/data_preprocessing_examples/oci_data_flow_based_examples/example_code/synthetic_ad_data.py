import json
import math

from datetime import datetime
from pyspark.sql import SparkSession
import argparse
from pyspark.sql import functions as F
from pyspark.mllib.random import RandomRDDs
import numpy as np
import pandas as pd
import string
import random
from pyspark.sql.types import StringType


def create_spark_session():
    spark_session = SparkSession.builder.appName(
        "PySpark_synthetic_data_generator"
    ).getOrCreate()
    return spark_session


def gen_synthetic_pivot(spark, args):
    """
    Generate a dataset with three columns as timestamp, signal ID and signal value
    and num_signals x num_observations rows,
    the number of observations per signals are identical
    Args:
        spark: Spark session
        args: contains
            args.output: destination directory to store generated sample_datasets as a CSV file
            args.num_signals: number of signals
            args.num_observations: number of observations
            args.freq: sampling interval for time series
        max_cat: maximum number of categories in each signal/ column
    """
    destination = '/'.join([args.output, 'pivot', str(datetime.now())])

    timestamps = pd.date_range(
        start="1/1/2018", periods=args.observations, freq=args.frequency
    )
    rdd = spark.sparkContext.parallelize(range(int(args.values))).flatMap(
        lambda x: [
            (
                str(dt),
                f"meter-{x}",
                np.random.rand(),
            )
            + tuple(map(str, np.random.randint(3, size=args.max_pivot - 1)))
            for dt in timestamps
        ]
    )
    df = rdd.toDF(
        ["timestamp", "meter-ID", "value"]
        + [f"dim{i + 1}" for i in range(args.max_pivot - 1)]
    )
    # df = df.orderBy(F.rand())
    df.coalesce(1).write.csv(destination, header=True)


def generate(destination, values, categories, observations, shuffle=False, frequency='1min', offset=0):
    """
    Generate an anomaly detection dataset.

    :param destination: Output location
    :param values: list containing range of values in each column that has uniform distribution
    :param categories: list containing maximum number of categories in each categorical column
    :param observations: number of observations
    :param shuffle: shuffle column order (default = False)
    :param frequency: sampling interval for time series (default = 1 min)
    :param offset: offset in the feature index (default = 0)
    """
    destination = '/'.join([destination, 'synthetic', str(datetime.now())])

    def get_uniform_float(a, b):
        return np.random.uniform(a, b)

    def get_continuous_value(metadata):
        if isinstance(metadata, int):
            return get_uniform_float(0.0, metadata)

        if 'nan_prob' in metadata:
            is_nan_value = np.random.binomial(1, metadata['nan_prob'])
            if is_nan_value > 0:
                return math.nan

        if 'max' in metadata:
            if metadata['max'] == 1:
                return get_uniform_float(0, metadata['max'])
            return get_uniform_int(1, metadata['max'])

        if 'range' in metadata:
            a, b = metadata['range']
            if abs(a - b) == 1:
                return get_uniform_float(a, b)
            return get_uniform_int(a, b)

        raise AssertionError(f"Continuous is not given in proper format{json.dumps(metadata)}")

    def get_uniform_int(a, b):
        return np.random.randint(a, b + 1)

    def get_categorical_value(metadata):
        if isinstance(metadata, int) and metadata > 1:
            return get_uniform_int(1, metadata)

        if 'range' in metadata and len(metadata['range']) == 2:
            return get_uniform_int(*metadata['range'])

        if 'values' in metadata and len(metadata['values']) >= 2:
            ind = get_uniform_int(0, len(metadata['values']) - 1)
            return metadata['values'][ind]

        raise AssertionError("Category is not given in proper format")

    spark_session = create_spark_session()
    timestamps = pd.date_range(
        start="1/1/2018", periods=observations, freq=frequency
    )
    if isinstance(values, int):
        values = [1 for _ in range(values)]
    if isinstance(categories, int):
        categories = [2 for _ in range(categories)]
    columns = ["timestamp"] + [f"feat-{i + offset}" for i in range(len(values))] + [
        f"feat-{i + len(values) + offset}" for i in range(len(categories))]

    df = (
        spark_session.sparkContext.parallelize(range(observations))
        .map(lambda x: tuple([str(timestamps[x])]) + tuple(
            [get_continuous_value(continuous_metadata) for continuous_metadata in values]) + tuple(
            [get_categorical_value(category_metadata) for category_metadata in categories]))
        .toDF(columns)
    )

    if shuffle:
        columns = columns[1:]
        random.shuffle(columns)
        columns = ['timestamp'] + columns

    df.select(*columns).coalesce(1).write.csv(destination, header=True)


if __name__ == "__main__":
    DEFAULT_CATEGORIES = 0
    DEFAULT_SHUFFLE = False
    DEFAULT_FREQUENCY = "1min"
    DEFAULT_OFFSET = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--values", required=True, type=str)
    parser.add_argument("--categories", required=False, type=str, default=DEFAULT_CATEGORIES)
    parser.add_argument("--observations", required=True, type=int)
    parser.add_argument("--shuffle", required=False, type=bool, default=DEFAULT_SHUFFLE)
    parser.add_argument("--frequency", required=False, default=DEFAULT_FREQUENCY)
    parser.add_argument("--offset", required=False, type=int, default=DEFAULT_OFFSET)
    parser.add_argument("--max_pivot", required=False, type=int, default=DEFAULT_OFFSET)
    args = parser.parse_args()
    # gen_synthetic_pivot(create_spark_session(), args)
    generate(args.output, json.loads(str(args.values)), json.loads(str(args.categories)), args.observations,
             args.shuffle, args.frequency, int(args.offset))
