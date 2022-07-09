import json
import math
import time

import datetime
from pyspark.sql import SparkSession
import argparse
import numpy as np
import pandas as pd
import random

from pyspark.sql.utils import AnalysisException

DEFAULT_CATEGORIES = 0
DEFAULT_COLUMN_PREFIX = 'meter-'
DEFAULT_SHUFFLE = False
DEFAULT_FREQUENCY = "1min"
DEFAULT_OFFSET = 0
DEFAULT_DIMENSIONS = 2
DEFAULT_TIMESTAMP_FORMAT = None
DEFAULT_IGNORE_NAN = False
DEFAULT_ADDITIONAL_COLUMNS = 0


def get_spark_context():
    return SparkSession.builder.appName("PySpark_synthetic_data_generator").getOrCreate()


def get_vectorized_definition(metadata, default_value):
    if isinstance(metadata, int):
        return [default_value for _ in range(metadata)]
    return metadata


def formatted_timestamp(timestamp, timestamp_format):
    if timestamp_format is None or len(timestamp_format) == 0:
        return str(time.mktime(timestamp.timetuple()))
    return timestamp.strftime(timestamp_format)


def generate_pivoted(destination, values, observations, dimensions=DEFAULT_DIMENSIONS, frequency=DEFAULT_FREQUENCY,
                     column_prefix=DEFAULT_COLUMN_PREFIX, offset=DEFAULT_OFFSET,
                     timestamp_format=DEFAULT_TIMESTAMP_FORMAT, ignore_nan=DEFAULT_IGNORE_NAN, **kwargs):
    """
    Generate a dataset with three columns as timestamp, signal ID and signal value
    and num_signals x num_observations rows,
    the number of observations per signals are identical

    :param destination: Output location
    :param values: list containing range of values in each column that has uniform distribution
    :param observations: number of observations
    :param dimensions: number of dimensions that can be used for pivoting
    :param frequency: sampling interval for time series (default = 1 min)
    :param column_prefix: prefix of the column name (default = "meter-")
    :param offset: offset in the feature index (default = 0)
    :param timestamp_format: timestamp_format (default = None, that returns unix timestamp)
    :param ignore_nan: ignore nan probs in columns (default = False)
    """
    timestamps = pd.date_range(start="1/1/2018", periods=observations, freq=frequency)
    values = get_vectorized_definition(values, 1)

    sc = get_spark_context()
    rdd = sc.sparkContext.parallelize(range(len(values))).flatMap(
        lambda x: [
            (
                formatted_timestamp(timestamp, timestamp_format),
                f"{column_prefix}{offset + x}",
                float(get_continuous_value(values[x], ignore_nan)),
            )
            + tuple(map(str, np.random.randint(3, size=dimensions - 1)))
            for timestamp in timestamps
        ]
    )
    df = rdd.toDF(
        ["timestamp", f"{column_prefix}ID", "value"] + [f"dim{i + 1}" for i in range(dimensions - 1)]).coalesce(1)
    try:
        df.write.csv(destination, header=True)
    except AnalysisException:
        destination = '/'.join([destination, f'pivot-{str(datetime.datetime.now())}'])
        df.write.csv(destination, header=True)


def get_uniform_float(a, b):
    return np.random.uniform(a, b)


def get_continuous_value(metadata, ignore_nan=DEFAULT_IGNORE_NAN):
    if isinstance(metadata, int):
        return get_uniform_float(0.0, metadata)

    if not ignore_nan and 'nan_prob' in metadata:
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


def generate(destination, values, categories, observations, shuffle=DEFAULT_SHUFFLE, frequency=DEFAULT_FREQUENCY,
             column_prefix=DEFAULT_COLUMN_PREFIX, offset=DEFAULT_OFFSET, timestamp_format=DEFAULT_TIMESTAMP_FORMAT,
             ignore_nan=DEFAULT_IGNORE_NAN, additional_columns=DEFAULT_ADDITIONAL_COLUMNS, **kwargs):
    """
    Generate an anomaly detection dataset.

    :param destination: Output location
    :param values: list containing range of values in each column that has uniform distribution
    :param categories: list containing maximum number of categories in each categorical column
    :param observations: number of observations
    :param shuffle: shuffle column order (default = False)
    :param frequency: sampling interval for time series (default = 1 min)
    :param column_prefix: prefix of the column name (default = "meter-")
    :param offset: offset in the feature index (default = 0)
    :param timestamp_format: timestamp_format (default = None, that returns unix timestamp)
    :param additional_columns: number of additional random uniform columns (default = 0)
    :param ignore_nan: ignore nan probs in columns (default = False)
    """
    values = get_vectorized_definition(values, 1)
    additional_columns = get_vectorized_definition(additional_columns, 1)
    categories = get_vectorized_definition(categories, 2)

    sc = get_spark_context()
    timestamps = pd.date_range(start="1/1/2018", periods=observations, freq=frequency)
    columns = ["timestamp"] + [f"{column_prefix}{i + offset}" for i in range(len(values) + len(categories))] + [
        f"random-{i + offset}" for i in range(len(additional_columns))]

    df = (
        sc.sparkContext.parallelize(range(observations))
        .map(lambda x: tuple([formatted_timestamp(timestamps[x], timestamp_format)]) + tuple(
            [float(get_continuous_value(continuous_metadata, ignore_nan)) for continuous_metadata in values]) + tuple(
            [int(get_categorical_value(category_metadata)) for category_metadata in categories]) + tuple(
            [float(get_continuous_value(continuous_metadata, ignore_nan)) for continuous_metadata in additional_columns]))
        .toDF(columns)
    )

    if shuffle:
        columns = columns[1:]
        random.shuffle(columns)
        columns = ['timestamp'] + columns

    df = df.select(*columns).coalesce(1)
    try:
        df.write.csv(destination, header=True)
    except AnalysisException:
        destination = '/'.join([destination, f'unpivot-{str(datetime.datetime.now())}'])
        df.write.csv(destination, header=True)


def boolean(value):
    if value.lower() in ["false", "no", "n", "0", "f"]:
        return False
    return True


def mode(value):
    if value.lower() in ['p', 'pivot', 'pivoted', 'generate_pivot', 'generated_pivoted']:
        return generate_pivoted
    return generate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=False, type=mode, default=generate)
    parser.add_argument("--output", required=True)
    parser.add_argument("--values", required=True, type=str)
    parser.add_argument("--categories", required=False, type=str, default=DEFAULT_CATEGORIES)
    parser.add_argument("--observations", required=True, type=int)
    parser.add_argument("--shuffle", required=False, type=boolean, default=str(DEFAULT_SHUFFLE))
    parser.add_argument("--frequency", required=False, default=DEFAULT_FREQUENCY)
    parser.add_argument("--column_prefix", required=False, default=DEFAULT_COLUMN_PREFIX)
    parser.add_argument("--offset", required=False, type=int, default=DEFAULT_OFFSET)
    parser.add_argument("--dimensions", required=False, type=int, default=DEFAULT_DIMENSIONS)
    # acceptable format: %Y-%m-%dT%H:%M:%S
    parser.add_argument("--timestamp_format", required=False, type=str, default=DEFAULT_TIMESTAMP_FORMAT)
    parser.add_argument("--ignore_nan", required=False, type=boolean, default=str(DEFAULT_IGNORE_NAN))
    parser.add_argument("--additional_columns", required=False, type=int, default=DEFAULT_ADDITIONAL_COLUMNS)
    args = parser.parse_args()

    args.mode(destination=args.output, values=json.loads(str(args.values)),
              categories=json.loads(str(args.categories)), observations=args.observations, dimensions=args.dimensions,
              shuffle=args.shuffle, frequency=args.frequency, column_prefix=args.column_prefix, offset=int(args.offset),
              timestamp_format=args.timestamp_format, ignore_nan=args.ignore_nan,
              additional_columns=args.additional_columns)
