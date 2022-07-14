import datetime
import json
import time
from argparse import ArgumentParser

from oci.ai_anomaly_detection.models import CreateModelDetails, ModelTrainingDetails
from pyspark.sql import SparkSession
import oci
import os

DEFAULT_LOCATION = os.path.join('~', '.oci', 'config')
DEFAULT_PROFILE = "DEFAULT"


# Helper Functions
def get_spark_context():
    return SparkSession.builder.appName("AnomalyDetectionClient").getOrCreate()


def get_token_path():
    sc = get_spark_context()
    token_key = "spark.hadoop.fs.oci.client.auth.delegationTokenPath"
    token_path = sc.sparkContext.getConf().get(token_key)
    return token_path


def get_authenticated_client(token_path, client, file_location=DEFAULT_LOCATION, profile_name=DEFAULT_PROFILE,
                             **kwargs):
    if token_path is None:
        # You are running locally, so use our API Key.
        config = oci.config.from_file(file_location, profile_name)
        kwargs['config'] = config
        authenticated_client = client(**kwargs)
    else:
        # You are running in Data Flow, so use our Delegation Token.
        with open(token_path) as fd:
            delegation_token = fd.read()
        signer = oci.auth.signers.InstancePrincipalsDelegationTokenSigner(
            delegation_token=delegation_token
        )
        kwargs['config'] = {}
        kwargs['signer'] = signer
        authenticated_client = client(**kwargs)
    return authenticated_client


def ad_train(project_id, compartment_id, data_asset_id, target_fap, training_fraction):
    model_training_details = ModelTrainingDetails(target_fap=target_fap, training_fraction=training_fraction,
                                                  data_asset_ids=[data_asset_id])
    create_model_details = CreateModelDetails(compartment_id=compartment_id, project_id=project_id,
                                              model_training_details=model_training_details)
    return ad_client.create_model(create_model_details)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--profile_name", required=False, type=str, default=DEFAULT_PROFILE)
    parser.add_argument("--service_endpoint", required=False, type=str, default=None)
    parser.add_argument("--project_id", required=True, type=str)
    parser.add_argument("--compartment_id", required=True, type=str)
    parser.add_argument("--data_asset_id", required=True, type=str)
    parser.add_argument("--target_fap", required=False, type=lambda v: float(v), default=0.01)
    parser.add_argument("--training_fraction", required=False, type=lambda v: float(v), default=0.7)
    parser.add_argument("--namespace", required=True, type=str)
    parser.add_argument("--bucket", required=True, type=str)
    parser.add_argument("--output", required=False, type=str,
                        default=str(time.mktime(datetime.datetime.now().timetuple())))
    args = parser.parse_args()

    _token_path = get_token_path()

    obj_client = get_authenticated_client(_token_path, oci.object_storage.ObjectStorageClient,
                                          profile_name=args.profile_name)

    if args.service_endpoint:
        ad_client = get_authenticated_client(_token_path, oci.ai_anomaly_detection.AnomalyDetectionClient,
                                             profile_name=args.profile_name, service_endpoint=args.service_endpoint)
    else:
        ad_client = get_authenticated_client(_token_path, oci.ai_anomaly_detection.AnomalyDetectionClient,
                                             profile_name=args.profile_name)
    training_response = ad_train(args.project_id, args.compartment_id, args.data_asset_id, args.target_fap,
                                 args.training_fraction)
    response = {
        'status': training_response.status,
        'headers': dict(training_response.headers),
        'data': json.loads(str(training_response.data))
    }
    obj_client.put_object(args.namespace, args.bucket, args.output, json.dumps(response, indent=2, sort_keys=True))
