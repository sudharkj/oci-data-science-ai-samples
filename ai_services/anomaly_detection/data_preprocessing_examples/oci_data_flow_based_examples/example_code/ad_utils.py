import json
from argparse import ArgumentParser

from oci.ai_anomaly_detection import AnomalyDetectionClient
from oci.ai_anomaly_detection.models import CreateModelDetails, ModelTrainingDetails, CreateDataAssetDetails, \
    DataSourceDetailsObjectStorage
from oci.object_storage import ObjectStorageClient
from pyspark.sql import SparkSession
import oci
import os

DEFAULT_LOCATION = os.path.join('~', '.oci', 'config')
DEFAULT_PROFILE = "DEFAULT"
DEFAULT_TARGET_FAP = 0.01
DEFAULT_TRAINING_FRACTION = 0.7


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


class AdUtils:
    def __init__(self, profile_name=DEFAULT_PROFILE, service_endpoint=None):
        token_path = get_token_path()

        client_args = {'profile_name': profile_name}
        self.obj_client = get_authenticated_client(token_path, ObjectStorageClient, **client_args)

        if service_endpoint:
            client_args['service_endpoint'] = service_endpoint
        self.ad_client = get_authenticated_client(token_path, AnomalyDetectionClient, **client_args)

    def train(self, project_id, compartment_id, data_assets, target_fap=DEFAULT_TARGET_FAP,
              training_fraction=DEFAULT_TRAINING_FRACTION):
        assert data_assets['type'] == 'object_storage', "Unknown dataset details"
        list_objects_response = self.obj_client.list_objects(namespace_name=data_assets['namespace'],
                                                             bucket_name=data_assets['bucket'],
                                                             prefix=data_assets['prefix'])
        assert list_objects_response.status == 200, f'Error listing objects: {list_objects_response.text}'
        objects_details = list_objects_response.data

        model_ids = []
        for object_details in objects_details.objects:
            if object_details.name.endswith('.csv'):
                data_asset_id = self._create_data_asset_(project_id=project_id, compartment_id=compartment_id,
                                                         namespace=data_assets['namespace'],
                                                         bucket=data_assets['bucket'],
                                                         object_name=object_details.name)
                model_id = self._create_model_(project_id=project_id, compartment_id=compartment_id,
                                               data_asset_id=data_asset_id, target_fap=target_fap,
                                               training_fraction=training_fraction)
                model_ids.append(model_id)
        return model_ids

    def _create_data_asset_(self, project_id, compartment_id, namespace, bucket, object_name):
        data_source_details = DataSourceDetailsObjectStorage(namespace=namespace, bucket_name=bucket,
                                                             object_name=object_name)
        create_data_asset_details = CreateDataAssetDetails(compartment_id=compartment_id, project_id=project_id,
                                                           data_source_details=data_source_details)
        data_asset_create_response = self.ad_client.create_data_asset(create_data_asset_details)
        assert data_asset_create_response.status == 200, f"Error creating data-asset: {data_asset_create_response.text}"
        return data_asset_create_response.data.id

    def _create_model_(self, project_id, compartment_id, data_asset_id, target_fap, training_fraction):
        model_training_details = ModelTrainingDetails(target_fap=target_fap, training_fraction=training_fraction,
                                                      data_asset_ids=[data_asset_id])
        create_model_details = CreateModelDetails(compartment_id=compartment_id, project_id=project_id,
                                                  model_training_details=model_training_details)
        create_model_response = self.ad_client.create_model(create_model_details)
        assert create_model_response.status == 201, f"Error creating model: {create_model_response.text}"
        return create_model_response.data.id


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--profile_name", required=False, type=str, default=DEFAULT_PROFILE)
    parser.add_argument("--service_endpoint", required=False, type=str, default=None)
    parser.add_argument("--project_id", required=True, type=str)
    parser.add_argument("--compartment_id", required=True, type=str)
    parser.add_argument("--target_fap", required=False, type=lambda v: float(v), default=DEFAULT_TARGET_FAP)
    parser.add_argument("--training_fraction", required=False, type=lambda v: float(v),
                        default=DEFAULT_TRAINING_FRACTION)
    parser.add_argument("--staging", required=True, type=str)
    args = parser.parse_args()

    ad_utils = AdUtils(profile_name=args.profile_name, service_endpoint=args.service_endpoint)
    staging = json.loads(str(args.staging))
    models = ad_utils.train(project_id=args.project_id, compartment_id=args.compartment_id, data_assets=staging)
    print(f"Model ids: {models}")
