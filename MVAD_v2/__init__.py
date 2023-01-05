import logging

import azure.functions as func
import os #, json
import time
import tempfile
import pandas as pd
from datetime import datetime #, timedelta
from azure.storage.blob import BlobClient, BlobServiceClient 
from azure.ai.anomalydetector import AnomalyDetectorClient
from azure.ai.anomalydetector.models import *
from azure.core.credentials import AzureKeyCredential
from azure.kusto.data import KustoClient, KustoConnectionStringBuilder 
from azure.kusto.data.helpers import dataframe_from_result_table
from dotenv import load_dotenv

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    #load_dotenv('.env')
    client_id = os.getenv("CLIENT_ID")
    client_secret = os.getenv("CLIENT_SECRET")
    authority_id = os.getenv("AUTHORITY_ID")
    connection_string = os.getenv("STORAGE_CONN_STR")
    container_name = os.getenv("CONTAINER_NAME")
    adx_cluster = os.getenv("ADX_CLUSTER")
    db_name = os.getenv("DB_NAME")


    table_name = req.params.get('table_name')
    trained_model_id = req.params.get('trained_model_id')
    kql_start_time = req.params.get('kql_start_time')
    kql_end_time = req.params.get('kql_end_time')
    input_file = req.params.get('input_file')
    result_file_name = "result_anomaly.csv"

    # ### Retrieve CSV file
    # ISSUE KQL Query
    kcsb = KustoConnectionStringBuilder.with_aad_application_key_authentication(adx_cluster, client_id, client_secret, authority_id)
    client = KustoClient(kcsb)


    #print parameters
    logging.info(f'kcsb: {kcsb}, client: {client}, db_name: {db_name}, table_name: {table_name}')


    db = db_name
    query = table_name + " \
    | where timestamp >= datetime(" + kql_start_time + ") \
        and timestamp <= datetime(" + kql_end_time + ") \
    | project timestamp, axis_a_RMS, axis_v_RMS, X_a_RMS, X_v_RMS, Y_a_RMS, Y_v_RMS, Z_a_RMS, Z_v_RMS;"

    response = client.execute(db, query)

    df = dataframe_from_result_table(response.primary_results[0])
    df['timestamp'] = df['timestamp'].apply(lambda x: x.strftime('%Y-%m-%dT%H:%M:%SZ'))


    BLOB_SAS_TEMPLATE = "https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}"

    def upload_to_blob(file, conn_str, container, blob_name):
        """
        A helper function to upload files to blob
        :param file: the path to the file to be uploaded
        :param conn_str: the connection string of the target storage account
        :param container: the container name in the storage account
        :param blob_name: the blob name in the container
        """
        blob_client = BlobClient.from_connection_string(conn_str, container_name=container, blob_name=blob_name)
        with open(file, "rb") as f:
            blob_client.upload_blob(f, overwrite=True)
        print("Upload Success!")


    def generate_data_source_sas(conn_str, container, blob_name):
        """
        A helper function to generate blob SAS.
        :param conn_str: the connection string of the target storage account
        :param container: the container name in the storage account
        :param blob_name: the blob name in the container
        :return: generated SAS
        """
        blob_service_client = BlobServiceClient.from_connection_string(conn_str=conn_str)
        blob_path = BLOB_SAS_TEMPLATE.format(account_name=blob_service_client.account_name,
                                            container_name=container,
                                            blob_name=blob_name)
        return blob_path


    tempFilePath = tempfile.gettempdir()
    t = datetime.now().strftime("%H-%M-%S-%f")
    source_folder = tempFilePath + "/sample_data_MVAD" + t
    os.makedirs(source_folder)


    df.to_csv(os.path.join(source_folder, input_file), mode='w', index=False, header=True, encoding='utf-8')

    # ### Upload to BLOB
    upload_to_blob(os.path.join(source_folder, input_file), connection_string, container_name, input_file)
    data_source = generate_data_source_sas(connection_string, container_name, input_file)
    print("Blob path url: " + data_source)


    # ## Set Credential

    SUBSCRIPTION_KEY = os.getenv("ANOMALY_DETECTOR_KEY")
    ANOMALY_DETECTOR_ENDPOINT = os.getenv("ANOMALY_DETECTOR_ENDPOINT")
    anomaly_detector_endpoint = 'https://{endpoint}'.format(endpoint=ANOMALY_DETECTOR_ENDPOINT)
    ad_client = AnomalyDetectorClient(anomaly_detector_endpoint, AzureKeyCredential(SUBSCRIPTION_KEY))


    test = {}
    test['start_time'] = df["timestamp"][29]       # Skip first 30 records for sliding_window
    test['end_time'] = df["timestamp"][len(df)-1]
    test['duration'] = len(df) -30                 # sliding_window = 30
    print(test)

    # ### A. Inference asynchronously
    # ### Set start-time and end-time

    # Specify the start time and end time for inference.
    start_inference_time = test['start_time']
    end_inference_time = test['end_time']


    batch_inference_body = MultivariateBatchDetectionOptions(
            data_source = data_source,
            # The topContributorCount specify how many contributed variables you care about in the results, from 1 to 50.
            top_contributor_count=10,
            start_time = start_inference_time,
            end_time = end_inference_time,
        )

    # Send inference request

    results = ad_client.detect_multivariate_batch_anomaly(trained_model_id, batch_inference_body)
    result_id = results.result_id
    print(f"A batch inference is triggered with the resultId: {result_id}")

    # Wait for inference to complete

    # Get results (may need a few seconds)
    r = ad_client.get_multivariate_batch_detection_result(result_id)
    print("Get detection result...(it may take a few seconds)")

    while r.summary.status != 'READY' and r.summary.status != 'FAILED':
        r = ad_client.get_multivariate_batch_detection_result(result_id)
        time.sleep(1)
                
    print("Result ID:\t", r.result_id)
    print("Result status:\t", r.summary.status)
    print("Result length:\t", len(r.results))


    anomaly_results_df = pd.DataFrame([{'timestamp': x['timestamp'], **x['value']} for x in r.results])

    # raw_data=pd.read_csv('./sample_data_MVAD/test.csv') -> eq. to df
    df_merge=pd.merge(anomaly_results_df, df, on="timestamp")

    result_file_path = os.path.join(source_folder, result_file_name)
    df_merge.to_csv(result_file_path, mode='w', index=False, encoding='utf-8')

    # Upload it
    upload_to_blob(result_file_path, connection_string, container_name, result_file_name)

    # ### End of processing
    return func.HttpResponse(f"Saved to {result_file_name}. Function executed successfully.")
