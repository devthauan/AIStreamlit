from cognite.client.credentials import Token
from cognite.client import CogniteClient, ClientConfig
from cognite.client.data_classes import TimeSeries
from datetime import datetime,timedelta
import pytz
import pandas as pd
import logging

import os,sys

ROOT_PATH    = os.getcwd()
ROOT_PATH    = os.path.abspath(os.path.join(ROOT_PATH, '..'))
sys.path.append(ROOT_PATH)

from extracting_process_timeseries import ExtractProcessTimeseries
from dump_process_timeseries import DumpProcessTimeseries

def setup_logging(log_file='cronjob.log', level=logging.DEBUG):
    """Set up logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_file,
        filemode='a'
    )

features_list = ['pressure(psig)', 'bfw_turbine_speed(rpm)', 'bfw_pump_flow(gpm)']

def get_model_response(
        df : pd.DataFrame,
):

    import joblib
    # load model
    pipeline = joblib.load("../data/best_pipeline.pkl")

    features_list = ['pressure(psig)', 'bfw_turbine_speed(rpm)', 'bfw_pump_flow(gpm)']

    return get_anomaly_probability(df[features_list])['anomaly_probability'].values

# function used for model training
def get_anomaly_probability(
        df : pd.DataFrame,
        gama : float = 0.1,
        kernel : str = 'sigmoid',
        nu : float = 0.1
        ):
    """
    This function uses a One-Class Support Vector Machine (SVM) to detect anomalies in the 
    data and calculates an anomaly probability for each observation based on the model's decision function.

    Parameters
    df (pd.DataFrame): 
        The DataFrame containing the data for anomaly detection.
    columns (list): 
        A list of column names from df that will be used as features for the anomaly detection model.
    gama (str, optional): 
        The kernel coefficient for the One-Class SVM model.
    nu (float, optional): 
        An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors. 
    
    Returns
        The function returns the original DataFrame with two new columns:
        anomaly: Whether the observation is an anomaly (1) or normal (0).
        anomaly_probability: The probability (in percentage) that the observation is an anomaly, based on its score.
    """
    from sklearn.preprocessing import RobustScaler
    from sklearn.svm import OneClassSVM
    import numpy as np

    # df = df[columns]
    scaler = RobustScaler()
    df_scaled_svm = scaler.fit_transform(df)

    ocsvm = OneClassSVM(gamma=gama, nu=nu, kernel=kernel)
    ocsvm.fit(df_scaled_svm)
    df[f'anomaly'] = ocsvm.predict(df_scaled_svm)
    df[f'anomaly'] = df[f'anomaly'].map({1:0,-1: 1})

    ocsvm_scores = - ocsvm.decision_function(df_scaled_svm)
    ocsvm_scores = (ocsvm_scores - np.min(ocsvm_scores)) / (np.max(ocsvm_scores) - np.min(ocsvm_scores)) #normalization of the pts
    df[f'anomaly_probability'] = ocsvm_scores
    df[f'anomaly_probability'] = (df[f'anomaly_probability'] * 100).round(2)
    df[f'anomaly_probability'] = df[f'anomaly_probability'].astype(int)

    return df

def build_model(model_filename):
    import numpy as np
    from sklearn.base import BaseEstimator,RegressorMixin
    from sklearn.preprocessing import MinMaxScaler
    import joblib
    class IncidentProb(BaseEstimator,RegressorMixin):
        def __init__(self,model_filename="../data/best_pipeline.pkl"):
            self.pipeline = joblib.load(model_filename)

        def fit(self,X):
            return self.pipeline.fit(X)

        def predict(self,X):

            # calculate the probability of event an event is an inlier ou outlier
            prob_func = self.pipeline.decision_function(X)
            scaler = MinMaxScaler(feature_range=(0,100))
            prob_func= scaler.fit_transform(prob_func.reshape(-1,1)).flatten()

            # # If prob_func is negative, the event is considered an anomally
            # if np.all(prob_func < 0):
            #     if np.all(np.abs(prob_func) > 100):
            #         prob_func = [100]
            #     prob_func=np.abs(prob_func)*100
            # else:
            #     prob_func = [0]

            return np.round(prob_func,2)
    model = IncidentProb(model_filename)
    return model

def retrieve_cognite_data(start_time,end_time, features_list = ['pressure(psig)', 'bfw_turbine_speed(rpm)', 'bfw_pump_flow(gpm)']):

    extract = ExtractProcessTimeseries(env='dev',
                             path='probability_score.json',
                             start_time=start_time,
                             end_time=end_time,
                             freq='1m')
       
    df_extract = extract.get_timeseries_data()
    df_extract = df_extract[features_list]
    logging.info(df_extract)

    # Change index name from date to timestamp
    df_extract = df_extract.reset_index().rename(columns={'date':'timestamp'})
    df_extract.set_index('timestamp',inplace=True)

    return df_extract

def retrieve_model_cognite_data(start_time,end_time):
    extract = ExtractProcessTimeseries(env='dev',
                             path='probability_score.json',
                             start_time=start_time,
                             end_time=end_time,
                             freq='1m')
       
    df_extract = extract.get_model_response_data()
    df_extract = df_extract
    logging.info(df_extract)

    # Change index name from date to timestamp
    df_extract = df_extract.reset_index().rename(columns={'date':'timestamp'})
    df_extract.set_index('timestamp',inplace=True)

    return df_extract


# def build_model_response(df_input,features_list):

#     # train model

#     # retrieve model response
#     model = build_model(model_filename="../data/best_pipeline.pkl")
#     df_input['anomaly_probability'] = model.predict(df_input[features_list].values)

#     logging.debug(df_input[features_list])

#     df_model_response = df_input['anomaly_probability'].to_frame().reset_index().rename(columns={'date':'timestamp'})
#     df_model_response.set_index('timestamp',inplace=True)
#     return df_model_response

def insert_cognite_data(start_time,end_time,x_pred,y_pred,config_file):
    """
    Parameters:
    - start_time (datetime)
    - end_time (datetime)
    - x_pred (pd.DataFrame)
    - y_pred (np.array)


    """

    x_pred['anomaly_probability'] = y_pred
    df_model_response = x_pred[['anomaly_probability']]

    dump = DumpProcessTimeseries(env='dev',
                                path=config_file,
                                start_time=start_time,
                                end_time=end_time,
                                freq='1m')

    output_tags = dump.get_retrieve_output_tag()

    # Write data on Cognite
    for output_tag in output_tags:
        logging.debug(f"""ExternalId: {output_tag['ExternalId']}\nDescription: {output_tag['Description']}\nName: {output_tag['KPICode']}""")
        df_model_response.columns = [output_tag['ExternalId']]
        logging.debug(df_model_response)

        dump.create_and_save_time_series_data(
            data=df_model_response,
            column_name=output_tag['ExternalId'],
            unit='%',
            # data_set_id
            # prod 2390801079406030
            # dev 5140341703636106
            data_set_id="5140341703636106", # testing prod
            name= output_tag['KPICode'],
            description=output_tag['Description']
        )


def cronjob(start_time,end_time):

    # incident dates
    incident_dates = ['2023-01-27 10:36:00', '2023-01-27 07:52:00']

    # start_time = pd.to_datetime('2023-01-27 07:50:00')
    # end_time = pd.to_datetime('2023-01-27 10:38:00')

    logging.info(f"[Started]\nUpdating dates from:\nstart: {start_time}\nend: {end_time}")

    extract = ExtractProcessTimeseries(env='dev',
                             path='probability_score.json',
                             start_time=start_time,
                             end_time=end_time,
                             freq='1m')
    
    dump = DumpProcessTimeseries(env='dev',
                             path='probability_score.json',
                             start_time=start_time,
                             end_time=end_time,
                             freq='1m')
    
    df_extract = extract.get_timeseries_data()
    df_extract = df_extract[features_list]
    logging.info(df_extract)

    # df_extract['anomaly_probability'] = get_model_response(df_extract)
    model = build_model(model_filename="../data/best_pipeline.pkl")
    df_extract['anomaly_probability'] = model.predict(df_extract[features_list].values)

    logging.debug(df_extract[features_list])

    df_model_response = df_extract['anomaly_probability'].to_frame().reset_index().rename(columns={'date':'timestamp'})
    df_model_response.set_index('timestamp',inplace=True)
    df_model_response.rename(columns={'anomaly_probability':'NRG:CLK:FuelGas:EQ_10071386:IncidentPredictionProbability'},inplace=True)
    logging.info(df_model_response)
    # df_model_response['NRG:CLK:FuelGas:EQ_10071386:IncidentPredictionProbability'] =  df_model_response['NRG:CLK:FuelGas:EQ_10071386:IncidentPredictionProbability']
    output_tags = dump.get_retrieve_output_tag()

    # Write data on Cognite
    for output_tag in output_tags:
        logging.debug(f"""ExternalId: {output_tag['ExternalId']}\nDescription: {output_tag['Description']}\nName: {output_tag['KPICode']}""")
        df_model_response.columns = [output_tag['ExternalId']]
        logging.debug(df_model_response)

        dump.create_and_save_time_series_data(
            data=df_model_response,
            column_name=output_tag['ExternalId'],
            unit='%',
            data_set_id="5140341703636106",
            name= output_tag['KPICode'],
            description=output_tag['Description']
        )

    # dump.create_and_save_time_series_data(
    #     data=df_model_response,
    #     column_name='NRG:CLK:FuelGas:EQ_10071386:IncidentPredictionProbability',
    #     unit='%',
    #     data_set_id="5140341703636106",
    #     name= "P1221 Incident Prediction Probability",
    #     description="""Indicates the score of the anomaly for pump p1221, where there score of 0% represents a low chance to fail and 100% represents a very high chance to fail."""
    # )        

    logging.info(f"[Finished] start: {start_time}\nend: {end_time}")


if __name__ == "__main__":

    setup_logging(log_file='../data/cronjob.log', level=logging.DEBUG)

    tzinfo= pytz.utc

    end_time = datetime.now(tzinfo)
    start_time = end_time - timedelta(minutes=20)

    cronjob(start_time=start_time,end_time=end_time)

    # # incident dates
    # incident_dates = ['2023-01-27 10:36:00', '2023-01-27 07:52:00']

    # # start_time = pd.to_datetime('2023-01-27 07:50:00')
    # # end_time = pd.to_datetime('2023-01-27 10:38:00')

    # for output_tag in output_tags:
    #     print(f"""ExternalId: {output_tag['ExternalId']}\nDescription: {output_tag['Description']}\nName: {output_tag['KPICode']}""")
    #     df_model_response.columns = [output_tag['ExternalId']]
    #     print(df_model_response)

    # dump.create_and_save_time_series_data(
    #     data=df_model_response,
    #     column_name='NRG:CLK:FuelGas:EQ_10071386:IncidentPredictionProbability',
    #     unit='%',
    #     data_set_id="5140341703636106",
    #     name= "P1221 Incident Prediction Probability",
    #     description="""Indicates the score of the anomaly for pump p1221, where there score of 0% represents a low chance to fail and 100% represents a very high chance to fail."""
    # )
