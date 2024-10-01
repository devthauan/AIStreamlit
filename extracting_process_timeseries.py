import pandas as pd
import os
import json
import math
import atexit

#from cognite.client.credentials import Token
from cognite.client import CogniteClient, ClientConfig
from msal import SerializableTokenCache, PublicClientApplication
from cognite.client.credentials import Token
import streamlit as st

class ExtractProcessTimeseries():

    def __init__(self, 
                env : str = 'prod',
                path : str = '', 
                start_time : pd.Timestamp = pd.to_datetime('2024-01-01'),
                end_time : pd.Timestamp = pd.to_datetime('2025-01-01'),
                freq : str = '1m',
                aggregates : str = 'average' # Could be 'interpolation'
                 ):

        """
        Constructor of the RagFunctions class.
        Initializes an instance of the class, but does not perform any operations.
        """

        #self.token = self.get_token()
        self.env = env
        self.tags = self.read_tags(path)
        self.start_time = start_time
        self.end_time = end_time
        self.freq = freq
        self.path = path
        self.aggregates = aggregates

        # Initialize OpenAI client
        self.client = self.create_cognite_client(
            client_name="energy-management-utility-collector-client",
            project='celanese-dev', # testing in prod
            base_url="https://az-eastus-1.cognitedata.com",
        )

    def create_cognite_client(
            self,
            client_name: str, 
            project: str, 
            base_url: str
            ):
        
        """
        Connection information to the extraction environment.

        Parameters
        ----------
        client_name : str
            Name of the extraction environment.
        project : str
            Project name.
        base_url : str
            The extraction environment URL.

        Returns
        -------
            Connection information to access the database.
        """

        creds = self._get_credentials()
        token_credentials = Token(creds["access_token"])

        return CogniteClient(
            config=ClientConfig(
                client_name=client_name,
                project=project,
                base_url=base_url,
                credentials=token_credentials,
            )
        )

    def _get_credentials(self):

        """
        Loading access credentials.

        Parameters
        ----------

        Returns
        -------
            Access credentials.
        """
        authority  = st.secrets["authority"]
        client_id = st.secrets["client_id"]

        scopes = [
            "https://az-eastus-1.cognitedata.com/.default",
        ]
        port = "0"
        app = PublicClientApplication(
            client_id=client_id, authority=authority, token_cache=self._create_cache()
        )
        accounts = app.get_accounts()
        if accounts:
            return app.acquire_token_silent(scopes, account=accounts[0])
        
        #credentials = json.load(open(ROOT_PATH + os.sep + 'credentials.json'))
        return app.acquire_token_by_username_password(
                username=st.secrets["celanese_email"], 
                password=st.secrets["celanese_password"],                         
                scopes=[
                    "https://az-eastus-1.cognitedata.com/.default",
                ]
        )

    def _create_cache(self):

        """
        Cache creating.

        Parameters
        ----------

        Returns
        -------
            Temporary storage space in the browser that saves data from websites you have already visited
        """
        cache_file_name = "cache.bin"
        cache = SerializableTokenCache()

        if os.path.exists(cache_file_name):
            cache.deserialize(open(cache_file_name, "r").read())
        atexit.register(
            lambda: open(cache_file_name, "w").write(cache.serialize())
            if cache.has_state_changed
            else None
        )
        return cache
    
    def get_model_response_data(self):
        """
        extracts the data based on a list of
        external ids

        Returns
        pd.DataFrame
            dataframe with the time series
            extracted
        """

        tags = 'NRG:CLK:FuelGas:EQ_10071386:IncidentPredictionProbability' # using brute force wihout config
        freq = self.freq
        start_time = self.start_time
        end_time = self.end_time
        client = self.client
        path = self.path
        aggregates = self.aggregates

        data = client.time_series.data.retrieve_dataframe(
            external_id  = tags,
            aggregates   = aggregates, 
            granularity  = freq, 
            start        = start_time, 
            end          = end_time, 
            include_aggregate_name = False,
        )


        if(len(data)==0):

            data = self.rename_columns(path, data)
            return data
        
        for column in data.columns:
            if(data[column].iloc[0] is None or math.isnan(data[column].iloc[0])):
                data[column].iloc[0] = 0.0
                value = client.time_series.data.retrieve_latest(external_id=column,before=data[column].index[0]).value
                if len(value) != 0:
                    if not math.isnan(value[0]):
                        data[column].iloc[0] = value[0]    
            data[column]  = data[column].ffill()

        data = self.rename_columns(path, data)

        return data
        
    def get_timeseries_data(
                self,
                ):
        """
        extracts the data based on a list of
        external ids

        Returns
        pd.DataFrame
            dataframe with the time series
            extracted
        """

        tags = self.tags 
        freq = self.freq
        start_time = self.start_time
        end_time = self.end_time
        client = self.client
        path = self.path
        aggregates = self.aggregates

        data = client.time_series.data.retrieve_dataframe(
            external_id  = tags,
            aggregates   = aggregates, 
            granularity  = freq, 
            start        = start_time, 
            end          = end_time, 
            include_aggregate_name = False,
        )


        if(len(data)==0):

            data = self.rename_columns(path, data)
            return data
        
        for column in data.columns:
            if(data[column].iloc[0] is None or math.isnan(data[column].iloc[0])):
                data[column].iloc[0] = 0.0
                value = client.time_series.data.retrieve_latest(external_id=column,before=data[column].index[0]).value
                if len(value) != 0:
                    if not math.isnan(value[0]):
                        data[column].iloc[0] = value[0]    
            data[column]  = data[column].ffill()

        data = self.rename_columns(path, data)

        return data
    
    def read_tags(
            self,
            path : str
        ):
        
        file_extension = os.path.splitext(path)[1]

        if file_extension == '.txt':
            with open(path, 'r') as f:
                tags = [linha.strip() for linha in f.readlines()]

        elif file_extension == '.json':

            with open(path, 'r') as file:
                tags = json.load(file)

            tags = [tag['ExternalId'] for tag in tags['InputTags']]
        
        else:
            print('Cannot open this type of file!')

            tags = None

        return tags
    
    def rename_columns(
            self,
            path : str,
            data : pd.DataFrame
        ):
        
        file_extension = os.path.splitext(path)[1]

        if file_extension == '.json':

            with open(path, 'r') as file:
                tags = json.load(file)

            rename_dict = {f'{tag["ExternalId"]}': f'{tag["Alias"]}' for tag in tags['InputTags']}

            data.rename(columns = rename_dict, inplace = True)
            data.rename_axis('date', inplace = True)

        else:
            data.rename_axis('date', inplace = True)
            data = data

        return data