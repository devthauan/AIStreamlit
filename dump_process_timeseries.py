import pandas as pd
import os
import json
import math
import atexit

from cognite.client.credentials import Token
# from cognite.client import CogniteClient
from cognite.client import CogniteClient, ClientConfig
from msal import SerializableTokenCache, PublicClientApplication
from cognite.client.data_classes import TimeSeries
from typing import List,Dict,Any

class DumpProcessTimeseries():

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
        self.output_tags = self.read_outputs(path)
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
        authority  = "https://login.microsoftonline.com/7a3c88ff-a5f6-449d-ac6d-e8e3aa508e37"
        client_id = "1fbf5ca4-0dda-4fc0-bf1e-3648abf580ec"

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
                username="heitor.santos_contractor@celanese.com", 
                password="Zjds5wm7!@#$%&*",                         
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

            tags = [tag['ExternalId'] for tag in tags['OutputTags']]
        
        else:
            print('Cannot open this type of file!')

            tags = None

        return tags
    
    def read_outputs(
            self,
            path : str
        ):

        # returns a dict from OutputTags
        
        file_extension = os.path.splitext(path)[1]

        if file_extension == '.txt':
            with open(path, 'r') as f:
                tags = [linha.strip() for linha in f.readlines()]

        elif file_extension == '.json':

            with open(path, 'r') as file:
                tags = json.load(file)

            tags = tags['OutputTags']
        
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

            rename_dict = {f'{tag["ExternalId"]}': f'{tag["Alias"]}' for tag in tags['OutputTags']}

            data.rename(columns = rename_dict, inplace = True)
            data.rename_axis('date', inplace = True)

        else:
            data.rename_axis('date', inplace = True)
            data = data

        return data
    
    def get_retrieve_output_tag(self):
        # return self.tags
        return self.output_tags
    
    def create_and_save_time_series_data(self,
                                     data: pd.DataFrame,
                                     column_name: str,
                                     unit: str,
                                     data_set_id: str,
                                     name: str = None,
                                     description: str = None,
                                     kpi_metadata: Dict[str, float] = None):
        """Function to create the time series and save the data

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe with the data to be saved in Cognite
        column_name : str
            Columns name
        unit : str
            unit of measurement
        data_set_id : str
            Dataset id
        kpi_metadata : Dict[str, float]
            KPI metadata.
        """
        
        client = self.client
        ts_external_id = column_name
        cdf_ts = client.time_series.retrieve(external_id=ts_external_id)
        
        if cdf_ts is None:
            if name is None:
                name = ts_external_id

            if kpi_metadata is None:
                ts = TimeSeries(external_id=ts_external_id,
                                name=name,
                                description=description,
                                unit=unit,
                                data_set_id=data_set_id)
            else:
                ts = TimeSeries(external_id=ts_external_id,
                                name=name,
                                description=description,
                                unit=unit,
                                data_set_id=data_set_id,
                                metadata=kpi_metadata)
            client.time_series.create(ts)
        else:
            cdf_ts.name = name
            cdf_ts.description = description
            cdf_ts.metadata = kpi_metadata
            client.time_series.update(cdf_ts)

        dps = []
        for index, r in data.iterrows():
            if math.isnan(r[column_name]) or math.isinf(r[column_name]): 
                continue
            dps= dps+[{"timestamp": r.name, "value": r[column_name]}]
        client.time_series.data.insert(datapoints = dps,external_id = ts_external_id)