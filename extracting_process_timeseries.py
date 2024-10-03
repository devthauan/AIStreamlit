import pandas as pd
import os
import json
import math
import atexit

from msal import SerializableTokenCache, PublicClientApplication
from cognite.client.credentials import Token
from cognite.client import CogniteClient, ClientConfig
from cognite.client.credentials import Token
from typing import Dict
from cognite.client.data_classes import TimeSeries

class ExtractProcessTimeseries():

    def __init__(self, 
                env : str = 'prod',
                path : str = '', 
                client_name : str = '',
                base_url : str = '',
                authority : str = '',
                client_id : str = '',
                scopes : str = '',
                username : str = '',
                password : str = ''
                 ):

        """
        Constructor of the ExtractionProcessTimeseries class.
        Initializes an instance of the class, but does not perform any operations.
        """

        self.env = env
        self.path = path
        self.client_name = client_name
        self.base_url = base_url
        self.authority = authority
        self.client_id = client_id
        self.scopes = scopes
        self.username = username
        self.password = password
        self.input_tags = self.read_inputs()
        self.output_tags = self.read_outputs()

        # Initialize OpenAI client
        self.client = self.create_cognite_client()

    def create_cognite_client(self):
        
        """
        Connection information to the extraction environment.

        Parameters
        ----------

        Returns
        -------
            Connection information to access the database.
        """

        creds = self._get_credentials()
        token_credentials = Token(creds["access_token"])
        env = self.env
        client_name = self.client_name
        base_url = self.base_url

        project = 'celanese'
        if env == 'dev':
            project += f'-{env}'

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
        authority  = self.authority
        client_id = self.client_id
        username = self.username
        password = self.password
        scopes = [self.scopes]
        port = "0"
        app = PublicClientApplication(
            client_id=client_id, authority=authority, token_cache=self._create_cache()
        )
        accounts = app.get_accounts()
        if accounts:
            return app.acquire_token_silent(scopes, account=accounts[0])
        
        return app.acquire_token_by_username_password(
                username = username, 
                password = password,                         
                scopes = scopes
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
                start_time : pd.Timestamp = pd.to_datetime('2024-01-01'),
                end_time : pd.Timestamp = pd.to_datetime('2025-01-01'),
                freq : str = '1m',
                aggregates : str = 'average', # Could be 'interpolation'
                fillna : bool = True,
                in_out : str = 'inputs'
                ):
        """
        Extracts the data based on a list of external ids.

        Parameters
        ----------
        start_time (pd.Timestamp):
            Extraction start date.
        end_time (pd.Timestamp):
            Extraction end date.
        freq (str):
            The frequency of data to be extradited.
        aggregates (str):
            How the missing data will be filled.
        fillna (bool):
            Specify whether missing data will be filled with 
            the last timeseries value.
        in_out (str):
            Specify whether the extracted data will be inputs or outputs.

        Returns
        -------
        pd.DataFrame
            Dataframe with the time series extracted.
        """

        if in_out == 'inputs':

            tags = self.input_tags 
        
        else:
            tags = self.output_tags
            tags = [tag['ExternalId'] for tag in tags]

        client = self.client

        data = client.time_series.data.retrieve_dataframe(
            external_id  = tags,
            aggregates   = aggregates, 
            granularity  = freq, 
            start        = start_time, 
            end          = end_time, 
            include_aggregate_name = False,
        )

        if(len(data)==0):

            data = self.rename_columns(data, in_out)
            return data
        
        if fillna:

            for column in data.columns:
                if(data[column].iloc[0] is None or math.isnan(data[column].iloc[0])):
                    data[column].iloc[0] = 0.0
                    value = client.time_series.data.retrieve_latest(external_id=column,before=data[column].index[0]).value
                    if len(value) != 0:
                        if not math.isnan(value[0]):
                            data[column].iloc[0] = value[0]    
                data[column]  = data[column].ffill()

        data = self.rename_columns(data, in_out)

        return data
    
    def read_inputs(
            self,
        ):
        
        """
        Opens the file that contains the external IDs 
        and creates a list of all the variables to be read in cognite.

        Parameters
        ----------

        Returns
        -------
        list:
            A list of all the External IDs to be read in cognite.
        """

        path = self.path
        file_extension = os.path.splitext(path)[1]

        if file_extension == '.txt':
            with open(path, 'r') as f:
                input_tags = [linha.strip() for linha in f.readlines()]

        elif file_extension == '.json':

            with open(path, 'r') as file:
                input_tags = json.load(file)

            input_tags = [tag['ExternalId'] for tag in input_tags['InputTags']]
        
        else:
            print('Cannot open this type of file!')

            input_tags = None

        return input_tags
        
    def get_retrieve_output_tag(self):
        # return self.tags
        return self.output_tags
    
    def read_outputs(
            self,
        ):

        """
        Reads and returns the output tags from a file, based on the file type (.txt or .json).

        Parameters
        -----------

        Returns
        --------
        dict or list
            - If the file is a `.txt`, it returns a list of tags, where each line in the file is considered a tag.
            - If the file is a `.json`, it extracts the 'OutputTags' key from the JSON file and returns its value (expected to be a dictionary or list).
            - Returns `None` if the file type is unsupported.
        """

        path = self.path

        # returns a dict from OutputTags
        
        file_extension = os.path.splitext(path)[1]

        if file_extension == '.txt':
            with open(path, 'r') as f:
                output_tags = [linha.strip() for linha in f.readlines()]

        elif file_extension == '.json':

            with open(path, 'r') as file:
                output_tags = json.load(file)

            if "OutputTags" in output_tags:
                output_tags = output_tags['OutputTags']
            else:
                output_tags = None
        
        else:
            print('Cannot open this type of file!')

            output_tags = None

        return output_tags
    
    def rename_columns(
            self,
            data : pd.DataFrame,
            in_out : str = 'inputs'
        ):

        """
        Rename the columns of the extracted dataframe,
        if the file with the external IDs is .json type
        with EXTERNALID and NAME.

        Exemple: 
            "InputTags": [
                    {
                        "ExternalId": "CL-UTL:TI30192.PV",
                        "Name": "BFWTemp"
                    }

        Parameters
        ----------

        data (pd.DataFrame):
            Dataframe with the columns to be renamed.
        in_out (str):
            Specify whether the extracted data will be inputs or outputs.

        Returns
        -------
        pd.DataFrame:
            Dataframe with the columns renamed, if the file path is
            .json type.
        """

        path = self.path

        file_extension = os.path.splitext(path)[1]

        if file_extension == '.json':

            with open(path, 'r') as file:
                tags = json.load(file)

            if in_out == 'inputs':
                rename_dict = {f'{tag["ExternalId"]}': f'{tag["Name"]}' for tag in tags['InputTags']}

            else:
                rename_dict = {f'{tag["ExternalId"]}': f'{tag["Name"]}' for tag in tags['OutputTags']}
                
            data.rename(columns = rename_dict, inplace = True)
            data.rename_axis('timestamp', inplace = True)

        else:
            data.rename_axis('timestamp', inplace = True)

        return data
    
    def create_and_save_time_series_data(
            self,
            data: pd.DataFrame,
            column_name: str,
            unit: str,
            data_set_id: str,
            name: str = None,
            description: str = None,
            kpi_metadata: Dict[str, float] = None
                                     ):
        
        """
        Function to create the time series and save the data

        Parameters
        ----------
        data (pd.DataFrame): 
            Dataframe with the data to be saved in Cognite
        column_name (str): 
            Columns name
        unit (str): 
            Unit of measurement
        data_set_id (str): 
            Dataset id
        name (str):
            Timeserie KPICode
        description (str):
            Timeserie description
        kpi_metadata (Dict[str, float]): 
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