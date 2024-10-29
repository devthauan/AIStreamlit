import sys 
import os
import streamlit as st
import atexit
import json

ROOT_PATH = os.path.abspath(os.path.join(os.getcwd()))
sys.path.append(ROOT_PATH)
print('ROOT_PATH:', ROOT_PATH)

from msal import SerializableTokenCache, PublicClientApplication
from cognite.client import CogniteClient, ClientConfig
from cognite.client.credentials import Token


AUTHORITY        = st.secrets['AUTHORITY']
CLIENT_ID        = st.secrets['CLIENT_ID']
CACHE_FILE_NAME  = st.secrets['CACHE_FILE_NAME']
COGNITE_USERNAME = st.secrets['COGNITE_USERNAME']
COGNITE_PASSWORD = st.secrets['COGNITE_PASSWORD']

class CogniteDataExtractionRequest:
    """
    Extract historic Cognite data
    """

    def create_client(
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

        creds = CogniteDataExtractionRequest._get_credentials()
        token_credentials = Token(creds["access_token"])

        return CogniteClient(
            config=ClientConfig(
                client_name=client_name,
                project=project,
                base_url=base_url,
                credentials=token_credentials,
            )
        )

    def _get_credentials():

        """
        Loading access credentials.

        Parameters
        ----------

        Returns
        -------
            Access credentials.
        """
        authority = AUTHORITY
        client_id = CLIENT_ID

        scopes = [
            "https://az-eastus-1.cognitedata.com/.default",
        ]
        port = "0"
        app = PublicClientApplication(
            client_id=client_id, authority=authority, token_cache=CogniteDataExtractionRequest._create_cache()
        )
        accounts = app.get_accounts()
        if accounts:
            return app.acquire_token_silent(scopes, account=accounts[0])
                
        return app.acquire_token_by_username_password(
                username=COGNITE_USERNAME, 
                password=COGNITE_PASSWORD,                         
                scopes=[
                    "https://az-eastus-1.cognitedata.com/.default",
                ]
        )

    def _create_cache():

        """
        Cache creating.

        Parameters
        ----------

        Returns
        -------
            Temporary storage space in the browser that saves data from websites you have already visited
        """
        cache_file_name = CACHE_FILE_NAME
        cache = SerializableTokenCache()

        if os.path.exists(cache_file_name):
            cache.deserialize(open(cache_file_name, "r").read())
        atexit.register(
            lambda: open(cache_file_name, "w").write(cache.serialize())
            if cache.has_state_changed
            else None
        )
        return cache