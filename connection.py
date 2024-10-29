import os 
import sys 
import streamlit as st
import logging as log 

ROOT_PATH = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(ROOT_PATH)
import authentication


CLIENT_NAME = st.secrets['CLIENT_NAME_PCC']
PROJECT     = st.secrets['PROJECT']
PROJECT_DEV = st.secrets['PROJECT_DEV']
BASE_URL    = st.secrets['BASE_URL']

def get_client_connection(env='prd'):
    """ Gets Cognite client connection.

    Returns:
        client: Cognite client connection that will be used to read data from Cognite. 
    """

    log.info(f"Getting client connection for env: {env}...")
    if(env=='dev'):
        client = authentication.CogniteDataExtractionRequest.create_client(client_name=CLIENT_NAME, project=PROJECT_DEV, base_url=BASE_URL)
    else:
        client = authentication.CogniteDataExtractionRequest.create_client(client_name=CLIENT_NAME, project=PROJECT, base_url=BASE_URL)
    log.info("Connection successful.")
    
    return client