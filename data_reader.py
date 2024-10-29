import pandas as pd 
import logging as log 
from dateutil.relativedelta import *
import streamlit as st

from connection import get_client_connection

@st.cache_data(ttl = '60min')
def read_sql(sql, env='prd'):
    """ Read data from Cognite Data Models using sql. 

    Args:
        sql (string): Sql query to run on Cognite and select the data needed. 

    Returns:
        df (Dataframe): Dataframe with the data read. 
    """

    client = get_client_connection(env)
    edges_query = client.transformations.preview(
        query=sql,
        limit=None,
        source_limit=None
    )
    log.info("Reading data...")
    df = pd.DataFrame(edges_query.results)
    log.info(f"Data read. Data shape: {df.shape}")
    return df

def read_movingAveragePrice(df):

    florence_plant_number = '3002'
    df['MaterialRequirement_externalId'] = f'MATR-{florence_plant_number}-' + df['Material']
    
    df_externalId = df[['MaterialRequirement_externalId', 'Material']].drop_duplicates()
    externalId_list = df_externalId['MaterialRequirement_externalId'].values
    log.info(f"len(externalId_list): {len(externalId_list)}")
    sql = f"""   select * from cdf_data_models("EDG-COR-ALL-DMD", "MaterialDOM", "2_2_1", "MaterialRequirement") materialRequirement
                WHERE materialRequirement.externalId IN {tuple(externalId_list)}
    """
    df_movingAveragePrice = read_sql(sql=sql)
    df_movingAveragePrice = df_movingAveragePrice[["externalId", "movingAveragePrice"]].drop_duplicates()
    df_movingAveragePrice = df_movingAveragePrice.rename(columns={"externalId": "MaterialRequirement_externalId"})
    df_with_movingAveragePrice = pd.merge(df, df_movingAveragePrice, on="MaterialRequirement_externalId")
    try:
        df_with_movingAveragePrice['movingAveragePrice_bylbs'] = df_with_movingAveragePrice['movingAveragePrice'] / df_with_movingAveragePrice["planQuantityWeight"] 
    except:
        df_with_movingAveragePrice['movingAveragePrice_bylbs'] = df_with_movingAveragePrice['movingAveragePrice'] / df_with_movingAveragePrice["PlanQty (lbs)"] 

    return df_with_movingAveragePrice

def read_ReportingProduction(START_DATE, END_DATE):
    """ Get ReportingProduction data from Cognite for the data between the passed dates: START_DATE and END_DATE. 
    
    Args: 
        START_DATE (string): String date in the format: YYYY-MM-DD HH:MM:SS to be used in sql where clause.'   
        END_DATE (string): String date in the format: YYYY-MM-DD HH:MM:SS to be used in sql where clause'

    Returns:
        df (Dataframe): data frame with the values for ReportingProduction with actualStartDate between START_DATE and END_DATE.
    """
    
    sql = f"""
        select *
        from cdf_data_models("PCC-COR-ALL-DML", "ProductionConversionCostSOL", "1_3_3", "ReportingProduction") production
        where production.plant.externalId = 'PLANT-3002'
            and production.actualStartDate > '{START_DATE}'
            and production.actualStartDate < '{END_DATE}'
        """
    log.info(f"reading data...")
    df = read_sql(sql)
    log.info(f"Data read. Data shape: {df.shape}")
    return df 