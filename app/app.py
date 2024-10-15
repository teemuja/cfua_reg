import streamlit as st
import pandas as pd 
import geopandas as gpd
from shapely import Point, wkt, ops
from shapely.wkt import loads, dumps
import h3 as h3
import math
import json
import ast
import geocoder

#DB con
import io
import requests
import duckdb
import ibis
#ibis: https://ibis-project.org/how-to/configure/basics
ibis.options.interactive = True
from ibis import _
import ibis.selectors as s
# https://ibis-project.org/posts/ibis-duckdb-geospatial/

import plotly.express as px
import plotly.graph_objects as go
px.set_mapbox_access_token(st.secrets['mapbox']['MAPBOX_TOKEN'])
my_style = st.secrets['mapbox']['MAPBOX_STYLE']

#page confs
title = "CFUA reg"
st.set_page_config(page_title=title, layout="wide")
st.header(title,divider="red")

def check_password():
    def password_entered():
        if (
            st.session_state["password"]
            == st.secrets["passwords"]["cfua"]
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False
    if "password_correct" not in st.session_state:
        st.text_input(label="password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input(label="password", type="password", on_change=password_entered, key="password")
        return False
    else:
        return True

auth = check_password()
if not auth:
    st.stop()

@st.cache_data()
def read_allas_file(filepath):
    allas_url = st.secrets['allas']['allas_url']
    r = requests.get(f"{allas_url}/{filepath}", stream=True)
    data = io.BytesIO(r.content)
    return data

with st.status('Connecting database..') as dbstatus:
    st.caption(f'Ibis v{ibis.__version__} / Duckdb v{ibis.__version__}')
    #con = ibis.duckdb.connect(threads=4, memory_limit="4GB") #fine tune depending on instance
    allas_url = st.secrets['allas']['allas_url']
    set_s3_endpoint = f"""
        SET s3_endpoint = '{allas_url}';
        SET s3_region = 'a3s';
        SET s3_access_key_id = '{st.secrets['allas']['allas_secret_key']}';
        SET s3_secret_access_key = '{st.secrets['allas']['allas_access_key']}';
    """
    #con.raw_sql(set_s3_endpoint)
    
    #st.cache_data(max_entries=1)
    #def set_view(_s3_path):
        #con.raw_sql(f"CREATE TEMPORARY TABLE temp_table AS SELECT * FROM parquet_scan('{_s3_path}');")
        #cfua_table = con.table('temp_table')
    #    return cfua_table
    #set_view(f"{allas_url}/RD/{case_file}.parquet")
    
    #case
    case_files = ['Helsinki','Stockholm','Copenhagen','Oslo','Reykjavik']
    case = st.selectbox('Use data for..',case_files)
    
    #ibis way
    case_file = f"cfua_data_{case}.parquet"
    sourcepath=f"{allas_url}/CFUA/RD/{case_file}"
    
    @st.cache_data()
    def read_case(sourcepath):
        df = ibis.read_parquet(sourcepath).to_pandas()
        return df
    
    df = read_case(sourcepath)
    st.data_editor(df)
    
    #mem ..whole Nordics is too heavy..
    st.info(f"Memory usage of the data {round(df.memory_usage(deep=True).sum() / (1024**2),3)} MB")
    
    dbstatus.update(label="DB connected!", state="complete", expanded=False)


st.stop()


# -------- utils functions -----------

# H3 -> https://nbviewer.org/github/uber/h3-py-notebooks/blob/master/notebooks/usage.ipynb
# https://github.com/uber/h3-py-notebooks/blob/master/notebooks/urban_analytics.ipynb

#dev on..

