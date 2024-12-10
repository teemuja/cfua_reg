import streamlit as st
import pandas as pd
import numpy as np
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
import boto3

import utils

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

with st.status('Connecting database..') as dbstatus:
    st.caption(f'Duckdb v{duckdb.__version__}')
    s3_url = f"{st.secrets['allas']['allas_url']}/CFUA/RD/cfua_data_nordics_merged_V2_N2119.parquet"
    #not merged file: cfua_data_with_landuse_nordics_cf_N1827.parquet"
    
    @st.cache_data()
    def load_db(s3_url):
        duckdb.sql("INSTALL httpfs;")
        duckdb.sql("LOAD httpfs;")
        #column_name = 'fua_name'
        #fua_names_query = f"SELECT DISTINCT {column_name} FROM read_parquet('{s3_url}');"
        all_query = f"SELECT * FROM read_parquet('{s3_url}');"
        #fuas = duckdb.query(all_query).to_df()[column_name].tolist()
        cfua = duckdb.query(all_query).to_df()
        return cfua #fuas
    
    cfua_data = load_db(s3_url).drop(columns=['__index_level_0__'])
    
    #package lu_cols
    #lu_cols_orig = [col for col in cfua_data.columns if col.startswith('lu')]
    def package_cols(df,target_cols=["lu_other", "lu_nan"],new_col_name="lu_unknown"):
        suffixes = ["R1", "R3", "R5", "R7", "R9"]
        for suffix in suffixes:
            # Find columns matching the current suffix and target groups
            matching_cols = [
                col for col in df.columns
                if any(col.startswith(group) and col.endswith(suffix) for group in target_cols)
            ]
            # Sum the selected columns row-wise and assign to a new column
            if matching_cols:  # Only if matching columns exist
                df[f"{new_col_name}_{suffix}"] = df[matching_cols].sum(axis=1)
            #drop targets
            for col in target_cols:
                drop_col = f"{col}_{suffix}"
                df = df.drop(columns=drop_col)
        return df
    
    cfua_data = package_cols(cfua_data,
                             target_cols=["lu_continuous","lu_discont_high"],
                             new_col_name="lu_urban_fabric")
    
    cfua_data = package_cols(cfua_data,
                             target_cols=["lu_discont_med","lu_discont_low"],
                             new_col_name="lu_suburban_fabric")
    
    # cfua_data = package_cols(cfua_data,
    #                          target_cols=["lu_discont_low"],
    #                          new_col_name="lu_periurban_fabric")
    
    cfua_data = package_cols(cfua_data,
                             target_cols=["lu_shopping_retail","lu_food_dining"],
                             new_col_name="lu_consumer_services")
    
    cfua_data = package_cols(cfua_data,
                             target_cols=["lu_leisure_landuse","lu_green_areas"],
                             new_col_name="lu_green_and_recreation")
    
    # cfua_data = package_cols(cfua_data,
    #                          target_cols=["lu_diversity"],
    #                          new_col_name="lu_high_diversity")
    
    cfua_data = package_cols(cfua_data,
                             target_cols=["lu_diversity","lu_facility_landuse","lu_other", "lu_nan"],
                             new_col_name="lu_unknown")
    
    drop_lu_unknowns = [col for col in cfua_data.columns if col.startswith('lu_unknown')]
    cfua_data = cfua_data.drop(columns=drop_lu_unknowns)
    
    cf_cols = [
            'Total footprint',
            'Housing footprint',
            'Vehicle possession footprint',
            'Public transportation footprint',
            'Leisure travel footprint',
            'Goods and services footprint',
            'Pets footprint',
            'Summer house footprint',
            'Diet footprint',
            ]
    s1,s2,s3 = st.columns(3)
    cluster = s1.select_slider("Clustering using median aggregation on h3 level",['No clustering',9,8,7])
    if cluster != "No clustering":
        def agg_cf_values(df_in,r=9):
            #drop these when clustered
            drop_cols = ['Country','Number of persons in household','Car in household']
            df = df_in.drop(columns=drop_cols)
            
            df[f'h3_0{r}'] = df["h3_10"].apply(lambda x: h3.cell_to_parent(x, r))
            all_cols = df.select_dtypes(include=['number']).columns.tolist()
            aggregated_df = df.groupby(f'h3_0{r}')[all_cols].median().reset_index()
            #add new 10 level id using centers of aggregated 09 cells
            aggregated_df['new_h3_10'] = aggregated_df[f'h3_0{r}'].apply(lambda x: h3.cell_to_center_child(x, 10))
            df_out = df.merge(aggregated_df, on=f'h3_0{r}', suffixes=('', '_agg'))
            for col in all_cols:
                df_out[col] = df_out[f"{col}_agg"]
                df_out = df_out.drop(columns=[f"{col}_agg" ])
            #replace h3_10 with new centers of aggregation
            df_out['h3_10'] = df_out['new_h3_10']
            #df_out = df_out.drop_duplicates(subset=['h3_10'])
            df_out = df_out.groupby('h3_10', as_index=False).agg('first')
            return df_out.drop(columns=[f'h3_0{r}','new_h3_10'])
        cfua_data = agg_cf_values(df_in=cfua_data,r=cluster)
        
    st.data_editor(cfua_data.describe())
    #st.data_editor(cfua_data)
    total_n = len(cfua_data[cfua_data['Total footprint'] != 0])
    st.info(f"Nordic CF data N{total_n}. Memory usage {round(cfua_data.memory_usage(deep=True).sum() / (1024**2),1)} MB")
    
    dbstatus.update(label="DB connected!", state="complete", expanded=True)

cities = cfua_data['fua_name'].unique().tolist()
s1,s2,s3 = st.columns(3)
target_cities = s1.multiselect("Target Cities",cities,default="Helsinki")
target_col = s2.selectbox("Target domain",cf_cols)
if cluster == "No clustering":
    cat_cols = ['Household per capita income decile', 'Household type', 'Car in household']
    base_cols = ['Household type', 'Car in household','Age']
else:
    cat_cols = ['Household per capita income decile', 'Household type']
    base_cols = ['Household type', 'Age']
    
control_cols = s3.multiselect('Control cols',cat_cols,default='Household per capita income decile')
    
my_reg_results = None
if len(target_cities) > 0:
    
    #filter with targets
    cfua_data_for_city = cfua_data[cfua_data['fua_name'].isin(target_cities)]

    #normalize for reg
    def normalize_df(df_in,cols=None):
        df = df_in.copy()
        if cols is None:
            cols = df.select_dtypes(include=np.number).columns.tolist()
        for col in cols:
            df[col] = df[col] / df[col].abs().max()
        return df
    
    df_for_reg_normalized = normalize_df(cfua_data_for_city)
    
    #st.markdown("normalized...")
    #st.data_editor(df_for_reg_normalized.describe())
    lu_cols = [col for col in df_for_reg_normalized.columns if col.startswith('lu')]
    
    my_reg_results = utils.ols_reg_table(df_for_reg_normalized,target_col,base_cols,cat_cols,lu_cols,control_cols)

    with st.expander(f"Sample {target_cities} with N{len(cfua_data_for_city)} on {target_col} , Cluster reso: {cluster}", expanded=True):
        #simple_chart_data = utils.prepare_chart_data(my_reg_results)
        #st.area_chart(data=simple_chart_data,x_label='Radius',y_label='ext_r Value') 
        @st.fragment()
        def plotter(df):
            plot_holder = st.empty()
            p1,p2 = st.columns(2)
            if p1.toggle('P-value limit'):
                p_mean = my_reg_results['ext_p'].mean()
                p = p2.slider('P-value limit (sample mean as max)',0.05,p_mean,p_mean,step=0.01)
                p1.caption("Graph becomes fragmented if any radius(x-axis) is out of P-value limit.")
            else:
                p = 1
            fig = utils.prepare_data_for_plotly_chart(df,p_limit=p)
            with plot_holder:
                st.plotly_chart(fig, use_container_width=True)
        plotter(my_reg_results)

            
    with st.expander(f'Regression table {target_cities}', expanded=False):    
        st.data_editor(my_reg_results,use_container_width=True, height=900)
        reg_csv_to_save = my_reg_results.to_csv().encode('utf-8')
        target_domain = target_col.lower().replace(' ', '_')
        target_cases = '_'.join(target_cities).lower()
        file_name = f"CFUAregs_{target_domain}_{target_cases}.csv"
        st.download_button(label="Save as CSV",
                            data=reg_csv_to_save,
                            file_name=file_name,
                            mime='text/csv')

    with st.expander(f'Partial correlation {target_cities}', expanded=False):
        target_lu = st.selectbox('Select land-use target by radius',lu_cols)
        df_part = utils.partial_corr(df=df_for_reg_normalized,cf_cols=cf_cols,corr_target=target_lu,covar=control_cols)
        st.data_editor(df_part,use_container_width=True)
        st.caption('https://pingouin-stats.org/build/html/generated/pingouin.partial_corr.html , https://en.wikipedia.org/wiki/Partial_correlation')

    with st.expander(f'Sample data {target_cities}', expanded=False):    
            st.data_editor(cfua_data_for_city.describe(),use_container_width=True)
            csv_to_save = cfua_data_for_city.to_csv().encode('utf-8')
            file_name = f"CFUA_{target_cities}.csv"
            st.download_button(label="Save as CSV",
                                data=csv_to_save,
                                file_name=file_name,
                                mime='text/csv')