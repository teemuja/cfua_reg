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

import regs

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
    s3_url = f"{st.secrets['allas']['allas_url']}/CFUA/RD/cfua_data_with_landuse_nordics_cf_N1827.parquet"
    
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
    
    cfua_data = load_db(s3_url).drop(columns=['__index_level_0__','lat','lng'])
    total_n = len(cfua_data[cfua_data['Total footprint'] != 0])
    st.info(f"Nordic CF data N{total_n}. Memory usage {round(cfua_data.memory_usage(deep=True).sum() / (1024**2),1)} MB")
    s1,s2 =st.columns(2)
    s1.data_editor(cfua_data)
    s2.text(cfua_data.dtypes)
    #st.text(cfua_data.isna().sum())
    dbstatus.update(label="DB connected!", state="complete", expanded=False)



#dict for custom naming the landuse values used as a integers in cfua__data

lu_dict = {
        0:"other",
        
        # city classes
        3:"compact", #"Continuous urban fabric (S.L. : > 80%)": 3,
        
        1:"fragments_dense", #"Discontinuous dense urban fabric (S.L. : 50% -  80%)": 1,
        2:"fragments_airy", #"Discontinuous medium density urban fabric (S.L. : 30% - 50%)": 2,
        
        4:"sprawled", #"Discontinuous low density urban fabric (S.L. : 10% - 30%)": 4
        5:"sprawled",#"Discontinuous very low density urban fabric (S.L. : < 10%)": 5,
        
        6:"facilities", #"Industrial, commercial, public, military and private units": 6,
        7:"leisure_sport", #"Sports and leisure facilities": 7,
        8:"green_area", #"Green urban areas": 8,
        9:"forest", #"Forests": 9,
        10:"open_nature"
        # combine following classes under '10'..
        #"Open spaces with little or no vegetation (beaches, dunes, bare rocks, glaciers)": 10,
        #"Herbaceous vegetation associations (natural grassland, moors...)": 10
        }

#categories for services
retail_categories = {
        "Food & Dining": {
            "restaurant", "cafe", "bar", "pizza_restaurant", "coffee_shop", 
            "bakery", "italian_restaurant", "thai_restaurant", "sushi_restaurant",
            "fast_food_restaurant", "wine_bar", "burger_restaurant", 
            "cocktail_bar", "pub", "food_truck", "ice_cream_shop", "diner",
            "buffet", "steakhouse", "seafood_restaurant", "vegetarian_restaurant",
            "breakfast_restaurant", "food_delivery", "food_stand", "cafeteria"
        },
        "Shopping & Retail": {
            "shopping", "clothing_store", "jewelry_store", "grocery_store", 
            "furniture_store", "flowers_and_gifts_shop", "bookstore", "retail",
            "shoe_store", "antique_store", "electronics", "sporting_goods",
            "home_improvement_store", "department_store", "convenience_store",
            "supermarket", "boutique", "thrift_store", "gift_shop", "market"
        },
        "Health & Wellness": {
            "hospital", "counseling_and_mental_health", "gym", "spas", "dentist",
            "naturopathic_holistic", "health_and_medical", "yoga_studio",
            "physical_therapy", "psychotherapist", "psychologist", "fitness_trainer",
            "acupuncture", "chiropractor", "medical_center", "clinic",
            "mental_health", "fitness_center", "wellness_center", "pharmacy"
        },
        "Professional Services": {
            "professional_services", "real_estate", "software_development",
            "advertising_agency", "lawyer", "architectural_designer",
            "graphic_designer", "financial_service", "marketing_agency",
            "contractor", "interior_design", "insurance_agency", "accounting",
            "consulting", "bank", "legal_services", "business_consultant",
            "web_design", "property_management", "recruitment"
        },
        "Arts & Entertainment": {
            "art_gallery", "theatre", "arts_and_entertainment", "music_venue",
            "movie_theatre", "museum", "comedy_club", "dance_club",
            "concert_venue", "performing_arts_venue", "bowling_alley", "casino",
            "arcade", "amusement_park", "cinema", "nightclub", "art_museum",
            "cultural_center", "art_school", "entertainment_center"
        },
        "Education": {
            "school", "college_university", "education", "library", "preschool",
            "language_school", "art_school", "music_school", "high_school",
            "elementary_school", "tutoring_center", "educational_services",
            "vocational_school", "dance_school", "driving_school"
        },
        "Beauty & Personal Care": {
            "beauty_salon", "hair_salon", "nail_salon", "barber", "spa",
            "massage", "tanning_salon", "cosmetics", "beauty_supply",
            "hair_care", "skin_care", "makeup_artist", "personal_care"
        },
        "Community & Religious": {
            "community_services_non_profits", "religious_organization", 
            "church_cathedral", "mosque", "temple", "synagogue", 
            "community_center", "social_services", "charity_organization",
            "place_of_worship"
        },
        "Other": set()  # This will catch any uncategorized amenities
    }

@st.cache_data(max_entries=1) #@st.fragment()
def aggregation(df,nd_size_km,retail_categories,lu_dict):
    #we first classify service sdi in 'activity_class' column for all hexas in the df
    df_v2 = regs.classify_service_diversity(df, service_col='services', categories_dict=retail_categories)
    #..count hexas with different landuse types in the nd of the size set above
    df_for_reg = regs.count_landuse_types_in_nd(df_in=df_v2, r=nd_size_km, lu_dict=lu_dict)
    return df_for_reg


with st.container():
    #cols
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
    lu_cols = ['lu_malls',
                'lu_fragments_dense',
                'lu_compact',
                'lu_green_area',
                'lu_leisure_sport',
                'lu_fragments_airy',
                'lu_facilities',
                'lu_forest',
                'lu_sprawled',
                'lu_open_nature'
              ]
    with st.form('reg'):
        cities = cfua_data['fua_name'].unique().tolist()
        s1,s2,s3 = st.columns(3)
        target_cities = s1.multiselect("Target Cities",cities,default="Helsinki")
        target_col = s2.selectbox("Target domain",cf_cols)
        cat_cols = ['Household per capita income decile', 'Household type'] #,'Urban degree']
        control_cols = s3.multiselect('Control cols',cat_cols,default='Household per capita income decile')
        nd_size_km = st.slider("set radius for the neighborhood to use (km)",1,9,3,step=2)
        base_cols = ['Household type','Age']
        ext_cols = lu_cols
        gen = st.form_submit_button('Generate')
    
    my_reg_results = None
    if gen and len(target_cities) > 0:
        cfua_data_for_city = cfua_data[cfua_data['fua_name'].isin(target_cities)]
        
        #agg here only for the city df
        with st.status('Aggregating landuse and services..') as aggstatus:
            df_for_reg = aggregation(cfua_data_for_city,nd_size_km,retail_categories,lu_dict)
            st.data_editor(df_for_reg.describe(),height=200)
            st.info(f"Memory usage of the data {round(df_for_reg.memory_usage(deep=True).sum() / (1024**2),1)} MB")
            aggstatus.update(label="Aggregation ready!", state="complete", expanded=False)
        
        dropcols = ['fua_name','services']
        df_for_reg_cleaned = df_for_reg[df_for_reg['Total footprint'] != 0].drop(columns=dropcols)
        st.subheader(f"**Regression table**")
        st.caption(f"{target_cities} with N{len(df_for_reg_cleaned)}")
        
        #normalize for reg
        def normalize_df(df_in,cols=None):
            df = df_in.copy()
            if cols is None:
                cols = df.select_dtypes(include=np.number).columns.tolist()
            for col in cols:
                df[col] = df[col] / df[col].abs().max()
            return df
        
        df_for_reg_normalized = normalize_df(df_for_reg_cleaned)
        
        #st.data_editor(df_for_reg_cleaned)
        #st.markdown("normalized...")
        #st.data_editor(df_for_reg_normalized)
        #st.stop()
        my_reg_results = regs.ols_reg_table(df_for_reg_normalized,target_col,base_cols,cat_cols,ext_cols,control_cols)
        st.data_editor(my_reg_results,use_container_width=True, height=900)

if my_reg_results is not None:
    def gen_merged(index_name='index'):
        dfs = []
        my_bar = st.progress(0, text="generating research data..")
        for r in [1,3,5,9]:
            dfr = aggregation(cfua_data_for_city,r,retail_categories,lu_dict)
            reg_r = regs.ols_reg_table(dfr,target_col,base_cols,cat_cols,ext_cols,control_cols)
            dfs.append(reg_r)
            my_bar.progress(r*10)

        cols = pd.MultiIndex.from_product([[df.columns[0] for df in dfs], df.columns[1:]], names=[index_name, ''])
        merged = pd.DataFrame(index=range(len(dfs[0])), columns=cols, dtype=float)
        for i, df in enumerate(dfs):
            for col in df.columns[1:]:
                merged[(df.columns[0], col)][i] = df[col]
        my_bar.progress(100)
        my_bar.empty()
        return merged

    cities = ""
    for c in target_cities:
        cities += c
    csv_to_save = my_reg_results.to_csv().encode('utf-8')
    file_name = f"REG_{cities}_{target_col}_R{nd_size_km}.csv"
    st.download_button(label="Save as CSV",
                        data=csv_to_save,
                        file_name=file_name,
                        mime='text/csv')

cf_cols = ['Housing footprint',
    'Vehicle possession footprint',
    'Public transportation footprint',
    'Leisure travel footprint',
    'Goods and services footprint',
    'Pets footprint',
    'Summer house footprint',
    'Total footprint']
#df_part = regs.partial_corr(df=df_for_reg_normalized,cf_cols=cf_cols,corr_target="lu_malls",covar=control_cols)
#st.data_editor(df_part,use_container_width=True, height=700)