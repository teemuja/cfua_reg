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
    s3_url = f"{st.secrets['allas']['allas_url']}/CFUA/RD/cfua_data_nordics.parquet"
    @st.cache_data()
    def load_db(s3_url):
        duckdb.sql("INSTALL httpfs;")
        duckdb.sql("LOAD httpfs;")
        column_name = 'fua_name'
        fua_names_query = f"SELECT DISTINCT {column_name} FROM read_parquet('{s3_url}');"
        fuas = duckdb.query(fua_names_query).to_df()[column_name].tolist()
        return fuas
    fuas = load_db(s3_url)
    dbstatus.update(label="DB connected!", state="complete", expanded=False)

s1,s2 =st.columns(2)
selected_fua = [s1.selectbox("Select case area", fuas)] #take [] away if using multiselect
df = None
getit = st.button("Fetch CFUA data")
if getit:
    if selected_fua:
        @st.cache_data()
        def get_cfua_data(selected_fuas,column='fua_name'):
            if len(selected_fuas) > 1:
                # Convert list of selected values into a SQL IN clause format
                values_str = ', '.join(f"'{v}'" for v in selected_fuas)
                filter_query = f"""
                SELECT * 
                FROM read_parquet('{s3_url}') 
                WHERE {column} IN ({values_str})
                """
            else:
                value_str = f"'{selected_fuas[0]}'"
                filter_query = f"""
                SELECT * 
                FROM read_parquet('{s3_url}') 
                WHERE {column} = {value_str}
                """
                
            df = duckdb.query(filter_query).to_df()
            cols = df.columns.to_list()[:-2]
            return df[cols]
        df = get_cfua_data(selected_fuas=selected_fua)
        df = df.dropna(subset="h3_10").set_index("h3_10")
        #st.data_editor(df)
        #st.stop()
    else:
        st.warning('Not enough memory..')
        st.stop()

if df is None:
    st.stop()
    
# now we need to aggregate and classify the cfua_data
# to get landuse features of neighborhoods in different sizes for each record

#dict for custom naming the landuse values used as a integers in cfua__data
lu_dict = {
        0:"other",
        1:"dense", #"Discontinuous dense urban fabric (S.L. : 50% -  80%)": 1,
        2:"compact", #"Discontinuous medium density urban fabric (S.L. : 30% - 50%)": 2,
        3:"continuous", #"Continuous urban fabric (S.L. : > 80%)": 3,
        4:"sprawled",#"Discontinuous very low density urban fabric (S.L. : < 10%)": 4,
        5:"facilities", #"Industrial, commercial, public, military and private units": 5,
        6:"leisure_sport", #"Sports and leisure facilities": 6,
        7:"green_area", #"Green urban areas": 7,
        8:"forest", #"Forests": 8,
        9:"open_nature"
        # combine following classes under '9'..
        #"Open spaces with little or no vegetation (beaches, dunes, bare rocks, glaciers)": 9,
        #"Herbaceous vegetation associations (natural grassland, moors...)": 9
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


def aggregation(df):
    #bin age
    def binit(df_in,bins = [0, 25, 50, 100]):
        df = df_in.copy()
        bins = [0, 25, 50, np.inf]
        names = ["young", "adult", "senior"]
        df['Age'] = pd.cut(df['Age'], bins, labels=names)
        return df
    df = binit(df)
    with st.status('Aggregating landuse and services..') as aggstatus:
        #we first classify service sdi in 'activity_class' column for all hexas in the df
        df_v2 = regs.classify_service_diversity(df, service_col='services', categories_dict=retail_categories)
        nd_size_km = st.slider("set radius for the neighborhood to use (km)",1,9,1,step=2)
        nd_size_dict = {1:7,3:20,5:33,7:47,9:60}
        nd_size_r = nd_size_dict[nd_size_km]
        #then we count hexas with different landuse types in the nd of the size set above
        df_for_reg = regs.count_landuse_types_in_nd(df_in=df_v2, k=nd_size_r, lu_dict=lu_dict)
        #st.data_editor(df)
        #st.data_editor(df_v2)
        #st.data_editor(df_for_reg)
        #st.stop()
        del df
        del df_v2
        st.data_editor(df_for_reg.describe())
        st.info(f"Memory usage of the data {round(df_for_reg.memory_usage(deep=True).sum() / (1024**2),1)} MB")
        aggstatus.update(label="Aggregation ready!", state="complete", expanded=False)
    return df_for_reg

df_for_reg = aggregation(df)

with st.expander('Regression',expanded=True):
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
    lu_cols = [
            'lu_malls',
            'lu_continuous',
            'lu_facilities',
            'lu_dense',
            'lu_green_area',
            'lu_leisure_sport',
            'lu_compact',
            'lu_open_nature',
            'lu_forest']
    
    
    @st.fragment()
    def generate_regs(df_for_reg):
        with st.form('reg'):
            s1,s2,s3 = st.columns(3)
            target_col=s1.selectbox("Target domain",cf_cols)
            cat_cols = ['Age','Education level', 'Household per capita income decile', 'Household type']#,'Urban degree']
            control_cols = s2.multiselect('Control cols',cat_cols,default='Household per capita income decile')
            base_cols = ['Age','Education level','Household type']
            ext_cols = lu_cols
            gen = st.form_submit_button('Generate')
            
        if gen:
            reg_df = regs.ols_reg_table(df_for_reg,target_col,base_cols,cat_cols,ext_cols,control_cols).fillna('-')
            #st.text(f"cat_cols: {cat_cols}")
            #st.text(f"base_cols: {base_cols}")
            #st.text(f"ext_cols: {ext_cols}")
            st.data_editor(reg_df,use_container_width=True, height=700)

    generate_regs(df_for_reg=df_for_reg)
    
