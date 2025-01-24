# NDP app always beta a lot
import streamlit as st
import pandas as pd
import geopandas as gpd
import h3
import numpy as np
from shapely import wkt
from shapely.geometry import MultiPoint, Point
import math
import geocoder
from sklearn.cluster import DBSCAN
import duckdb
import requests
import io
import re
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import json
import utils
#ML
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

px.set_mapbox_access_token(st.secrets['mapbox']['MAPBOX_TOKEN'])
mbtoken = st.secrets['mapbox']['MAPBOX_TOKEN']
my_style = st.secrets['mapbox']['MAPBOX_STYLE']
cfua_allas = st.secrets['allas']['url']
allas_key = st.secrets['allas']['access_key_id']
allas_secret = st.secrets['allas']['secret_access_key']

st.set_page_config(page_title="Research App", layout="wide", initial_sidebar_state='expanded')
st.markdown("""
<style>
button[title="View fullscreen"]{
        visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# content
st.header("CFUA",divider="red")
st.subheader("Climate Friendly Urban Architecture")
st.markdown("###")

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
    
    @st.cache_data()
    def load_data(R=None):
        allas_url = f"{st.secrets['allas']['url']}"
        duckdb.sql("INSTALL httpfs;")
        duckdb.sql("LOAD httpfs;")
        if R is None:
            combined_table = f"""
                    DROP TABLE IF EXISTS combined_table;
                    CREATE TABLE combined_table AS 
                    SELECT *, 'R1' as R FROM read_parquet('{allas_url}/CFUA/RD/cfua_data_nordics_IQR1-5_R30-10_ND1km_no-sdi.parquet')
                    UNION ALL
                    SELECT *, 'R3' as R FROM read_parquet('{allas_url}/CFUA/RD/cfua_data_nordics_IQR1-5_R30-10_ND3km_no-sdi.parquet')
                    UNION ALL
                    SELECT *, 'R5' as R FROM read_parquet('{allas_url}/CFUA/RD/cfua_data_nordics_IQR1-5_R30-10_ND5km_no-sdi.parquet')
                    UNION ALL
                    SELECT *, 'R9' as R FROM read_parquet('{allas_url}/CFUA/RD/cfua_data_nordics_IQR1-5_R30-10_ND9km_no-sdi.parquet')
                """
            duckdb.sql(combined_table)
            df = duckdb.sql("SELECT * FROM combined_table").to_df().drop(columns=['__index_level_0__'])
            cols = ['R'] + [col for col in df.columns if col != 'R']
            df_out = df[cols]
        else:
            df_out = duckdb.sql(f"SELECT * FROM read_parquet('{allas_url}/CFUA/RD/cfua_data_nordics_IQR1-5_R30-10_ND{R}km_no-sdi.parquet')").to_df().drop(columns=['__index_level_0__'])
        
        #fix
        df_out['lu_recreational'] = df_out['lu_green_areas'] + df_out['lu_leisure_landuse']
        df_out['lu_consumerism'] = df_out['lu_shopping_and_retail'] + df_out['lu_food_and_dining']
        drop_cols = [
            'Diet footprint',
            'Pets footprint',
            'Summer house footprint',
            "lu_other",
            "amenity_sdi",
            "lu_facility_landuse",
            'lu_leisure_landuse',
            'lu_green_areas',
            'lu_food_and_dining',
            'lu_shopping_and_retail',
            'Country'
            ]
        
        cf_cols_to_use = [
            'Housing footprint',
            'Vehicle possession footprint',
            'Public transportation footprint',
            'Leisure travel footprint',
            'Goods and services footprint',
            'Total footprint',
            'Total footprint unit',
            ]

        df_out.rename(columns={col: 'cf_' + col for col in cf_cols_to_use}, inplace=True)

        return df_out.drop(columns=drop_cols)


    data = load_data()
    if st.toggle('Described'):
        st.data_editor(data.describe(),key="init_desc")
    else:
        # agg_h3 = st.radio("Set aggregation h3-level",[None,9,8],horizontal=True)
        # cf_cols = [col for col in data.columns if col.startswith('cf_')]
        # mean_cols = [col for col in data.columns if col.startswith('lu')]
        # mode_cols = [
        #     'Education level',
        #     'Household per capita income decile',
        #     'Household unit income decile',
        #     'Household type',
        #     'Car in household',
        #     'fua_name',
        #     'R',
        #     ]
        # if agg_h3 is not None:
        #     data_agg = pd.DataFrame()
        #     for r in ['R1','R3','R5','R9']:
        #         dfr = data[data['R'] == r]
        #         dfr_agg = utils.agg_values(df_in=dfr,
        #                                 cf_cols=cf_cols,
        #                                 mean_cols=mean_cols,
        #                                 mode_cols=mode_cols,
        #                                 r=agg_h3)
        #         data_agg = pd.concat([data_agg,dfr_agg])

        #     data = data_agg.copy()
            
        st.caption(f"N {len(data)}")
        st.data_editor(data,key="init")
        #st.stop()
        
        @st.dialog("Download data")
        def download(df):
            r = st.radio("Select which neigborhood radius data to download",[1,3,5,9],horizontal=True)
            dfr = df[df['R'] == f"R{1}"]
            st.download_button(
                                label=f"Download {r}km data as CSV",
                                data=dfr.to_csv().encode("utf-8"),
                                file_name=f"cfua_data_nordics_with_landuse_radius_{r}km.csv",
                                mime="text/csv",
                            )

        if st.button('Download'):
            download(df=data)
    cf_cols = [col for col in data.columns if col.startswith('cf_')]
    
    dbstatus.update(label="DB connected!", state="complete", expanded=False)

def normalize_df(df_in,cols=None):
    df = df_in.copy()
    if cols is None:
        cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in cols:
        df[col] = df[col] / df[col].abs().max()
    return df

#@st.cache_data()
def ols_reg_table(df_in, cf_col, base_cols, cat_cols, ext_cols, control_cols):
    df = normalize_df(df_in)
    def ols(df, cf_col, base_cols, cat_cols, ext_cols, control_cols):
        cat_cols_lower = [col.lower().replace(' ', '_') for col in cat_cols]
        
        def format_col(col):
            col_lower = col.lower().replace(' ', '_')
            return f'C({col_lower})' if col_lower in cat_cols_lower else col_lower
        
        domain_col = cf_col.lower().replace(' ', '_')
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        base_cols_str = ' + '.join([format_col(col) for col in base_cols])
        ext_cols_str = ' + '.join([format_col(col) for col in ext_cols])
        
        if control_cols is not None and len(control_cols) > 0:
            control_cols_str = ' + '.join([format_col(col) for col in control_cols])
            base_formula = f'{domain_col} ~ {base_cols_str} + {control_cols_str}'
            ext_formula = f'{domain_col} ~ {base_cols_str} + {ext_cols_str} + {control_cols_str}'
        else:
            base_formula = f'{domain_col} ~ {base_cols_str}'
            ext_formula = f'{domain_col} ~ {base_cols_str} + {ext_cols_str}'
        
        # Remove extra spaces for column selection
        formula_vars = re.split(r'\s*\+\s*', ext_formula.split('~')[1].strip())
        formula_vars = [col.strip() for col in formula_vars]
        
        base_model = smf.ols(formula=base_formula, data=df)
        base_results = base_model.fit()
        ext_model = smf.ols(formula=ext_formula, data=df)
        ext_results = ext_model.fit()

        return base_results, ext_results
    
    base_results, ext_results = ols(df, cf_col, base_cols, cat_cols, ext_cols, control_cols)
    rounder = 3
    reg_results = pd.DataFrame({
        'base_r': round(base_results.params, rounder),
        'base_p': round(base_results.pvalues, rounder),
        'ext_r': round(ext_results.params, rounder),
        'ext_p': round(ext_results.pvalues, rounder)
        })
    
    return reg_results

#reg_df = ols_reg_table(df_in, cf_col, base_cols, cat_cols, ext_cols, control_cols)
cities = data['fua_name'].unique().tolist()
lu_cols_in_use = [col for col in data.columns if col.startswith('lu')]
base_cols = ['Household type', 'Car in household','Age']

st1,st2 = st.columns(2)
target_cities = st1.multiselect("Set sample regions",cities,default=["Helsinki","Stockholm"])
if target_cities:
    datac = data[data['fua_name'].isin(target_cities)]
else:
    st.stop()


with st.expander(f'Reg.settings', expanded=True):
    s1,s2,s3 = st.columns(3)
    target_col = s1.selectbox("Target domain",cf_cols,index=6) #last
    if target_col == "Total footprint unit":
        cat_cols = ['Household unit income decile', 'Household type', 'Car in household']
    else:
        cat_cols = ['Household per capita income decile', 'Household type', 'Car in household']
    control_col = s2.selectbox('Control column',cat_cols,index=0)
    remove_cols = s3.multiselect('Remove landuse classes',lu_cols_in_use)
    if remove_cols:
        datac.drop(columns=remove_cols, inplace=True)
        norm_cols = list(set(cf_cols + lu_cols_in_use) - set(remove_cols))
    else:
        norm_cols = list(set(cf_cols + lu_cols_in_use))

#normalice case data for reg
datac = datac.dropna(subset=norm_cols)
datac = normalize_df(datac,cols=norm_cols)
#st.data_editor(datac)
#st.stop()

#@st.cache_data()
def gen_regs_for_plot(data,
                      target_col,
                      base_cols,
                      cat_cols,
                      lu_cols,
                      control_cols,
                      p_limit=False):
    radius_dfs = {}
    for r in [1,3,5,9]:
        df_r = data[data['R'] == f"R{r}"]
        reg_df = ols_reg_table(df_in=df_r,
                                cf_col=target_col,
                                base_cols=base_cols,
                                cat_cols=cat_cols,
                                ext_cols=lu_cols,
                                control_cols=control_cols
                                )
        #use only lu cols
        reg_lu_cols = [col for col in reg_df.T.columns if col.startswith('lu')]
        reg = reg_df.T[reg_lu_cols].T

        if p_limit:
            reg = reg[reg['ext_p'] < 0.07]

        radius_dfs[f"R{r}"] = reg
        
    return radius_dfs

def plot_ext_r_across_radii(dataframes_dict, regions, target_domain, p_value_threshold=0.07):

    # Convert radius keys to numeric values (removing 'R' prefix if exists)
    numeric_keys = {}
    for k in dataframes_dict.keys():
        if isinstance(k, str) and k.startswith('R'):
            numeric_keys[k] = float(k[1:])  # Convert 'R1', 'R5' etc to 1, 5
        else:
            numeric_keys[k] = float(k)  # Already numeric
            
    # Sort radii numerically
    sorted_radii = sorted(dataframes_dict.keys(), key=lambda x: numeric_keys[x])
    
    # Get all unique variables across all DataFrames
    all_variables = set()
    for df in dataframes_dict.values():
        all_variables.update(df.index.tolist())
    
    # Create figure
    fig = go.Figure()
    
    # Process data for each variable
    for var in all_variables:
        y_values = []
        x_values = []
        p_values = []
        x_labels = []
        
        # Collect valid points (where p-value meets threshold)
        for radius in sorted_radii:
            df = dataframes_dict[radius]
            if var in df.index:
                p_value = df.loc[var, 'ext_p']
                if p_value <= p_value_threshold:
                    x_values.append(numeric_keys[radius])
                    x_labels.append(str(radius))
                    y_values.append(df.loc[var, 'ext_r'])
                    p_values.append(p_value)
        
        # Only add trace if we have at least two valid points
        if len(x_values) >= 2:
            fig.add_trace(go.Scatter(
                x=x_values,
                y=y_values,
                name=var,
                mode='lines+markers',
                hovertemplate=(
                    'Radius: %{text}<br>' +
                    'ext_r: %{y:.3f}<br>' +
                    'p-value: %{customdata:.3f}<extra></extra>'
                ),
                text=x_labels,  # Original radius labels for hover
                customdata=p_values
            ))
    
    # Update layout
    fig.update_layout(
        title=f"Correlation between '{target_domain}' and landuse types in {regions} (p<0.07)",
        xaxis_title='Radius',
        yaxis_title='ext_r Value',
        hovermode='x unified',
        showlegend=True,
        legend_title='Variables',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        ),
        margin=dict(r=200)
    )
    
    # Update x-axis with proper tick labels
    fig.update_xaxes(
        tickmode='array',
        tickvals=[numeric_keys[r] for r in sorted_radii],
        ticktext=[str(r) for r in sorted_radii]
    )
    
    return fig


if target_cities:
    lu_cols = [col for col in datac.columns if col.startswith('lu')]
    radius_dfs = gen_regs_for_plot(data=datac,
                                   target_col=target_col,
                                   base_cols=base_cols,
                                   cat_cols=cat_cols,
                                   lu_cols=lu_cols,
                                   control_cols=[control_col],
                                   p_limit=False)
    
    fig = plot_ext_r_across_radii(radius_dfs,regions=target_cities, target_domain=target_col, p_value_threshold=0.07)
    st.plotly_chart(fig, use_container_width=True)


with st.expander(f'Regression table {target_cities}', expanded=False):
    p_filter = st.toggle('p-value filtering')
    radius = st.radio('Radius of neighborhood in km',[1,3,5,9],horizontal=True)
    df_r = datac[datac['R'] == f"R{radius}"]
    reg_df = ols_reg_table(df_in=df_r,
                            cf_col=target_col,
                            base_cols=base_cols,
                            cat_cols=cat_cols,
                            ext_cols=lu_cols,
                            control_cols=[control_col]
                            )
    
    if p_filter:
        reg_df = reg_df[reg_df['ext_p'] < 0.07]
        p_limit = True

    st.data_editor(reg_df,use_container_width=True, height=500)


