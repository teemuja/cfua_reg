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
                    SELECT *, 'R1' as R FROM read_parquet('{allas_url}/CFUA/RD/cfua_data_nordics_IQR1-5_R30-10_ND1km.parquet')
                    UNION ALL
                    SELECT *, 'R3' as R FROM read_parquet('{allas_url}/CFUA/RD/cfua_data_nordics_IQR1-5_R30-10_ND3km.parquet')
                    UNION ALL
                    SELECT *, 'R5' as R FROM read_parquet('{allas_url}/CFUA/RD/cfua_data_nordics_IQR1-5_R30-10_ND5km.parquet')
                    UNION ALL
                    SELECT *, 'R9' as R FROM read_parquet('{allas_url}/CFUA/RD/cfua_data_nordics_IQR1-5_R30-10_ND9km.parquet')
                """
            duckdb.sql(combined_table)
            df = duckdb.sql("SELECT * FROM combined_table").to_df().drop(columns=['__index_level_0__'])
            cols = ['R'] + [col for col in df.columns if col != 'R']
            df_out = df[cols]
        else:
            df_out = duckdb.sql(f"SELECT * FROM read_parquet('{allas_url}/CFUA/RD/cfua_data_nordics_IQR1-5_R30-10_ND{R}km.parquet')").to_df().drop(columns=['__index_level_0__'])
        
        return df_out.drop(columns="lu_other")


    data = load_data()
    if st.toggle('Described'):
        st.data_editor(data.describe(),key="init_desc")
    else:
        st.data_editor(data,key="init")

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
cf_cols = data.columns.tolist()[5:15]
lu_cols_orig = [col for col in data.columns if col.startswith('lu')]
base_cols = ['Household type', 'Car in household','Age']

st1,st2 = st.columns(2)
target_cities = st1.multiselect("Set sample regions",cities,default=["Helsinki","Stockholm"])
if target_cities:
    datac = data[data['fua_name'].isin(target_cities)]
else:
    st.stop()



with st.expander(f'Reg.settings', expanded=True):
    s1,s2,s3 = st.columns(3)
    target_col = s1.selectbox("Target domain",cf_cols,index=9) #last
    if target_col == "Total footprint unit":
        cat_cols = ['Household unit income decile', 'Household type', 'Car in household']
    else:
        cat_cols = ['Household per capita income decile', 'Household type', 'Car in household']
    control_col = s2.selectbox('Control column',cat_cols,index=0)
    remove_cols = s3.multiselect('Remove landuse classes',lu_cols_orig)
    if remove_cols:
        datac.drop(columns=remove_cols, inplace=True)

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
    """
    Create a line plot of ext_r values for different variables across multiple radii,
    considering p-value threshold.
    
    Parameters:
    dataframes_dict: dict
        Dictionary with radius as key and DataFrame as value
    p_value_threshold: float
        Maximum p-value to consider a relationship significant (default: 0.05)
    """
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


