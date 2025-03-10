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
from scipy.stats import yeojohnson
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
                    SELECT *, 'R1' as R FROM read_parquet('{allas_url}/CFUA/RD/cfua_data_nordics_IQR1-5_R30-10_reso10_ND1km_F.parquet')
                    UNION ALL
                    SELECT *, 'R5' as R FROM read_parquet('{allas_url}/CFUA/RD/cfua_data_nordics_IQR1-5_R30-10_reso10_ND5km_F.parquet')
                    UNION ALL
                    SELECT *, 'R9' as R FROM read_parquet('{allas_url}/CFUA/RD/cfua_data_nordics_IQR1-5_R30-10_reso10_ND9km_F.parquet')
                """
            duckdb.sql(combined_table)
            dropcols = ['__index_level_0__']
            df = duckdb.sql("SELECT * FROM combined_table").to_df()#.drop(columns=dropcols)
            
            cols = ['R'] + [col for col in df.columns if col != 'R']
            df_out = df[cols]
        else:
            df_out = duckdb.sql(f"SELECT * FROM read_parquet('{allas_url}/CFUA/RD/cfua_data_nordics_IQR1-5_R30-10_reso10_ND{R}km_F.parquet')").to_df().drop(columns=['__index_level_0__'])
        
        cf_cols = ['Housing footprint',
                'Diet footprint',
                'Vehicle possession footprint',
                'Public transportation footprint',
                'Leisure travel footprint',
                'Goods and services footprint',
                'Pets footprint',
                'Summer house footprint',
                'Total footprint',
                'Total footprint unit'
                ]
        
        df_out.rename(columns={col: 'cf_' + col for col in cf_cols}, inplace=True)

        #fix lu_lu_ ..
        lu_cols_orig = [col for col in df_out.columns if col.startswith('lu_')]
        df_out.rename(columns={col: col.replace('lu_lu_', 'lu_') for col in lu_cols_orig}, inplace=True)

        #add age group
        #df_out.loc[df_out['Age'] < 25, "Age group"] = 0
        #df_out.loc[df_out['Age'] >= 25, "Age group"] = 1
        #df_out.loc[df_out['Age'] >= 50, "Age group"] = 2
        #drop cols
        df_out.drop(columns=["landuse_class","Urban degree","Number of persons in household"],inplace=True)

        return df_out

    data = load_data()
    #cols
    cf_cols = [col for col in data.columns if col.startswith('cf_')]
    lu_cols_orig = [col for col in data.columns if col.startswith('lu_')]
    lu_cols_map = {
        "lu_other":"lu_other",
        "lu_Forests":"lu_forest",
        "lu_Discontinuous very low density urban fabric":"lu_exurb",
        "lu_Continuous urban fabric":"lu_urban",
        "lu_Discontinuous dense urban fabric":"lu_modern",
        "lu_Sports and leisure facilities":"lu_leisure",
        "lu_Discontinuous medium density urban fabric":"lu_suburb",
        "lu_Industrial, commercial, public, military and private units":"lu_facility",
        "lu_Discontinuous low density urban fabric":"lu_suburb",
        "lu_Green urban areas":"lu_green",
        "lu_Herbaceous vegetation associations":"lu_green",
        "lu_Open spaces with little or no vegetation":"lu_open"
        }
    
    temp_df = data.rename(columns=lu_cols_map)
    summed_cols = temp_df.groupby(axis=1, level=0).sum()
    for new_col in summed_cols.columns:
        data[new_col] = summed_cols[new_col]
    
    data.drop(columns=lu_cols_orig,inplace=True)

    st.data_editor(data.describe(),key="init_desc")

        
    @st.dialog("Download data")
    def download(df):
        cities = df['fua_name'].unique().tolist()
        c = st.selectbox('Select case city',cities + ["All"])
        r = st.radio("Select which neigborhood radius data to download",[1,5,9,"all"],horizontal=True)
        if c != "All":
            dfc = df[df['fua_name'] == c]
        else:
            dfc = df.copy()
        if r != "all":
            dfr = dfc[dfc['R'] == f"R{r}"]
        else:
            dfr = dfc.copy()
            del dfc
        
        st.download_button(
                            label=f"Download {c} {r}km data as CSV",
                            data=dfr.to_csv().encode("utf-8"),
                            file_name=f"cfua_data_{c}_with_landuse_radius_{r}km.csv",
                            mime="text/csv",
                        )

    if st.button('Download'):
        download(df=data)

    st.table(lu_cols_map)
    dbstatus.update(label="DB connected!", state="complete", expanded=False)

#st.stop()


def ols_reg_table(df_in, cf_col, base_cols, cat_cols, ext_cols, control_cols):
    
    def normalize_df(df_in,cols=None):
        df = df_in.copy()
        if cols is None:
            cols = df.select_dtypes(include=np.number).columns.tolist()
        for col in cols:
            df[col] = df[col] / df[col].abs().max()
        return df
    
    df = normalize_df(df_in)

    def ols(df, cf_col, base_cols, cat_cols, ext_cols, control_cols):
        cat_cols_lower = [col.lower().replace(' ', '_') for col in cat_cols]
        
        def format_col(col):
            col_lower = col.lower().replace(' ', '_').replace(',', '_').replace('&', '_')
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
        
        try:
            base_model = smf.ols(formula=base_formula, data=df)
            base_results = base_model.fit()
            ext_model = smf.ols(formula=ext_formula, data=df)
            ext_results = ext_model.fit()
            return base_results, ext_results
        except Exception as e:
            st.error(f"Regression failed: {e}")
            return None, None
    
    def ols_debug(df, cf_col, base_cols, cat_cols, ext_cols, control_cols):
        """
        Perform OLS regression with debugging statements to handle edge cases.
        """

        try:
            # Step 1: Debugging DataFrame Basics
            print(f"[DEBUG] DataFrame shape: {df.shape}")
            print(f"[DEBUG] Columns in DataFrame: {df.columns.tolist()}")
            
            if df.empty:
                raise ValueError("[ERROR] DataFrame is empty. Cannot perform regression.")

            # Step 2: Normalize Column Names
            df.columns = df.columns.str.lower().str.replace(' ', '_')
            cf_col = cf_col.lower().replace(' ', '_')
            base_cols = [col.lower().replace(' ', '_') for col in base_cols]
            ext_cols = [col.lower().replace(' ', '_') for col in ext_cols]
            control_cols = [col.lower().replace(' ', '_') for col in control_cols]
            cat_cols = [col.lower().replace(' ', '_') for col in cat_cols]

            print("[DEBUG] Normalized columns:")
            print(f"  cf_col: {cf_col}")
            print(f"  base_cols: {base_cols}")
            print(f"  ext_cols: {ext_cols}")
            print(f"  control_cols: {control_cols}")
            print(f"  cat_cols: {cat_cols}")

            # Step 3: Validate Columns in DataFrame
            required_cols = base_cols + ext_cols + control_cols + [cf_col]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"[ERROR] Missing columns in DataFrame: {missing_cols}")

            # Step 4: Check for Zero Variance
            zero_variance_cols = [col for col in required_cols if col in df and df[col].var() == 0]
            if zero_variance_cols:
                raise ValueError(f"[ERROR] Columns with zero variance: {zero_variance_cols}")

            # Step 5: Construct Formulas
            def format_col(col):
                return f"C({col})" if col in cat_cols else col

            base_formula = f"{cf_col} ~ " + " + ".join(format_col(col) for col in base_cols)
            ext_formula = base_formula + " + " + " + ".join(format_col(col) for col in ext_cols)
            if control_cols:
                control_formula = " + " + " + ".join(format_col(col) for col in control_cols)
                base_formula += control_formula
                ext_formula += control_formula

            print(f"[DEBUG] Base formula: {base_formula}")
            print(f"[DEBUG] Extended formula: {ext_formula}")

            # Step 6: Perform Regression
            base_model = smf.ols(formula=base_formula, data=df)
            base_results = base_model.fit()
            ext_model = smf.ols(formula=ext_formula, data=df)
            ext_results = ext_model.fit()

            print("[DEBUG] Base regression successful")
            print("[DEBUG] Extended regression successful")

            return base_results, ext_results

        except Exception as e:
            print(f"[ERROR] Regression failed: {e}")
            return None, None

    
    base_results, ext_results = ols_debug(df, cf_col, base_cols, cat_cols, ext_cols, control_cols)
    rounder = 3
    if base_results and ext_results:
        reg_results = pd.DataFrame({
            'base_r': round(base_results.params, rounder),
            'base_p': round(base_results.pvalues, rounder),
            'ext_r': round(ext_results.params, rounder),
            'ext_p': round(ext_results.pvalues, rounder)
            })
        
        return reg_results
    else:
        st.warning('Error in regression data')

with st.expander('Case cities', expanded=False):
    cities = data['fua_name'].unique().tolist()
    
    st1,st2 = st.columns(2)
    target_cities = st1.multiselect("Set sample regions",cities,default=["Helsinki","Stockholm"])
    datac = data[data['fua_name'].isin(target_cities)]
    lu_cols_in_use = [col for col in datac.columns if col.startswith('lu')]
    st.data_editor(datac.describe())

if target_cities:
    with st.expander(f'Regression settings', expanded=True):
        s1,s2,s3 = st.columns(3)
        target_col = s1.selectbox("Target domain",cf_cols,index=5)
        #base_cols = ['Age','Education level','Household type','Car in household']
        if target_col == "cf_Total footprint unit":
            base_cols = ['Age','Education level','Household type','Car in household','Household unit income decile ORIG']
        else:
            base_cols = ['Age','Education level','Household type','Car in household','Household per capita income decile ORIG']
        cat_cols = ['Household type', 'Car in household']
        control_col = s2.selectbox('Control column',base_cols,index=4)

        #reclassification
        combine_cols1 = s1.multiselect('Combine landuse classes',lu_cols_in_use)
        if len(combine_cols1) > 1:
            new_col_name = "_".join(combine_cols1)
            datac[new_col_name] = datac[combine_cols1].sum(axis=1)
            datac.drop(columns=combine_cols1, inplace=True)
            lu_cols_in_use = [col for col in datac.columns if col.startswith('lu')]
            #another comb set
            combine_cols2 = s2.multiselect('Combine landuse classes',lu_cols_in_use)
            if len(combine_cols2) > 1:
                new_col_name2 = "_".join(combine_cols2)
                datac[new_col_name2] = datac[combine_cols2].sum(axis=1)
                datac.drop(columns=combine_cols2, inplace=True)
                lu_cols_in_use = [col for col in datac.columns if col.startswith('lu')]

        remove_cols = s3.multiselect('Remove landuse classes',lu_cols_in_use)
        if remove_cols:
            datac.drop(columns=remove_cols, inplace=True)
        st.info("E.g. Remove 'lu_open' and combine 'lu_exurb' with 'lu_suburb' ..and/or 'lu_facility' with 'lu_leisure'")
        
        lu_cols = [col for col in datac.columns if col.startswith('lu')]
        st.markdown("---")
        power_lu = st.toggle("Power transform land-use distribution",help="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.yeojohnson.html")
        power_cf = st.toggle("Power transform footprint distribution",help="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.yeojohnson.html")
        
        if power_lu:
            for radius in [1,5,9]:
                df_r = datac[datac['R'] == f"R{radius}"]
                transformed_values = {}
                for col in lu_cols:
                    df_r[col], lam = yeojohnson(df_r[col])
                    transformed_values[col] = df_r[col]

                # Update datac with transformed values for pearson later in the code..
                datac.loc[datac['R'] == f"R{radius}", lu_cols] = df_r[lu_cols]
        
        if power_cf:
            for radius in [1,5,9]:
                df_r = datac[datac['R'] == f"R{radius}"]
                transformed_values = {}
                for col in cf_cols:
                    df_r[col], lam = yeojohnson(df_r[col])
                    transformed_values[col] = df_r[col]

                # Update datac with transformed values for pearson later in the code..
                datac.loc[datac['R'] == f"R{radius}", cf_cols] = df_r[cf_cols]

        #plot histos
        yks,viis,ysi,cf = st.tabs(['1km','5km','9km','CF domains'])
 
        with yks:
            histo_traces_lu = []
            df_r = datac[datac['R'] == f"R1"]
            for col in lu_cols:
                histo = go.Histogram(x=df_r[col],opacity=0.75,name=col,nbinsx=20)
                histo_traces_lu.append(histo)
            layout_histo = go.Layout(title='Landuse type histograms',barmode='overlay')
            histo_fig_lu = go.Figure(data=histo_traces_lu, layout=layout_histo)
            st.plotly_chart(histo_fig_lu, use_container_width=True)
        with viis:
            histo_traces = []
            df_r = datac[datac['R'] == f"R5"]
            for col in lu_cols:
                histo = go.Histogram(x=df_r[col],opacity=0.75,name=col,nbinsx=20)
                histo_traces.append(histo)
            layout_histo = go.Layout(title='Landuse type histograms',barmode='overlay')
            histo_fig = go.Figure(data=histo_traces, layout=layout_histo)
            st.plotly_chart(histo_fig, use_container_width=True,key=viis)
        with ysi:
            histo_traces = []
            df_r = datac[datac['R'] == f"R9"]
            for col in lu_cols:
                histo = go.Histogram(x=df_r[col],opacity=0.75,name=col,nbinsx=20)
                histo_traces.append(histo)
            layout_histo = go.Layout(title='Landuse type histograms',barmode='overlay')
            histo_fig = go.Figure(data=histo_traces, layout=layout_histo)
            st.plotly_chart(histo_fig, use_container_width=True,key=ysi)
        with cf:
            histo_traces_cf = []
            df_r = datac[datac['R'] == f"R1"]
            for col in cf_cols:
                histo = go.Histogram(x=df_r[col],opacity=0.75,name=col,nbinsx=20)
                histo_traces_cf.append(histo)
            layout_histo = go.Layout(title='CF domain histograms',barmode='overlay')
            histo_fig_cf = go.Figure(data=histo_traces_cf, layout=layout_histo)
            st.plotly_chart(histo_fig_cf, use_container_width=True)

    #st.data_editor(datac)

    with st.expander(f'Regression tables {target_cities}', expanded=False):

        yksi,viisi,yhdeksan = st.tabs(['1km','5km','9km'])
        with yksi:
            df_r = datac[datac['R'] == f"R1"]
            #table for r
            r_lu_cols = [col for col in df_r.columns if col.startswith('lu')]
            reg_df = ols_reg_table(df_in=df_r.fillna(0),
                                    cf_col=target_col,
                                    base_cols=base_cols,
                                    cat_cols=cat_cols,
                                    ext_cols=r_lu_cols,
                                    control_cols=[control_col]
                                    )
            st.data_editor(reg_df,use_container_width=True, height=500,key=yksi)
        with viisi:
            df_r = datac[datac['R'] == f"R5"]
            #table for r
            r_lu_cols = [col for col in df_r.columns if col.startswith('lu')]
            reg_df = ols_reg_table(df_in=df_r.fillna(0),
                                    cf_col=target_col,
                                    base_cols=base_cols,
                                    cat_cols=cat_cols,
                                    ext_cols=r_lu_cols,
                                    control_cols=[control_col]
                                    )
            st.data_editor(reg_df,use_container_width=True, height=500,key=viisi)
        with yhdeksan:
            df_r = datac[datac['R'] == f"R9"]
            #table for r
            r_lu_cols = [col for col in df_r.columns if col.startswith('lu')]
            reg_df = ols_reg_table(df_in=df_r.fillna(0),
                                    cf_col=target_col,
                                    base_cols=base_cols,
                                    cat_cols=cat_cols,
                                    ext_cols=r_lu_cols,
                                    control_cols=[control_col]
                                    )
            st.data_editor(reg_df,use_container_width=True, height=500,key=yhdeksan)


else:
    st.stop()


with st.expander(f'Regression plot {target_cities}', expanded=False):
    c1,c2 =st.columns(2)
    p_value_thres = c1.slider('P-value filter',0.01,0.2,0.09,step=0.01)
    def gen_regs_for_plot(data,
                        target_col,
                        base_cols,
                        cat_cols,
                        lu_cols,
                        control_cols,
                        p_limit=False):
        radius_dfs = {}
        for r in [1,5,9]:
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
                limit = 0.07
                reg = reg[reg['ext_p'] < limit]

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
        
        custom_color_map = {
                            "lu_facility": "violet",
                            "lu_modern": "brown",
                            "lu_suburb": "burlywood",
                            "lu_exurb":"palegoldenrod",
                            "lu_urban":"red",
                            "lu_urban_lu_modern":"red",
                            "lu_leisure":"orange",
                            "lu_open":"skyblue",
                            "lu_green": "olive",
                            "lu_green_lu_leisure":"olive",
                            "lu_forest":"darkgreen",
                        }
        
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
                    line=dict(color=custom_color_map.get(var, 'gray')),
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
            title=f"Correlation between '{target_domain}' and landuse types in {regions}",
            xaxis_title='Radius (km)',
            yaxis_range=[-0.5,0.5],
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

    lu_cols_plot = [col for col in datac.columns if col.startswith('lu')]
    radius_dfs = gen_regs_for_plot(data=datac,
                                    target_col=target_col,
                                    base_cols=base_cols,
                                    cat_cols=cat_cols,
                                    lu_cols=lu_cols_plot,
                                    control_cols=[control_col],
                                    p_limit=False)

    fig = plot_ext_r_across_radii(radius_dfs,regions=target_cities, target_domain=target_col, p_value_threshold=p_value_thres)
    st.plotly_chart(fig, use_container_width=True)


