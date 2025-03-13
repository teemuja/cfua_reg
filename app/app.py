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
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import mode


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
        "lu_Sports and leisure facilities":"lu_sports",
        "lu_Discontinuous medium density urban fabric":"lu_suburb",
        "lu_Industrial, commercial, public, military and private units":"lu_facilities",
        "lu_Discontinuous low density urban fabric":"lu_suburb",
        "lu_Green urban areas":"lu_parks",
        "lu_Herbaceous vegetation associations":"lu_parks",
        "lu_Open spaces with little or no vegetation":"lu_open"
        }
    
    temp_df = data.rename(columns=lu_cols_map)
    summed_cols = temp_df.groupby(axis=1, level=0).sum()
    for new_col in summed_cols.columns:
        data[new_col] = summed_cols[new_col]
    del temp_df
    data.drop(columns=lu_cols_orig,inplace=True)

    #also rename h3_10 for clustering use
    data = data.rename(columns={'h3_10':'h3_id'})

    st.data_editor(data,key="init_cols",height=200)

    st.markdown("**Land-use reclassification**")
    st.table(lu_cols_map)
    dbstatus.update(label="DB connected!", state="complete", expanded=False)


def clusterize(df_in, agg_cols_mean, agg_cols_mode, toreso=9, min_cluster_size=2):
    df = df_in.copy()

    #add parent cell
    df[f'h3_0{toreso}'] = df['h3_id'].apply(lambda x: h3.cell_to_parent(x, toreso))

    # Prepare an empty list to store the results
    result_list = []

    # Loop through each unique R value to process separately
    for r_value in df['R'].unique():
        # Filter the DataFrame by the current R value
        subset_df = df[df['R'] == r_value]

        # Count the number of occurrences of each H3 hex (aggregated by the parent hex)
        cluster_counts = subset_df.groupby(f'h3_0{toreso}').size().reset_index(name='count')

        # Keep only valid clusters based on the minimum cluster size
        valid_clusters = cluster_counts[cluster_counts['count'] >= min_cluster_size][f'h3_0{toreso}']

        # Filter the DataFrame to only include rows that belong to valid clusters
        filtered_df = subset_df[subset_df[f'h3_0{toreso}'].isin(valid_clusters)]

        # Aggregate by calculating the mean for numerical columns and rounding
        agg_mean_df = filtered_df.groupby(f'h3_0{toreso}')[agg_cols_mean].mean().round(0).reset_index()

        # Aggregate the mode for categorical columns
        def safe_mode(series):
            """Returns the most frequent value (mode), or None if no mode exists."""
            modes = series.mode()
            return modes.iloc[0] if not modes.empty else None

        agg_mode_df = filtered_df.groupby(f'h3_0{toreso}')[agg_cols_mode].agg(safe_mode).reset_index()

        # Combine the mean and mode dataframes
        agg_df = pd.merge(agg_mean_df, agg_mode_df, on=f'h3_0{toreso}', how='inner')

        # Add the current R value as a column
        agg_df['R'] = r_value

        # Append the aggregated dataframe for this R value
        result_list.append(agg_df)

    # Combine all R results back together
    final_df = pd.concat(result_list, ignore_index=True)

    # Optional: Remove any duplicates that may have occurred after merging
    final_df = final_df.drop_duplicates(subset=[f'h3_0{toreso}', 'R'])

    return final_df.rename(columns={f'h3_0{toreso}':'h3_id'}) #rename h3 back to id col


#city selector expander
with st.expander('Case cities & Clusters', expanded=False):
    cities = data['fua_name'].unique().tolist()
    
    st1,st2,st3 = st.columns(3)
    all_cities = ["Helsinki","Stockholm","København","Oslo","Reykjavík"]
    target_cities = st1.multiselect("Set sample regions",cities,default=all_cities)

    if target_cities:
        cluster = st2.radio("Clusterize", ['None',9,8], horizontal=True,
                            help="https://h3geo.org/docs/core-library/restable/")

        min_cluster_size = st3.radio("Min cf_records for cluster", [2,3], horizontal=True)

        datac = data[data['fua_name'].isin(target_cities)]

        #col types for clusterizing
        feat_cols = ['R','fua_name']
        cat_cols = ['Country','Education level','Household type','Car in household']
        lu_cols_in_use = [col for col in datac.columns if col.startswith('lu')]
        non_cat_base_cols = ['Age','Household per capita income decile ORIG','Household unit income decile ORIG']
        agg_cols_mean = cf_cols + lu_cols_in_use + non_cat_base_cols
        agg_cols_mode = feat_cols + cat_cols

        if cluster != 'None':
            datac = clusterize(df_in=datac, agg_cols_mean=agg_cols_mean, agg_cols_mode=agg_cols_mode,
                            toreso=cluster, min_cluster_size=min_cluster_size)

        col_order = ['h3_id'] + feat_cols + cat_cols + non_cat_base_cols + cf_cols + lu_cols_in_use
        datac = datac[col_order]

        st.data_editor(datac.describe())
        st.data_editor(datac, height=200)

    else:
        st.stop()

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
        download(df=datac)

def ols_reg_table(df_in, cf_col, base_cols, cat_cols, ext_cols, standardize=True):
    
    def normalize_df(df_in,cols=None):
        df = df_in.copy()
        if cols is None:
            cols = df.select_dtypes(include=np.number).columns.tolist()
        for col in cols:
            df[col] = df[col] / df[col].abs().max()
        return df
    
    def standardize_df(df_in, cols=None):
        df = df_in.copy()
        if cols is None:
            cols = df.select_dtypes(include=np.number).columns.tolist()
        for col in cols:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
        return df
    
    do_not_norm_cols = cat_cols + feat_cols + ['h3_id']
    cols_to_normalize = [col for col in df_in.columns if col not in do_not_norm_cols]

    if standardize:
        df = standardize_df(df_in,cols=cols_to_normalize)
    else:
        df = normalize_df(df_in,cols=cols_to_normalize)
    
    #actual OLS function
    def ols(df, cf_col, base_cols, cat_cols, ext_cols):

        #convert all cols lower in df..
        df.columns = df.columns.str.lower().str.replace(' ', '_')

        #define cat cols as lower
        cat_cols_lower = [col.lower().replace(' ', '_') for col in cat_cols]
        
        #func to format col for categorical C if listed above
        def format_col(col):
            col_lower = col.lower().replace(' ', '_').replace(',', '_').replace('&', '_')
            return f'C({col_lower})' if col_lower in cat_cols_lower else col_lower

        #make domain col also lower
        domain_col = cf_col.lower().replace(' ', '_')
        
        #define strings for R -style formula
        base_cols_str = ' + '.join([format_col(col) for col in base_cols])
        ext_cols_str = ' + '.join([format_col(col) for col in ext_cols])
        
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
 
    #RUN ols
    base_results, ext_results = ols(df, cf_col, base_cols, cat_cols, ext_cols)

    #construct result df
    rounder = 3
    if base_results and ext_results:
        reg_results = pd.DataFrame({
            'base_r': round(base_results.params, rounder),
            'base_p': round(base_results.pvalues, rounder),
            'ext_r': round(ext_results.params, rounder),
            'ext_p': round(ext_results.pvalues, rounder)
            })
        
        #return all
        return reg_results, base_results, ext_results
    
    else:
        st.warning('Error in regression calculation. Probably needs more records. Change targets and settings.')
        st.stop()

if target_cities:
    with st.expander(f'Regression settings', expanded=True):
        s1,s2 = st.columns([1,3])
        target_col = s1.selectbox("Target domain",cf_cols,index=5)

        #base cols
        if target_col == "cf_Total footprint unit":
            con_cols_to_choose = ['Country','Education level','Household type','Car in household','Household unit income decile ORIG']
        else:
            con_cols_to_choose = ['Country','Education level','Household type','Car in household','Household per capita income decile ORIG']
        
        #multiselect cat cols to use = control
        base_cols_orig = ['Age']
        control_cols = s2.multiselect('Control columns',con_cols_to_choose, default=["Country",con_cols_to_choose[4]])

        #reclassification
        s1,s2,s3 = st.columns(3)

        remove_cols = s1.multiselect('Remove landuse classes',lu_cols_in_use, default='lu_open')
        if remove_cols:
            datac.drop(columns=remove_cols, inplace=True)

        combine_cols1 = s2.multiselect('Combine landuse classes',lu_cols_in_use)
        if len(combine_cols1) > 1:
            new_col_name = "_".join(combine_cols1)
            datac[new_col_name] = datac[combine_cols1].sum(axis=1)
            datac.drop(columns=combine_cols1, inplace=True)
            lu_cols_in_use = [col for col in datac.columns if col.startswith('lu')]
            
            #..add another comb set
            combine_cols2 = s3.multiselect('..Combine more',lu_cols_in_use)
            if len(combine_cols2) > 1:
                new_col_name2 = "_".join(combine_cols2)
                datac[new_col_name2] = datac[combine_cols2].sum(axis=1)
                datac.drop(columns=combine_cols2, inplace=True)
                lu_cols_in_use = [col for col in datac.columns if col.startswith('lu')]

        st.info("E.g. Remove 'lu_open' and combine 'lu_exurb' with 'lu_suburb' ..and/or 'lu_facility' with 'lu_leisure'")
        
        lu_cols = [col for col in datac.columns if col.startswith('lu')]

    with st.expander(f'Distribution settings', expanded=False):
        use_shares = st.toggle('Use percents with land-use types')
        standardize = st.toggle("Standardize instead of normalize")
        power_lu = st.toggle("Power transform land-use distribution",help="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.yeojohnson.html")
        power_cf = st.toggle("Power transform footprint distribution",help="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.yeojohnson.html")
        
        if power_lu:
            for radius in [1,5,9]:
                df_r = datac[datac['R'] == f"R{radius}"]
                for col in lu_cols:
                    df_r[col], lam = yeojohnson(df_r[col])
                datac.loc[datac['R'] == f"R{radius}", lu_cols] = df_r[lu_cols]
        
        if power_cf:
            for radius in [1,5,9]:
                df_r = datac[datac['R'] == f"R{radius}"]
                for col in cf_cols:
                    df_r[col], lam = yeojohnson(df_r[col])
                datac.loc[datac['R'] == f"R{radius}", cf_cols] = df_r[cf_cols]

        #shares
        if use_shares:
            nd_size_dict = {1: 7, 5: 35, 9: 63}
            for radius in [1, 5, 9]:
                df_r = datac[datac['R'] == f"R{radius}"].copy()
                k = nd_size_dict[radius]
                max_hex = 1 + 3 * k * (k + 1)

                # Update each column in lu_cols
                for col in lu_cols:
                    df_r[col] = round(df_r[col] / max_hex, 3)

                # Push the updated data back into the main DataFrame
                datac.loc[datac['R'] == f"R{radius}", lu_cols] = df_r[lu_cols].values

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

    with st.expander(f'**Regression tables** {target_cities}', expanded=True):
        st.caption('https://www.statsmodels.org/stable/example_formulas.html')

        yksi,viisi,yhdeksan = st.tabs(['1km','5km','9km'])
        with yksi:
            df_r1 = datac[datac['R'] == f"R1"]
            
            #table for r
            r1_lu_cols = [col for col in df_r1.columns if col.startswith('lu')]

            reg_df1, base_results, ext_results = ols_reg_table(df_in=df_r.fillna(0),
                                    cf_col=target_col,
                                    base_cols=base_cols_orig + control_cols,
                                    cat_cols=cat_cols,
                                    ext_cols=r1_lu_cols,
                                    standardize=standardize
                                    )
            st.data_editor(reg_df1,use_container_width=True, height=500,key=yksi)

        #st.stop() # ----------------------------------------------------------------------------

        with viisi:
            df_r5 = datac[datac['R'] == f"R5"]
            #table for r
            r5_lu_cols = [col for col in df_r5.columns if col.startswith('lu')]
            reg_df5, base_results, ext_results = ols_reg_table(df_in=df_r.fillna(0),
                                    cf_col=target_col,
                                    base_cols=base_cols_orig + control_cols,
                                    cat_cols=cat_cols,
                                    ext_cols=r5_lu_cols,
                                    standardize=standardize
                                    )
            st.data_editor(reg_df5,use_container_width=True, height=500,key=viisi)

        with yhdeksan:
            df_r9 = datac[datac['R'] == f"R9"]
            #table for r
            r9_lu_cols = [col for col in df_r9.columns if col.startswith('lu')]
            reg_df9, base_results, ext_results = ols_reg_table(df_in=df_r.fillna(0),
                                    cf_col=target_col,
                                    base_cols=base_cols_orig + control_cols,
                                    cat_cols=cat_cols,
                                    ext_cols=r9_lu_cols,
                                    standardize=standardize
                                    )
            st.data_editor(reg_df9,use_container_width=True, height=500,key=yhdeksan)

else:
    st.stop()


#st.stop()

with st.expander(f'Regression change plot {target_cities}', expanded=False):
    c1,c2 =st.columns(2)
    p_value_thres = c1.slider('P-value filter',0.01,0.3,0.2,step=0.01)

    def gen_regs_for_plot(data,
                        target_col,
                        base_cols,
                        cat_cols,
                        lu_cols,
                        standardize,
                        p_limit=False):
        radius_dfs = {}
        for r in [1,5,9]:
            df_r = data[data['R'] == f"R{r}"]
            reg_df, base_results, ext_results = ols_reg_table(df_in=df_r.fillna(0),
                                    cf_col=target_col,
                                    base_cols=base_cols,
                                    cat_cols=cat_cols,
                                    ext_cols=lu_cols,
                                    standardize=standardize
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
                            "lu_facilities": "violet",
                            "lu_modern": "brown",
                            "lu_suburb": "burlywood",
                            "lu_exurb":"palegoldenrod",
                            "lu_urban":"darkred",
                            "lu_urban_lu_modern":"red",
                            "lu_sports":"orange",
                            "lu_open":"skyblue",
                            "lu_parks": "olive",
                            "lu_parks_lu_sports":"olive",
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
            yaxis_range=[-0.9,0.9],
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
                                    base_cols=base_cols_orig + control_cols,
                                    cat_cols=cat_cols,
                                    lu_cols=lu_cols_plot,
                                    standardize=standardize,
                                    p_limit=False)

    fig = plot_ext_r_across_radii(radius_dfs,regions=target_cities, target_domain=target_col, p_value_threshold=p_value_thres)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    def gen_reg_df_all():
        reg_df1['ND_km'] = 1
        reg_df5['ND_km'] = 5
        reg_df9['ND_km'] = 9
        reg_df_all = pd.concat([reg_df1,reg_df5,reg_df9])
        setup = [f'Cluster_level:{cluster}',f'Min_cluster_size:{min_cluster_size}',
                 f'lu_shares: {use_shares}',f'Standardize: {standardize}',
                 f'lu_power_transformed: {power_lu}', f'cf_power_transformed: {power_cf}']
        st.download_button(
                            label=f"Download study as CSV",
                            data=reg_df_all.to_csv().encode("utf-8"),
                            file_name=f"cfua_data_{target_col}_{target_cities}_{setup}.csv",
                            mime="text/csv",
                            )
        
    gen_reg_df_all()







with st.expander(f'Heatmaps for the sample {target_cities}, N={len(datac)}', expanded=False):
    import matplotlib.pyplot as plt
    import plotly.express as px
    import altair as alt
    from scipy.stats import pearsonr

    dropcols = ['h3_id','fua_name','Country']

    c1,c2 =st.columns(2)
    p_value_thres = c1.slider('P-value threshold',0.01,0.2,0.09,step=0.01)

    def calculate_p_values(df):
        """Calculate p-values for pairwise correlations in a DataFrame."""
        cols = df.columns
        n = df.shape[0]
        p_values = np.zeros((len(cols), len(cols)))
        
        for i, col1 in enumerate(cols):
            for j, col2 in enumerate(cols):
                if i == j:
                    p_values[i, j] = 0  # Correlation of a feature with itself is 1, p-value is 0
                else:
                    _, p_value = pearsonr(df[col1], df[col2])
                    p_values[i, j] = p_value
                    
        return pd.DataFrame(p_values, index=cols, columns=cols)

    def plot_correlation_heatmap_plotly(df, p_value_limit=0.05, facet_col='R'):
        
        facet_values = df[facet_col].unique()
        facet_data = []
        
        for value in facet_values:
            subset = df[df[facet_col] == value].drop(columns=[facet_col])
            corr_matrix = subset.corr()
            p_values = calculate_p_values(subset)
            
            # Mask correlations with p-values above the limit
            mask = p_values > p_value_limit
            corr_matrix[mask] = np.nan
            
            # Melt the correlation matrix for plotting
            melted_corr = corr_matrix.reset_index().melt(id_vars='index', value_name='Correlation')
            melted_corr.columns = ['Feature1', 'Feature2', 'Correlation']
            melted_corr['Facet'] = value
            facet_data.append(melted_corr)
        
        final_df = pd.concat(facet_data)
        
        # Plot faceted heatmaps
        fig = px.density_heatmap(final_df, x='Feature1', y='Feature2', z='Correlation',
                                facet_col='Facet', color_continuous_scale='magma',
                                labels={'sum of Correlation': 'Correlation'},
                                histfunc=None)

        fig.update_layout(
            title="Correlation Heatmap",
            font=dict(size=10),
            width=1200,
            height=600,
            margin=dict(l=50, r=50, b=50, t=50),
        )

        fig.update_xaxes(tickfont=dict(size=8))
        fig.update_yaxes(tickfont=dict(size=8))
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], font=dict(size=10)))  # Smaller facet label font
        
        return fig
    
    def plot_correlation_heatmap_altair(df, p_value_limit=0.05, facet_col='R'):
        """Plot faceted correlation heatmaps with p-value filtering using Altair."""
        
        facet_values = df[facet_col].unique()
        facet_data = []
        
        for value in facet_values:
            subset = df[df[facet_col] == value].drop(columns=[facet_col])
            corr_matrix = subset.corr()
            p_values = calculate_p_values(subset)
            
            # Mask correlations with p-values above the limit
            mask = p_values > p_value_limit
            corr_matrix[mask] = np.nan
            
            # Melt the correlation matrix for Altair
            melted_corr = corr_matrix.reset_index().melt(id_vars='index', value_name='Correlation')
            melted_corr.columns = ['Feature1', 'Feature2', 'Correlation']
            melted_corr['Facet'] = value
            facet_data.append(melted_corr)
        
        # Combine all faceted data into one DataFrame
        final_df = pd.concat(facet_data)
        
        # Create the base heatmap
        heatmap = alt.Chart(final_df).mark_rect().encode(
            x='Feature1:N',  # Feature1 as nominal (categorical)
            y='Feature2:N',  # Feature2 as nominal (categorical)
            color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='magma')),  # Color scale
            tooltip=['Feature1', 'Feature2', 'Correlation']  # Tooltip for interactivity
        ).properties(
            width=300,  # Width of each facet
            height=300  # Height of each facet
        )
        
        # Facet the heatmap by the 'Facet' column
        faceted_heatmap = heatmap.facet(
            facet=alt.Facet('Facet:N', title=None),  # Facet by the 'Facet' column
            columns=2  # Number of columns in the facet grid
        ).resolve_scale(
            x='independent',  # Independent x-axis for each facet
            y='independent'   # Independent y-axis for each facet
        )
        
        return faceted_heatmap

    def plot_scatter(df, x_col, y_cols):
        fig = px.scatter()
        for y_col in y_cols:
            fig.add_scatter(x=df[x_col], y=df[y_col], mode='markers', name=y_col)
        
        fig.update_layout(
            title=f"Scatter Plot of {x_col} vs Multiple Y Columns",
            xaxis_title=x_col,
            yaxis_title="Values",
            legend_title="Y Columns"
        )
        return fig

    df_r = datac[datac['R'] == f"R1"]
    df_h = df_r.drop(columns=dropcols)
    f1 = plot_correlation_heatmap_altair(datac.drop(columns=dropcols), p_value_limit=p_value_thres)
    lu_cols_in_h = [col for col in df_h.columns if col.startswith('lu')]
    #f1 = plot_scatter(df_h,x_col=target_col,y_cols=lu_cols_in_h)
    #st.plotly_chart(f1,use_container_width=True)
    st.altair_chart(f1)
