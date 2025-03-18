# NDP app always beta a lot
import streamlit as st
import pandas as pd
import h3
import numpy as np
import duckdb
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

#stats
from scipy import stats
from scipy.stats import yeojohnson
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import mode
import pingouin as pg


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

        #Make country num
        df_out.loc[df_out['Country'] == "FI", 'Country'] = 0
        df_out.loc[df_out['Country'] == "SE", 'Country'] = 1
        df_out.loc[df_out['Country'] == "DK", 'Country'] = 2
        df_out.loc[df_out['Country'] == "NO", 'Country'] = 3
        df_out.loc[df_out['Country'] == "IS", 'Country'] = 4

        #drop cols
        df_out.drop(columns=["landuse_class","Urban degree","Number of persons in household"],inplace=True)

        return df_out

    data = load_data()
    #cols
    cf_cols = [col for col in data.columns if col.startswith('cf_')]
    lu_cols_orig = [col for col in data.columns if col.startswith('lu_')]
    lu_cols_map = {
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


## ------- combined reg study ---------

target_col = st.selectbox("Target domain",cf_cols,index=5)

with st.expander(f'Regression settings', expanded=True):

    #base cols
    if target_col == "cf_Total footprint unit":
        con_cols_to_choose = ['Country','Education level','Household type','Car in household','Household unit income decile ORIG']
    else:
        con_cols_to_choose = ['Country','Education level','Household type','Car in household','Household per capita income decile ORIG']
    
    #multiselect cat cols to use = control
    base_cols_orig = ['Age']
    control_cols = st.multiselect('Control columns',con_cols_to_choose, default=["Country","Household type",con_cols_to_choose[4]])

    st.markdown("---")
    remove_cols = st.multiselect('Remove landuse classes',lu_cols_in_use, default=['lu_open','lu_facilities'])
    if remove_cols:
        datac.drop(columns=remove_cols, inplace=True)
        lu_cols_in_use = [col for col in datac.columns if col.startswith('lu')]

    #reclassification
    s1,s2,s3,s4 = st.columns(4)
    combine_cols1 = s1.multiselect('Combine landuse classes',lu_cols_in_use)
    if len(combine_cols1) > 1:
        new_col_name = "_".join(combine_cols1)
        datac[new_col_name] = datac[combine_cols1].sum(axis=1)
        datac.drop(columns=combine_cols1, inplace=True)
        lu_cols_in_use = [col for col in datac.columns if col.startswith('lu')]
        
        #..add another comb set
        combine_cols2 = s2.multiselect('..Combine more',lu_cols_in_use)
        if len(combine_cols2) > 1:
            new_col_name2 = "_".join(combine_cols2)
            datac[new_col_name2] = datac[combine_cols2].sum(axis=1)
            datac.drop(columns=combine_cols2, inplace=True)
            lu_cols_in_use = [col for col in datac.columns if col.startswith('lu')]

            #..add another comb set
            combine_cols3 = s3.multiselect('..Combine more',lu_cols_in_use)
            if len(combine_cols3) > 1:
                new_col_name3 = "_".join(combine_cols3)
                datac[new_col_name3] = datac[combine_cols3].sum(axis=1)
                datac.drop(columns=combine_cols3, inplace=True)
                lu_cols_in_use = [col for col in datac.columns if col.startswith('lu')]

                #..add another comb set
                combine_cols4 = s4.multiselect('..Combine more',lu_cols_in_use)
                if len(combine_cols4) > 1:
                    new_col_name4 = "_".join(combine_cols4)
                    datac[new_col_name4] = datac[combine_cols4].sum(axis=1)
                    datac.drop(columns=combine_cols4, inplace=True)
                    lu_cols_in_use = [col for col in datac.columns if col.startswith('lu')]

    lu_cols = [col for col in datac.columns if col.startswith('lu')]

    s1,s2,s3 = st.columns(3)
    power_lucf = s1.toggle("Power transform distribution",value=True,
                           help="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.yeojohnson.html")
    norm = s2.radio("Normalization",['none','normalize','standardize'],horizontal=True)

    if power_lucf:
        for radius in [1,5,9]:
            df_r = datac[datac['R'] == f"R{radius}"]
            for col in lu_cols + cf_cols:
                df_r[col], lam = yeojohnson(df_r[col])
            datac.loc[datac['R'] == f"R{radius}", lu_cols] = df_r[lu_cols]
            datac.loc[datac['R'] == f"R{radius}", cf_cols] = df_r[cf_cols]

    
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
    cols_to_normalize = [col for col in datac.columns if col not in do_not_norm_cols]
    
    if norm == 'normalize':
        datac = normalize_df(df_in=datac,cols=cols_to_normalize)
    elif norm == 'standardize':
        datac = standardize_df(df_in=datac,cols=cols_to_normalize)

    
    #plot histos
    yksi,viisi,yhdeksan,cf = st.tabs(['1km','5km','9km','CF'])
    with yksi:
        histo_traces_lu = []
        df_r1 = datac[datac['R'] == f"R1"]
        for col in lu_cols:
            histo = go.Histogram(x=df_r1[col],opacity=0.75,name=col,nbinsx=20)
            histo_traces_lu.append(histo)
        layout_histo = go.Layout(title='Landuse type histograms',barmode='overlay')
        histo_fig_lu = go.Figure(data=histo_traces_lu, layout=layout_histo)
        st.plotly_chart(histo_fig_lu, use_container_width=True)
    with viisi:
        histo_traces_lu = []
        df_r = datac[datac['R'] == f"R5"]
        for col in lu_cols:
            histo = go.Histogram(x=df_r[col],opacity=0.75,name=col,nbinsx=20)
            histo_traces_lu.append(histo)
        layout_histo = go.Layout(title='Landuse type histograms',barmode='overlay')
        histo_fig_lu = go.Figure(data=histo_traces_lu, layout=layout_histo)
        st.plotly_chart(histo_fig_lu, use_container_width=True)
    with yhdeksan:
        histo_traces_lu = []
        df_r = datac[datac['R'] == f"R9"]
        for col in lu_cols:
            histo = go.Histogram(x=df_r[col],opacity=0.75,name=col,nbinsx=20)
            histo_traces_lu.append(histo)
        layout_histo = go.Layout(title='Landuse type histograms',barmode='overlay')
        histo_fig_lu = go.Figure(data=histo_traces_lu, layout=layout_histo)
        st.plotly_chart(histo_fig_lu, use_container_width=True)

    with cf:
        histo_traces_cf = []
        for col in cf_cols:
            histo = go.Histogram(x=datac[col],opacity=0.75,name=col,nbinsx=20)
            histo_traces_cf.append(histo)
        layout_histo = go.Layout(title='CF domain histograms',barmode='overlay')
        histo_fig_lu = go.Figure(data=histo_traces_cf, layout=layout_histo)
        st.plotly_chart(histo_fig_lu, use_container_width=True)




with st.expander(f'Regression tables', expanded=True):
    st.caption('Method for pearson: https://www.statsmodels.org/stable/example_formulas.html')
    st.caption('Method for partial spearman: https://pingouin-stats.org/build/html/generated/pingouin.partial_corr.html#pingouin.partial_corr')
    st.caption("Distributions power transformed (and standardized) for all non-categorical variables using https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.yeojohnson.html")

    def gen_reg_table(df_in, cf_col, base_cols, cat_cols, ext_cols):
        
        df_for_ols = df_in.copy()
        df_for_partial = df_in.copy()

        #OLS function
        def ols(df, cf_col, base_cols, cat_cols, ext_cols):

            df.columns = df.columns.str.lower().str.replace(' ', '_')
            domain_col = cf_col.lower().replace(' ', '_')
            cat_cols_lower = [col.lower().replace(' ', '_') for col in cat_cols]

            def format_col(col):
                col_lower = col.lower().replace(' ', '_').replace(',', '_').replace('&', '_')
                return f'C({col_lower})' if col_lower in cat_cols_lower else col_lower

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
        base_results, ext_results = ols(df_for_ols, cf_col, base_cols, cat_cols, ext_cols)

        #construct result df
        rounder = 3
        if base_results and ext_results:
            reg_results = pd.DataFrame({
                'base_β': round(base_results.params, rounder),
                'base_p': round(base_results.pvalues, rounder),
                'ext_β': round(ext_results.params, rounder),
                'ext_p': round(ext_results.pvalues, rounder)
                })
            
            r2_row = pd.DataFrame({
                'base_β': [round(base_results.rsquared_adj, rounder)],
                'base_p': [None],
                'ext_β': [round(ext_results.rsquared_adj, rounder)],
                'ext_p': [None]
            }, index=["rsqr_adj"])

            # Concatenate the results with the rsqr_adj row
            reg_results = pd.concat([r2_row,reg_results])

            # Calculate partial correlations for each land-use type
            partial_corr_results = []
            for landuse in ext_cols:
                partial_corr_result = pg.partial_corr(data=df_for_partial, x=landuse, y=cf_col, covar=control_cols, method="spearman")
                partial_corr = round(partial_corr_result['r'].values[0],rounder)
                partial_pval = round(partial_corr_result['p-val'].values[0],rounder)
                partial_corr_results.append((landuse, partial_corr, partial_pval))

            # Add partial correlation results to the DataFrame
            partial_corr_df = pd.DataFrame(partial_corr_results, columns=['landuse', 'partial_r', 'partial_p'])
            partial_corr_df.set_index('landuse', inplace=True)

            # Merge partial correlation results with the regression results
            reg_results = reg_results.join(partial_corr_df, how='outer')

            return reg_results

    yksi,viisi,yhdeksan = st.tabs(['1km','5km','9km'])
    with yksi:
        df_r1 = datac[datac['R'] == f"R1"]
        
        #table for r
        r1_lu_cols = [col for col in df_r1.columns if col.startswith('lu')]

        reg_df1 = gen_reg_table(df_in=df_r1.fillna(0),
                                cf_col=target_col,
                                base_cols=base_cols_orig + control_cols,
                                cat_cols=cat_cols,
                                ext_cols=r1_lu_cols
                                )
        st.data_editor(reg_df1,use_container_width=True, height=500,key=yksi)

    with viisi:
        df_r5 = datac[datac['R'] == f"R5"]
        #table for r
        r5_lu_cols = [col for col in df_r5.columns if col.startswith('lu')]
        reg_df5 = gen_reg_table(df_in=df_r5.fillna(0),
                                cf_col=target_col,
                                base_cols=base_cols_orig + control_cols,
                                cat_cols=cat_cols,
                                ext_cols=r5_lu_cols
                                )
        st.data_editor(reg_df5,use_container_width=True, height=500,key=viisi)

    with yhdeksan:
        df_r9 = datac[datac['R'] == f"R9"]
        #table for r
        r9_lu_cols = [col for col in df_r9.columns if col.startswith('lu')]
        reg_df9 = gen_reg_table(df_in=df_r9.fillna(0),
                                cf_col=target_col,
                                base_cols=base_cols_orig + control_cols,
                                cat_cols=cat_cols,
                                ext_cols=r9_lu_cols
                                )
        st.data_editor(reg_df9,use_container_width=True, height=500,key=yhdeksan)
    

    # plot all
    def plot_correlation(df_in,landuse_cols, alpha=0.06):

        results_df = df_in.copy()

        # Add opacity columns based on significance
        results_df["ext_opacity"] = np.where(results_df["ext_p"] < alpha, 1, 0.2)
        results_df["partial_opacity"] = np.where(results_df["partial_p"] < alpha, 1, 0.1)

        # Define colors for each correlation type
        colors = {"ext_β": "orange", "partial_r": "green"}

        # Create subplots for facets based on 'R' column (R1 and R5)
        fig = make_subplots(
            rows=1, 
            cols=2, 
            shared_yaxes=True,   # If you want shared y-axis across facets
            subplot_titles=["R1", "R5"],  # Titles for facets
            column_widths=[0.5, 0.5]   # Equal width for both facets
        )

        # Loop through correlation types and add traces for each facet
        for corr_type, opacity_col in [
            ("ext_β", "ext_opacity"),
            ("partial_r", "partial_opacity")
        ]:
            # Loop through the values of 'R' (R1 and R5)
            for idx, r_value in enumerate(["R1", "R5"], start=1):
                # Filter data by 'R' column
                res_r = results_df[results_df["R"] == r_value]
                filtered_data = res_r[res_r['index'].isin(landuse_cols)]
                filtered_data.rename(columns={'index':'variable'}, inplace=True)
                
                # Add trace for each correlation type and facet
                fig.add_trace(go.Bar(
                    x=filtered_data['variable'],
                    y=filtered_data[corr_type],
                    name=f"{corr_type} ({r_value})",
                    marker=dict(color=colors[corr_type], opacity=filtered_data[opacity_col]),
                ), row=1, col=idx)

        # Update layout for better presentation
        desired_order = sorted(landuse_cols, reverse=True)
        fig.update_layout(
            xaxis=dict(categoryorder="array", categoryarray=desired_order),
            xaxis2=dict(categoryorder="array", categoryarray=desired_order),
            title=f"{target_col}",
            barmode="group",
            xaxis_title="Land-Use Type",
            yaxis_title="Metric Value",
            height=500,
            yaxis=dict(range=[-0.8, 0.8]),
            yaxis2=dict(range=[-0.8, 0.8])
        )

        return fig

    reg_df1['R'] = "R1"
    reg_df5['R'] = "R5"
    reg_df9['R'] = "R9"
    reg_df_all = pd.concat([reg_df1,reg_df5,reg_df9])
    cols = reg_df_all.columns.tolist()
    new_cols = cols[-1:] + cols[:-1]
    reg_df_all = reg_df_all[new_cols].reset_index()

    #PLOT
    fig = plot_correlation(reg_df_all,landuse_cols=lu_cols,alpha=0.05)
    st.plotly_chart(fig,use_container_width=True)

    def gen_reg_df_all():

        reg_dfs = []
        for r in datac['R'].unique().tolist():
            df_r = datac[datac['R'] == r]
            for target  in cf_cols:
                reg_df = gen_reg_table(df_in=df_r.fillna(0),
                                        cf_col=target,
                                        base_cols=base_cols_orig + control_cols,
                                        cat_cols=cat_cols,
                                        ext_cols=r9_lu_cols
                                        )
                reg_df['Radius'] = r
                reg_df['Domain'] = target
                reg_dfs.append(reg_df)
            
        reg_all = pd.concat(reg_dfs)

        if cluster != "None":
            setup = [f'Cluster_level:{cluster}',f'Min_cluster_size:{min_cluster_size}',
                    f'N={len(datac)}',f'Control_cols:{control_cols}',f'Lu_cols:{lu_cols}',f'norm:{norm}']
        else:
            setup = [f'Cluster_level:{cluster}',
                    f'N={len(datac)}',f'Control_cols:{control_cols}',f'Lu_cols:{lu_cols}',f'norm:{norm}']

        st.download_button(
                            label=f"Download study as CSV",
                            data=reg_all.to_csv().encode("utf-8"),
                            file_name=f"cfua_data_SETUP:{setup}.csv",
                            mime="text/csv",
                            )
            
    gen_reg_df_all()
