# NDP app always beta a lot
import streamlit as st
import pandas as pd
import numpy as np
import duckdb
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.stats import yeojohnson

#modules for stats etc
from regs import clusterize, gen_reg_table_multitarget, partial_corr_table_v3, box_plot, interaction_scan

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
                    SELECT *, 'R1' as R FROM read_parquet('{allas_url}/CFUA/RD/cfua_data_nordics_outliers_Q01+iqr_R40-10_reso10_ND1km_F.parquet')
                    UNION ALL
                    SELECT *, 'R5' as R FROM read_parquet('{allas_url}/CFUA/RD/cfua_data_nordics_outliers_Q01+iqr_R40-10_reso10_ND5km_F.parquet')
                    UNION ALL
                    SELECT *, 'R9' as R FROM read_parquet('{allas_url}/CFUA/RD/cfua_data_nordics_outliers_Q01+iqr_R40-10_reso10_ND9km_F.parquet')
                """
            duckdb.sql(combined_table)
            dropcols = ['__index_level_0__']
            df = duckdb.sql("SELECT * FROM combined_table").to_df()#.drop(columns=dropcols)
            
            cols = ['R'] + [col for col in df.columns if col != 'R']
            df_out = df[cols]
        else:
            df_out = duckdb.sql(f"SELECT * FROM read_parquet('{allas_url}/CFUA/RD/cfua_data_nordics_IQR1-5_R30-10_reso10_ND{R}km_F.parquet')").to_df().drop(columns=['__index_level_0__'])
        
        #renames
        df_out.rename(columns={'Vehicle possession footprint':'Vehicle footprint'}, inplace=True)
        cf_cols = ['Housing footprint',
                'Diet footprint',
                'Vehicle footprint',
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

    data_orig = load_data()
    data = data_orig.copy()

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

    dbstatus.update(label="DB connected!", state="complete", expanded=False)

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

        #viz with bars
        def country_plot(df_in,cf_cols,city='fua_name'):
            df = df_in.copy()
            cols_map = {col: col.removeprefix('cf_') for col in cf_cols if col not in ['cf_Total footprint unit']}
            cols = list(cols_map.values())
            df.rename(columns=cols_map, inplace=True)

            # compute N values
            df_counts = df.groupby(city).size().rename("N").reset_index()

            # Group means
            df_grouped = df.groupby(city)[cols].mean().reset_index()

            # Merge in N
            df_grouped = df_grouped.merge(df_counts, on=city, how='left')

            # Create city labels with N underneath
            df_grouped["city_label"] = df_grouped[city] + "<br><span style='font-size:10px'>N=" + df_grouped["N"].astype(str) + "</span>"

            cf_cols_filt = [col for col in cols if col != 'Total footprint']

            df_melted = df_grouped.melt(id_vars='city_label', value_vars=cf_cols_filt, 
                                        var_name='Carbon Footprint Type', value_name='Value')
            
            earthy_sky_colors = [
                "#FFD700",  # Golden Yellow (Sun)
                "#E97451",  # Burnt Sienna (Earth)
                "#468FAF",  # Sky Blue (Sky)
                "#87CEEB",  # Light Sky Blue (Sky)
                "#588157",  # Forest Green (Earth)
                "#A3B18A",  # Soft Olive (Earth)
                "#D4A373",  # Warm Sand (Earth)
                "#8B5A2B",  # Saddle Brown (Earth)
            ]
            myorder = [
                'Goods and services footprint',
                'Leisure travel footprint',
                'Vehicle footprint',
                'Public transportation footprint',
                'Housing footprint',
                'Diet footprint',
                'Pets footprint',
                'Summer house footprint'
            ]
            fig = px.bar(
                df_melted,
                x="city_label",
                y='Value',
                color='Carbon Footprint Type',
                title="Carbon Footprint Breakdown by City",
                labels={'Value': 'Carbon Footprint','city_label':'City'},
                category_orders={'Carbon Footprint Type': myorder},
                color_discrete_sequence=earthy_sky_colors
                # px.colors.qualitative.Vivid
                # https://plotly.com/python/discrete-color/
            )

            for i, row in df_grouped.iterrows():
                row['Total footprint'] = round(row['Total footprint']/1000,1)
                fig.add_annotation(
                    x=row[city],
                    y=50 * row['Total footprint'],
                    text=f"{row['Total footprint']} tCO2e", #:.2f
                    showarrow=False,
                    font=dict(size=12, color='black'),
                    yshift=5
                )

            # Make sure bars are stacked and ordered by total descending
            fig.update_layout(barmode='stack', xaxis={'categoryorder': 'total descending'})

            return fig, df_melted
        
        coutry_fig, df_melted = country_plot(df_in=datac[datac['R'] == 'R1'],cf_cols=cf_cols,city='fua_name')
        
        st.plotly_chart(coutry_fig, use_container_width=True)

        df_melted_enh = df_melted.copy()
        df_melted_enh['city_label'] = df_melted_enh['city_label'].str.replace("<br><span style='font-size:10px'>N="," (N=").str.replace("</span>",")")
        df_melted_enh['share_within_city_%'] = round(df_melted_enh['Value']/df_melted_enh.groupby('city_label')['Value'].transform('sum') * 100, 2)

        st.markdown("**Carbon footprint shares within case cities (%)**")
        st.data_editor(df_melted_enh,key="df_melted_editor",height=300)

        #check only with one radius data
        yks,viis,ysi = st.tabs(['1km','5km','9km'])
        yks.data_editor(datac[datac['R'] == 'R1'].describe())
        viis.data_editor(datac[datac['R'] == 'R5'].describe())
        ysi.data_editor(datac[datac['R'] == 'R9'].describe())


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



#reg tests in tabs

ve1, ve2 = st.tabs(['EDA tests','Partial corr method'])

with ve1:
    data1 = datac.copy()

    with st.expander(f'Regression settings', expanded=False):

        #reclassification
        use_predefined = st.toggle('Use pre-defined settings',value=True)

        ## ------- reg study ---------
        default_target_domains = ['cf_Goods and services footprint',
                                'cf_Leisure travel footprint',
                                'cf_Vehicle footprint',
                                'cf_Total footprint'
                                ]
        base_cols_orig = ['Age']

        if use_predefined:
            target_cols = default_target_domains
            control_cols = ['Household per capita income decile ORIG','Household type']
            #remove_cols = ['lu_open']
            #data1.drop(columns=remove_cols, inplace=True)
            combine_cols1 = ['lu_parks','lu_forest','lu_open']
            #combine_cols2 = ['lu_parks','lu_forest']
            new_col_name1 = "_".join(combine_cols1)
            #new_col_name2 = "_".join(combine_cols2)
            data1[new_col_name1] = data1[combine_cols1].sum(axis=1)
            #data1[new_col_name2] = data1[combine_cols2].sum(axis=1)
            data1.drop(columns=combine_cols1, inplace=True)
            #data1.drop(columns=combine_cols2, inplace=True)
            lu_cols_in_use = [col for col in data1.columns if col.startswith('lu')]
            lu_premap = {'lu_urban':'Urban',
                        'lu_modern':'Modern',
                        'lu_suburb':'Suburban',
                        'lu_exurb':'Exurban',
                        'lu_sports':'Recreation',
                        'lu_parks_lu_forest_lu_open':'Green',
                        'lu_facilities':'Facilities'
                        }
            preset_order = ['Urban','Facilities','Recreation','Suburban','Exurban','Green']
            power_lucf = True
            norm = 'standardize'

            s1,s2 = st.columns(2)
            s1.table(lu_cols_map)
            s2.table(lu_premap)


        else:
            target_cols = st.multiselect("Target domains",cf_cols,default=default_target_domains, max_selections=5)

            #base cols
            if "cf_Total footprint unit" in target_cols:
                con_cols_to_choose = ['Country','Education level','Household type','Car in household','Household unit income decile ORIG']
            else:
                con_cols_to_choose = ['Country','Education level','Household type','Car in household','Household per capita income decile ORIG']
            
            #multiselect cat cols to use = control
            control_cols = st.multiselect('Control columns',con_cols_to_choose, default=["Country","Household type",con_cols_to_choose[4]])

            remove_cols = st.multiselect('Remove landuse classes',lu_cols_in_use, default=['lu_open','lu_facilities','lu_forest'])
            if remove_cols:
                data1.drop(columns=remove_cols, inplace=True)
                lu_cols_in_use = [col for col in data1.columns if col.startswith('lu')]
            s1,s2,s3,s4 = st.columns(4)
            combine_cols1 = s1.multiselect('Combine landuse classes',lu_cols_in_use)
        
            if len(combine_cols1) > 1:
                new_col_name = "_".join(combine_cols1)
                data1[new_col_name] = data1[combine_cols1].sum(axis=1)
                data1.drop(columns=combine_cols1, inplace=True)
                lu_cols_in_use = [col for col in data1.columns if col.startswith('lu')]
                
                #..add another comb set
                combine_cols2 = s2.multiselect('..Combine more',lu_cols_in_use)
                if len(combine_cols2) > 1:
                    new_col_name2 = "_".join(combine_cols2)
                    data1[new_col_name2] = data1[combine_cols2].sum(axis=1)
                    data1.drop(columns=combine_cols2, inplace=True)
                    lu_cols_in_use = [col for col in data1.columns if col.startswith('lu')]

                    #..add another comb set
                    combine_cols3 = s3.multiselect('..Combine more',lu_cols_in_use)
                    if len(combine_cols3) > 1:
                        new_col_name3 = "_".join(combine_cols3)
                        data1[new_col_name3] = data1[combine_cols3].sum(axis=1)
                        data1.drop(columns=combine_cols3, inplace=True)
                        lu_cols_in_use = [col for col in data1.columns if col.startswith('lu')]

                        #..add another comb set
                        combine_cols4 = s4.multiselect('..Combine more',lu_cols_in_use)
                        if len(combine_cols4) > 1:
                            new_col_name4 = "_".join(combine_cols4)
                            data1[new_col_name4] = data1[combine_cols4].sum(axis=1)
                            data1.drop(columns=combine_cols4, inplace=True)
                            lu_cols_in_use = [col for col in data1.columns if col.startswith('lu')]

            #distribution settings
            power_lucf = st.toggle("Power transform distribution",value=True,
                                help="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.yeojohnson.html")
            norm = st.radio("Normalization",['none','normalize','standardize'],index=2,horizontal=True)
            st.caption('Method for pearson: https://www.statsmodels.org/stable/example_formulas.html')
            st.caption('Method for partial spearman: https://pingouin-stats.org/build/html/generated/pingouin.partial_corr.html#pingouin.partial_corr')
            st.caption("Distributions power transformed (and standardized) for all non-categorical variables using https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.yeojohnson.html")

        lu_cols = [col for col in data1.columns if col.startswith('lu')]

        if power_lucf:
            for radius in [1,5,9]:
                df_r = data1[data1['R'] == f"R{radius}"]
                for col in lu_cols + cf_cols:
                    df_r[col], lam = yeojohnson(df_r[col])
                data1.loc[data1['R'] == f"R{radius}", lu_cols] = df_r[lu_cols]
                data1.loc[data1['R'] == f"R{radius}", cf_cols] = df_r[cf_cols]


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
        cols_to_normalize = [col for col in data1.columns if col not in do_not_norm_cols]
        
        if norm == 'normalize':
            data1 = normalize_df(df_in=data1,cols=cols_to_normalize)
        elif norm == 'standardize':
            data1 = standardize_df(df_in=data1,cols=cols_to_normalize)

        if (norm == 'none') and (not power_lucf):
            show_shares = True
        else:
            show_shares = False
        
        #plot histos
        if not use_predefined:
            st.markdown("---")
            if st.toggle('Show histograms'):
                yksi,viisi,yhdeksan,cf = st.tabs(['1km','5km','9km','CF'])
                with yksi:
                    histo_traces_lu = []
                    df_r1 = data1[data1['R'] == f"R1"]
                    trueN = len(df_r1)
                    for col in lu_cols:
                        histo = go.Histogram(x=df_r1[col],opacity=0.75,name=col,nbinsx=20)
                        histo_traces_lu.append(histo)
                    layout_histo = go.Layout(title=f'Landuse type histograms {trueN}',barmode='overlay')
                    histo_fig_lu = go.Figure(data=histo_traces_lu, layout=layout_histo)
                    st.plotly_chart(histo_fig_lu, use_container_width=True)
                    st.data_editor(df_r1.describe())
                    st.markdown("---")

                    box_fig1  = box_plot(df=df_r1,group_col="fua_name",total_col="cf_Total footprint",cf_cols=target_cols,shares=show_shares)
                    st.plotly_chart(box_fig1, use_container_width=True, key='b1')

                with viisi:
                    histo_traces_lu = []
                    df_r5 = data1[data1['R'] == f"R5"]
                    for col in lu_cols:
                        histo = go.Histogram(x=df_r5[col],opacity=0.75,name=col,nbinsx=20)
                        histo_traces_lu.append(histo)
                    layout_histo = go.Layout(title='Landuse type histograms',barmode='overlay')
                    histo_fig_lu = go.Figure(data=histo_traces_lu, layout=layout_histo)
                    st.plotly_chart(histo_fig_lu, use_container_width=True)
                    st.data_editor(df_r5.describe())
                    st.markdown("---")

                    box_fig5  = box_plot(df=df_r5,group_col="fua_name",total_col="cf_Total footprint",cf_cols=target_cols,shares=show_shares)
                    st.plotly_chart(box_fig5, use_container_width=True, key='b5')

                with yhdeksan:
                    histo_traces_lu = []
                    df_r9 = data1[data1['R'] == f"R9"]
                    for col in lu_cols:
                        histo = go.Histogram(x=df_r9[col],opacity=0.75,name=col,nbinsx=20)
                        histo_traces_lu.append(histo)
                    layout_histo = go.Layout(title='Landuse type histograms',barmode='overlay')
                    histo_fig_lu = go.Figure(data=histo_traces_lu, layout=layout_histo)
                    st.plotly_chart(histo_fig_lu, use_container_width=True)
                    st.data_editor(df_r9.describe())
                    st.markdown("---")

                    box_fig9  = box_plot(df=df_r9,group_col="fua_name",total_col="cf_Total footprint",cf_cols=target_cols,shares=show_shares)
                    st.plotly_chart(box_fig9, use_container_width=True, key='b9')

                with cf:
                    histo_traces_cf = []
                    for col in cf_cols:
                        histo = go.Histogram(x=data1[col],opacity=0.75,name=col,nbinsx=20)
                        histo_traces_cf.append(histo)
                    layout_histo = go.Layout(title='CF domain histograms',barmode='overlay')
                    histo_fig_lu = go.Figure(data=histo_traces_cf, layout=layout_histo)
                    st.plotly_chart(histo_fig_lu, use_container_width=True)


    if len(target_cols) == 0 or len(target_cities) == 0:
        st.stop()
    ## ------- reg tables ---------
    with st.expander(f'Regression tables', expanded=False):
        

        if len(target_cols) >= 1:

            yksi,viisi,yhdeksan = st.tabs(['1km','5km','9km'])
            with yksi:
                df_r1 = data1[data1['R'] == f"R1"]

                r1_lu_cols = [col for col in df_r1.columns if col.startswith('lu')]
                
                reg_df1 = gen_reg_table_multitarget(data=df_r1.fillna(0),
                                        target_cf_cols=target_cols,
                                        base_cols=base_cols_orig + control_cols,
                                        cat_cols=cat_cols,
                                        ext_cols=r1_lu_cols,
                                        control_cols=control_cols
                                        )
                if use_predefined:
                    reg_df1.rename(index=lu_premap, inplace=True)
                st.data_editor(reg_df1.drop(columns='Radius'),use_container_width=True, height=500,key=yksi)
        
        with viisi:
            df_r5 = data1[data1['R'] == f"R5"]
            r5_lu_cols = [col for col in df_r5.columns if col.startswith('lu')]
            reg_df5 = gen_reg_table_multitarget(data=df_r5.fillna(0),
                                        target_cf_cols=target_cols,
                                        base_cols=base_cols_orig + control_cols,
                                        cat_cols=cat_cols,
                                        ext_cols=r5_lu_cols,
                                        control_cols=control_cols
                                        )
            if use_predefined:
                reg_df5.rename(index=lu_premap, inplace=True)
            st.data_editor(reg_df5.drop(columns='Radius'),use_container_width=True, height=500,key=viisi)

        with yhdeksan:
            df_r9 = data1[data1['R'] == f"R9"]
            r9_lu_cols = [col for col in df_r9.columns if col.startswith('lu')]
            reg_df9 = gen_reg_table_multitarget(data=df_r9.fillna(0),
                                        target_cf_cols=target_cols,
                                        base_cols=base_cols_orig + control_cols,
                                        cat_cols=cat_cols,
                                        ext_cols=r9_lu_cols,
                                        control_cols=control_cols
                                        )
            if use_predefined:
                reg_df9.rename(index=lu_premap, inplace=True)
            st.data_editor(reg_df9.drop(columns='Radius'),use_container_width=True, height=500,key=yhdeksan)


    with st.expander(f'Regression plots', expanded=False):
        
        def plot_partial_r_with_beta_text(df_in, landuse_cols, alpha=0.06):
            results_df = df_in.copy()

            # Add opacity columns based on significance
            results_df["partial_opacity"] = np.where(results_df["partial_p"] < alpha, 1, 0.1)
            results_df["beta_text_color"] = np.where(results_df["ext_p"] < alpha, "black", "lightgrey")

            # Create subplots for facets based on 'R' column (R1 and R5)
            fig = make_subplots(
                rows=1, 
                cols=2, 
                shared_yaxes=True,
                subplot_titles=["Radius 1km", "Radius 5km"],
                column_widths=[0.5, 0.5]
            )

            # Loop cf domains..
            cf_domains = results_df['Domain'].unique().tolist()
            # Define colors
            partial_colors = {"cf_Vehicle footprint":"skyblue",
                            "cf_Goods and services footprint":"orange",
                            "cf_Leisure travel footprint":"red"}

            domain_offsets = np.linspace(-0.3, 0.3, len(cf_domains))  # Spread offsets evenly around center

            # Loop through each domain and R value
            for dom_idx, dom in enumerate(cf_domains):
                results_df_dom = results_df[results_df['Domain'] == dom]
                partial_color = partial_colors.get(dom, 'grey')

                for idx, r_value in enumerate(["R1", "R5"], start=1):
                    res_r = results_df_dom[results_df_dom["R"] == r_value]
                    filtered_data = res_r[res_r['index'].isin(landuse_cols)]
                    filtered_data.rename(columns={'index': 'variable'}, inplace=True)

                    # Add bar trace for each domain
                    fig.add_trace(go.Bar(
                        x=filtered_data['variable'],
                        y=filtered_data['partial_r'],
                        name=f"{dom} ({r_value})",
                        marker=dict(color=partial_color, opacity=filtered_data["partial_opacity"]),
                    ), row=1, col=idx)

                    # Annotate each bar with the corresponding beta value
                    for var, y_val, beta_val, color in zip(
                        filtered_data['variable'], 
                        filtered_data['partial_r'], 
                        filtered_data['ext_β'], 
                        filtered_data['beta_text_color']
                    ):
                        fig.add_annotation(
                            x=var,
                            y=y_val + (0.05 * np.sign(y_val)),  # Vertical offset
                            text=f"{beta_val:.2f}",
                            showarrow=False,
                            row=1,
                            col=idx,
                            font=dict(color=color, size=12),
                            yanchor="bottom" if y_val >= 0 else "top",
                            xref=f"x{idx}",  # Ensure correct x-axis per subplot
                            xshift=domain_offsets[dom_idx] * 100  # Horizontal shift as a percentage
                        )


            # Update layout for better presentation
            if use_predefined:
                desired_order = preset_order
            else:
                desired_order = sorted(landuse_cols, reverse=True)

            fig.update_layout(
                xaxis=dict(categoryorder="array", categoryarray=desired_order),
                xaxis2=dict(categoryorder="array", categoryarray=desired_order),
                title=f"Partial r with β (greyed if insignificant)",
                barmode="group",
                xaxis_title=None,
                yaxis_title="Correlation",
                height=500,
                yaxis=dict(range=[-0.5, 0.5]),
                yaxis2=dict(range=[-0.5, 0.5]),
                showlegend=True
            )

            return fig

        reg_df1['R'] = "R1"
        reg_df5['R'] = "R5"
        reg_df9['R'] = "R9"
        reg_df_all = pd.concat([reg_df1,reg_df5,reg_df9])
        cols = reg_df_all.columns.tolist()
        new_cols = cols[-1:] + cols[:-1]
        reg_df_all = reg_df_all[new_cols].reset_index()
        
        #st.data_editor(reg_df_all)
        #st.stop()

        #PLOT
        if use_predefined:
            lu_cols_for_plot = preset_order
        else:
            lu_cols_for_plot = lu_cols

        fig = plot_partial_r_with_beta_text(reg_df_all,landuse_cols=lu_cols_for_plot,alpha=0.05)
        st.plotly_chart(fig,use_container_width=True)

        

        if cluster != "None":
            setup = [f'Cluster_level:{cluster}',f'Min_cluster_size:{min_cluster_size}',
                    f'N={len(data1)}',f'Control_cols:{control_cols}',f'Lu_cols:{lu_cols}',f'norm:{norm}']
        else:
            setup = [f'Cluster_level:{cluster}',
                    f'N={len(data1)}',f'Control_cols:{control_cols}',f'Lu_cols:{lu_cols}',f'norm:{norm}']

        st.download_button(
                            label=f"Download study as CSV",
                            data=reg_df_all.to_csv().encode("utf-8"),
                            file_name=f"cfua_data_SETUP:{setup}.csv",
                            mime="text/csv",
                            )

with ve2:
    
    use_predefined = st.toggle('Use pre-defined',value=True)

    if use_predefined:
        data2 = datac.copy()
        default_target_domains = ['cf_Goods and services footprint',
                                'cf_Leisure travel footprint',
                                'cf_Vehicle footprint',
                                'cf_Total footprint'
                                ]
        target_col = st.selectbox('Target cf domain',cf_cols)
        control_cols = ['Household per capita income decile ORIG','Household type']
        combine_cols1 = ['lu_parks','lu_forest','lu_open']
        new_col_name1 = "_".join(combine_cols1)
        data2[new_col_name1] = data2[combine_cols1].sum(axis=1)
        data2.drop(columns=combine_cols1, inplace=True)
        lu_cols_in_use = [col for col in data2.columns if col.startswith('lu')]

        lu_premap = {'lu_urban':'Urban',
                        'lu_modern':'Modern',
                        'lu_suburb':'Suburban',
                        'lu_exurb':'Exurban',
                        'lu_sports':'Recreation',
                        'lu_parks_lu_forest_lu_open':'Green',
                        'lu_facilities':'Facilities'
                    }
        preset_order = ['Urban','Facilities','Recreation','Suburban','Exurban','Green']

    else:
        data2 = datac.copy()
        lu_cols_in_use = [col for col in data2.columns if col.startswith('lu')]
        with st.container():
            target_col = st.selectbox('Target cf domain',cf_cols)
            remove_cols = st.multiselect('Remove landuse classes',lu_cols_in_use)
            if remove_cols:
                data2.drop(columns=remove_cols, inplace=True)
                lu_cols_in_use = [col for col in data2.columns if col.startswith('lu')]
            s1,s2,s3,s4 = st.columns(4)
            combine_cols1 = s1.multiselect('Combine landuse classes',lu_cols_in_use)
        
            if len(combine_cols1) > 1:
                new_col_name = "_".join(combine_cols1)
                data2[new_col_name] = data2[combine_cols1].sum(axis=1)
                data2.drop(columns=combine_cols1, inplace=True)
                lu_cols_in_use = [col for col in data2.columns if col.startswith('lu')]
                
                #..add another comb set
                combine_cols2 = s2.multiselect('..Combine more',lu_cols_in_use)
                if len(combine_cols2) > 1:
                    new_col_name2 = "_".join(combine_cols2)
                    data2[new_col_name2] = data2[combine_cols2].sum(axis=1)
                    data2.drop(columns=combine_cols2, inplace=True)
                    lu_cols_in_use = [col for col in data2.columns if col.startswith('lu')]

                    #..add another comb set
                    combine_cols3 = s3.multiselect('..Combine more',lu_cols_in_use)
                    if len(combine_cols3) > 1:
                        new_col_name3 = "_".join(combine_cols3)
                        data2[new_col_name3] = data2[combine_cols3].sum(axis=1)
                        data2.drop(columns=combine_cols3, inplace=True)
                        lu_cols_in_use = [col for col in data2.columns if col.startswith('lu')]

                        #..add another comb set
                        combine_cols4 = s4.multiselect('..Combine more',lu_cols_in_use)
                        if len(combine_cols4) > 1:
                            new_col_name4 = "_".join(combine_cols4)
                            data2[new_col_name4] = data2[combine_cols4].sum(axis=1)
                            data2.drop(columns=combine_cols4, inplace=True)
                            lu_cols_in_use = [col for col in data2.columns if col.startswith('lu')]

    
    r1,r5,r9 = st.tabs(['1km','5km','9km'])
    with r1:
        df_r1 = data2[data2['R'] == f"R1"]
        r1_lu_cols = [col for col in df_r1.columns if col.startswith('lu')]
        res1 = partial_corr_table_v3(
                                    df=df_r1,
                                    predictor_cols=r1_lu_cols,
                                    target_col=target_col,
                                    covar_cols=control_cols
                                    )
        
        if use_predefined:
            res1['Variable'] = res1['Variable'].map(lu_premap)
            res1.set_index('Variable', inplace=True)
            res1 = res1.reindex(preset_order)

        st.data_editor(res1, use_container_width=True)

        #interactions
        if st.toggle('Show interactions',key='int1'):
            int1 = interaction_scan(df=df_r1, target_col=target_col, predictors=r1_lu_cols)
            st.data_editor(int1, use_container_width=True)


    with r5:
        df_r5 = data2[data2['R'] == f"R5"]
        r5_lu_cols = [col for col in df_r5.columns if col.startswith('lu')]
        res5 = partial_corr_table_v3(
                                    df=df_r5,
                                    predictor_cols=r5_lu_cols,
                                    target_col=target_col,
                                    covar_cols=control_cols
                                )
        
        if use_predefined:
            res5['Variable'] = res5['Variable'].map(lu_premap)
            res5.set_index('Variable', inplace=True)
            res5 = res5.reindex(preset_order)

        st.data_editor(res5, use_container_width=True)

        #interactions
        if st.toggle('Show interactions',key='int5'):
            int5 = interaction_scan(df=df_r5, target_col=target_col, predictors=r5_lu_cols)
            st.data_editor(int5, use_container_width=True)
    
    with r9:
        df_r9 = data2[data2['R'] == f"R9"]
        r9_lu_cols = [col for col in df_r9.columns if col.startswith('lu')]
        res9 = partial_corr_table_v3(
                                    df=df_r9,
                                    predictor_cols=r9_lu_cols,
                                    target_col=target_col,
                                    covar_cols=control_cols
                                )
        
        if use_predefined:
            res9['Variable'] = res9['Variable'].map(lu_premap)
            res9.set_index('Variable', inplace=True)
            res9 = res9.reindex(preset_order)

        st.data_editor(res9, use_container_width=True)

        #interactions
        if st.toggle('Show interactions',key='int9'):
            int9 = interaction_scan(df=df_r9, target_col=target_col, predictors=r9_lu_cols)
            st.data_editor(int9, use_container_width=True)
    
    if use_predefined:
        with st.expander('Plot'):
            
            if st.toggle('Generate comparison plot'):
                cf_targets = ["cf_Vehicle footprint","cf_Goods and services footprint","cf_Leisure travel footprint","cf_Total footprint"]

                partial_df_all = pd.DataFrame()
                for r in [1,5]:
                    df_r = data2[data2['R'] == f"R{r}"]
                    
                    r_lu_cols = [col for col in df_r.columns if col.startswith('lu')]
                    for target in cf_targets:
                        res = partial_corr_table_v3(
                                                df=df_r,
                                                predictor_cols=r_lu_cols,
                                                target_col=target,
                                                covar_cols=control_cols
                                            )
                        res['Domain'] = target
                        res['R'] = f"R{r}"

                        partial_df_all = pd.concat([partial_df_all,res])

                if use_predefined:
                    partial_df_all['Land-use'] = partial_df_all['Variable'].map(lu_premap)
                    partial_df_all = partial_df_all.set_index('Land-use').drop(columns='Variable')
                else:
                    lu_cols_for_partial_plot = [col for col in partial_df_all.columns if col.startswith('lu')]

            else:
                st.stop()

            @st.cache_data()
            def plot_partial_corr(df_in, r_value="R1", alpha=0.06):
                results_df = df_in.copy().reset_index()
                my_order = ['Urban','Facilities','Recreation','Suburban','Exurban','Green']
                
                # Convert Land-use to categorical with specified order
                results_df['Land-use'] = pd.Categorical(
                    results_df['Land-use'], 
                    categories=my_order, 
                    ordered=True
                )
                
                # Create single plot with secondary y-axis
                fig = make_subplots(specs=[[{"secondary_y": True}]])

                # Color mapping - same for bars and lines
                domain_colors = {
                    "cf_Vehicle footprint": "#468FAF",  # Sky Blue (Sky)
                    "cf_Goods and services footprint": "#FFD700",  # Golden Yellow (Sun)
                    "cf_Leisure travel footprint": "#E97451", #Burnt Sienna (Earth)
                    "cf_Total footprint": "grey"
                }

                for dom in results_df['Domain'].unique():
                    # Filter and sort data
                    df = results_df[(results_df['Domain'] == dom) & (results_df['R'] == r_value)]
                    df = df.sort_values('Land-use')
                    
                    domain_color = domain_colors[dom]
                    spearman_significant = df['Spearman_p'] < alpha
                    pearson_significant = df['Pearson_p'] < alpha
                    
                    display_name = dom[3:] if dom.startswith("cf_") else dom

                    # Spearman bars (primary y-axis)
                    fig.add_trace(go.Bar(
                        x=df['Land-use'],
                        y=df['Spearman_r'],
                        name=display_name,
                        marker_color=domain_color,
                        marker_opacity=np.where(spearman_significant, 1, 0.25).tolist(),
                        legendgroup=dom
                    ), secondary_y=False)

                    # Pearson line (secondary y-axis)
                    fig.add_trace(go.Scatter(
                        x=df['Land-use'],
                        y=df['Pearson_r'],
                        mode='lines+markers',
                        line=dict(
                            color=domain_color,
                            width=1,
                            dash='dash'
                        ),
                        opacity=np.where(pearson_significant.any(), 1, 0.25).item(),  # Move opacity to trace level
                        marker=dict(
                            color=np.where(pearson_significant, domain_color, 'lightgrey'),
                            opacity=np.where(pearson_significant, 1, 0.25),
                            size=8,
                            line=dict(width=1, color='black')
                        ),
                        name=f"{dom} (Pearson)",
                        showlegend=False,
                        legendgroup=f"{dom}_pearson"
                    ), secondary_y=True)

                # Update layout
                fig.update_layout(
                    barmode='group',
                    height=500,
                    title=f"Partial correlations (Radius {r_value[1:]}km)",
                    legend=dict(orientation="h", y=1.1),
                    xaxis=dict(
                        title="Land-use type",
                        type='category',
                        categoryorder='array',
                        categoryarray=my_order,
                        tickmode='array',
                        tickvals=my_order
                    ),
                    yaxis=dict(
                        title="Spearman r (bars)",
                        range=[-0.5, 0.5]
                    ),
                    yaxis2=dict(
                        title="Pearson r (lines)",
                        range=[-0.5, 0.5],
                        overlaying="y",
                        side="right"
                    )
                )

                return fig

            
            #st.data_editor(partial_df_all)
            #st.stop()

            #p1,p2 = st.columns(2)

            fig_partial_1km = plot_partial_corr(df_in=partial_df_all, r_value="R1", alpha=0.06)
            st.plotly_chart(fig_partial_1km,use_container_width=True)
            
            fig_partial_5km = plot_partial_corr(df_in=partial_df_all, r_value="R5", alpha=0.06)
            st.plotly_chart(fig_partial_5km,use_container_width=True)