# NDP app always beta a lot
import streamlit as st
import pandas as pd
import h3
import numpy as np
import duckdb
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from collections import Counter, defaultdict

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

        #viz with bars
        def country_plot(df_in,cf_cols,city='fua_name'):
            df = df_in.copy()
            cols_map = {col: col.removeprefix('cf_') for col in cf_cols if col not in ['cf_Total footprint unit']}
            cols = list(cols_map.values())
            df.rename(columns=cols_map, inplace=True)
            df_grouped = df.groupby(city)[cols].mean().reset_index()
            cf_cols_filt = [col for col in cols if col != 'Total footprint']
            df_melted = df_grouped.melt(id_vars='fua_name', value_vars=cf_cols_filt, 
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
                x=city,
                y='Value',
                color='Carbon Footprint Type',
                title="Carbon Footprint Breakdown by City",
                labels={'Value': 'Carbon Footprint','fua_name':'City'},
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

            return fig
        
        st.plotly_chart(country_plot(df_in=datac[datac['R'] == 'R1'],cf_cols=cf_cols,city='fua_name'), use_container_width=True)

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


def box_plot(df,group_col,total_col,cf_cols,shares=True):
    df_melted = df.melt(id_vars=[group_col, total_col], value_vars=cf_cols, 
                var_name='CF_Domain', value_name='Carbon_Footprint')
    
    hel_N = len(df[df['fua_name'] == 'Helsinki'])
    sto_N = len(df[df['fua_name'] == 'Stockholm'])
    cop_N = len(df[df['fua_name'] == 'København'])
    osl_N = len(df[df['fua_name'] == 'Oslo'])
    rey_N = len(df[df['fua_name'] == 'Reykjavík'])

    city_dict = {'København':f'Copenhagen (N {cop_N})','Reykjavík':f'Reykjavik (N {rey_N})',
                'Helsinki':f'Helsinki (N {hel_N})','Oslo':f'Oslo (N {osl_N})','Stockholm':f'Stockholm (N {sto_N})'}
    df_melted['fua_name'] = df_melted['fua_name'].map(city_dict)

    df_melted['Share_of_Total'] = (df_melted['Carbon_Footprint'] / df_melted[total_col]) * 100

    avg_shares = df_melted.groupby([group_col, 'CF_Domain'])['Share_of_Total'].mean().reset_index()
    short_labels = {'cf_Goods and services footprint':'Gs',
                    'cf_Vehicle footprint':'Ve',
                    'cf_Leisure travel footprint':'Le'}
    avg_shares['CF_Domain'] = avg_shares['CF_Domain'].map(short_labels)

    grouped_text = avg_shares.groupby(group_col).apply(
        lambda x: "<br>".join([f"{row['CF_Domain']}: {row['Share_of_Total']:.1f}%" 
                                for _, row in x.iterrows()])
    ).reset_index(name='annotation_text')
    
    legend_labels_fix = {'cf_Goods and services footprint':'Goods and services (Gs)',
                    'cf_Vehicle footprint':'Vehicle (Ve)',
                    'cf_Leisure travel footprint':'Leisure travel (Le)'}
    df_melted['CF_Domain'] = df_melted['CF_Domain'].map(legend_labels_fix)
    fig = px.box(
        df_melted,
        x=group_col,
        y='Carbon_Footprint',
        color='CF_Domain',
        title='Carbon Footprint by Country and Domain',
        labels={'Carbon_Footprint': 'Carbon Footprint', 'fua_name': 'City', 'CF_Domain':'Footprint domain'},
        boxmode='group',
        color_discrete_sequence=px.colors.qualitative.Pastel,
        hover_data={'Share_of_Total': ':.1f%'}
    )
    fig.update_layout(
        margin={"r": 0, "t": 30, "l": 0, "b": 0}, height=700,
        legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.01)
    )

    if shares:
        for _, row in grouped_text.iterrows():
            fig.add_annotation(
                x=row[group_col],
                y=-0.05 * df_melted['Carbon_Footprint'].max(),  # Position just below the axis
                text=row['annotation_text'],  # Multi-line text per group
                showarrow=False,
                font=dict(size=10, color='black'),
                xanchor='center',
                yanchor='top',
                align='center'
            )

        fig.update_layout(
            margin=dict(b=100),  # Add space at the bottom for multi-line percentages
            xaxis_title_standoff=40  # Push x-axis title down for cleaner layout
        )

    return fig


def plot_lu_sankey(data, lu_to_reclass, reclass_to_final):

    df = data.copy()

    # 1. Extract LU columns and remove "lu_" prefix
    lu_cols = [col for col in df.columns if col.startswith("lu_")]
    lu_sums = df[lu_cols].sum()
    lu_sums.index = [col.replace("lu_", "") for col in lu_cols]

    # 2. Build first stage: original → reclass
    stage1 = defaultdict(float)
    for orig_class, val in lu_sums.items():
        if orig_class in lu_to_reclass:
            reclass = lu_to_reclass[orig_class]
            stage1[(orig_class, reclass)] += val

    # 3. Build second stage: reclass → final
    reclass_totals = defaultdict(float)
    for (_, reclass), val in stage1.items():
        reclass_totals[reclass] += val

    stage2 = defaultdict(float)
    for reclass, val in reclass_totals.items():
        if reclass in reclass_to_final:
            final = reclass_to_final[reclass]
            stage2[(reclass, final)] += val

    # 4. Gather all unique labels
    labels = list(set(
        [k[0] for k in stage1.keys()] +
        [k[1] for k in stage1.keys()] +
        [k[1] for k in stage2.keys()]
    ))
    label_idx = {label: i for i, label in enumerate(labels)}

    # 5. Create source-target-value lists
    def build_links(stage, label_idx):
        source = [label_idx[src] for (src, tgt) in stage.keys()]
        target = [label_idx[tgt] for (src, tgt) in stage.keys()]
        value = list(stage.values())
        return source, target, value

    s1_source, s1_target, s1_value = build_links(stage1, label_idx)
    s2_source, s2_target, s2_value = build_links(stage2, label_idx)

    # Combine both stages
    source = s1_source + s2_source
    target = s1_target + s2_target
    value = s1_value + s2_value

    # 6. Plot with Plotly
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels
        ),
        link=dict(
            source=source,
            target=target,
            value=value
        )
    )])

    fig.update_layout(title_text="Land-Use Reclassification Sankey", font_size=10)
    return fig


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
        #datac.drop(columns=remove_cols, inplace=True)
        combine_cols1 = ['lu_parks','lu_forest','lu_open']
        #combine_cols2 = ['lu_parks','lu_forest']
        new_col_name1 = "_".join(combine_cols1)
        #new_col_name2 = "_".join(combine_cols2)
        datac[new_col_name1] = datac[combine_cols1].sum(axis=1)
        #datac[new_col_name2] = datac[combine_cols2].sum(axis=1)
        datac.drop(columns=combine_cols1, inplace=True)
        #datac.drop(columns=combine_cols2, inplace=True)
        lu_cols_in_use = [col for col in datac.columns if col.startswith('lu')]
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

        #if st.toggle("Show land-use classification in preset"):
            # data_orig = data_orig[data_orig['fua_name'].isin(target_cities)]
            # t1,t5,t9 = st.tabs(['1km','5km','9km'])
            # with t1:
            #     data1 = data_orig[data_orig['R'] == "R1"]
            #     d1 = data1.drop(columns=["h3_10"]).sum().to_frame().T
            #     sankeyfig1 = plot_lu_sankey(data=d1, lu_to_reclass=lu_cols_map, reclass_to_final=lu_premap)
            #     st.plotly_chart(sankeyfig1, use_container_width=True,key='san1')
            # with t5:
            #     data5 = data_orig[data_orig['R'] == "R5"]
            #     d5 = data5.drop(columns=["h3_10"]).sum().to_frame().T
            #     sankeyfig5 = plot_lu_sankey(data=d5, lu_to_reclass=lu_cols_map, reclass_to_final=lu_premap)
            #     st.plotly_chart(sankeyfig5, use_container_width=True,key='san5')
            # with t9:
            #     data9 = data_orig[data_orig['R'] == "R9"]
            #     d9 = data9.drop(columns=["h3_10"]).sum().to_frame().T
            #     sankeyfig9 = plot_lu_sankey(data=d9, lu_to_reclass=lu_cols_map, reclass_to_final=lu_premap)
            #     st.plotly_chart(sankeyfig9, use_container_width=True,key='san9')



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
            datac.drop(columns=remove_cols, inplace=True)
            lu_cols_in_use = [col for col in datac.columns if col.startswith('lu')]
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

        #distribution settings
        power_lucf = st.toggle("Power transform distribution",value=True,
                            help="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.yeojohnson.html")
        norm = st.radio("Normalization",['none','normalize','standardize'],index=2,horizontal=True)
        st.caption('Method for pearson: https://www.statsmodels.org/stable/example_formulas.html')
        st.caption('Method for partial spearman: https://pingouin-stats.org/build/html/generated/pingouin.partial_corr.html#pingouin.partial_corr')
        st.caption("Distributions power transformed (and standardized) for all non-categorical variables using https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.yeojohnson.html")

    lu_cols = [col for col in datac.columns if col.startswith('lu')]

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
                df_r1 = datac[datac['R'] == f"R1"]
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
                df_r5 = datac[datac['R'] == f"R5"]
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
                df_r9 = datac[datac['R'] == f"R9"]
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
                    histo = go.Histogram(x=datac[col],opacity=0.75,name=col,nbinsx=20)
                    histo_traces_cf.append(histo)
                layout_histo = go.Layout(title='CF domain histograms',barmode='overlay')
                histo_fig_lu = go.Figure(data=histo_traces_cf, layout=layout_histo)
                st.plotly_chart(histo_fig_lu, use_container_width=True)



if len(target_cols) == 0 or len(target_cities) == 0:
    st.stop()

## ------- reg tables ---------
with st.expander(f'Regression tables', expanded=False):
    
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

            return reg_results, base_results, ext_results

    def gen_reg_table_multitarget(datac,target_cf_cols,
                       base_cols,
                       cat_cols,
                       ext_cols):
        reg_dfs = []
        for r in datac['R'].unique().tolist():
            df_r = datac[datac['R'] == r]
            for target  in target_cf_cols:
                reg_df, base_results, ext_results = gen_reg_table(df_in=df_r.fillna(0),
                                        cf_col=target,
                                        base_cols=base_cols,
                                        cat_cols=cat_cols,
                                        ext_cols=ext_cols
                                        )
                reg_df['Radius'] = r
                reg_df['Domain'] = target
                reg_dfs.append(reg_df)
            
        reg_all = pd.concat(reg_dfs)

        return reg_all
    

    if len(target_cols) >= 1:

        yksi,viisi,yhdeksan = st.tabs(['1km','5km','9km'])
        with yksi:
            df_r1 = datac[datac['R'] == f"R1"]

            r1_lu_cols = [col for col in df_r1.columns if col.startswith('lu')]
            
            reg_df1 = gen_reg_table_multitarget(datac=df_r1.fillna(0),
                                    target_cf_cols=target_cols,
                                    base_cols=base_cols_orig + control_cols,
                                    cat_cols=cat_cols,
                                    ext_cols=r1_lu_cols
                                    )
            if use_predefined:
                reg_df1.rename(index=lu_premap, inplace=True)
            st.data_editor(reg_df1.drop(columns='Radius'),use_container_width=True, height=500,key=yksi)

            # if len(target_cols) == 1:
            #     if st.toggle('Show OLS summary', key=1):
            #         r1,r2 = st.columns(2)
            #         with r1:
            #             st.markdown('**Base model**')
            #             st.text(base_results.summary())
            #         with r2:
            #             st.markdown('**Ext. model**')
            #             st.text(ext_results.summary())
    
    with viisi:
        df_r5 = datac[datac['R'] == f"R5"]
        r5_lu_cols = [col for col in df_r5.columns if col.startswith('lu')]
        reg_df5 = gen_reg_table_multitarget(datac=df_r5.fillna(0),
                                    target_cf_cols=target_cols,
                                    base_cols=base_cols_orig + control_cols,
                                    cat_cols=cat_cols,
                                    ext_cols=r5_lu_cols
                                    )
        if use_predefined:
            reg_df5.rename(index=lu_premap, inplace=True)
        st.data_editor(reg_df5.drop(columns='Radius'),use_container_width=True, height=500,key=viisi)

    with yhdeksan:
        df_r9 = datac[datac['R'] == f"R9"]
        r9_lu_cols = [col for col in df_r9.columns if col.startswith('lu')]
        reg_df9 = gen_reg_table_multitarget(datac=df_r9.fillna(0),
                                    target_cf_cols=target_cols,
                                    base_cols=base_cols_orig + control_cols,
                                    cat_cols=cat_cols,
                                    ext_cols=r9_lu_cols
                                    )
        if use_predefined:
            reg_df9.rename(index=lu_premap, inplace=True)
        st.data_editor(reg_df9.drop(columns='Radius'),use_container_width=True, height=500,key=yhdeksan)


with st.expander(f'Regression plots', expanded=True):
    
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
                f'N={len(datac)}',f'Control_cols:{control_cols}',f'Lu_cols:{lu_cols}',f'norm:{norm}']
    else:
        setup = [f'Cluster_level:{cluster}',
                f'N={len(datac)}',f'Control_cols:{control_cols}',f'Lu_cols:{lu_cols}',f'norm:{norm}']

    st.download_button(
                        label=f"Download study as CSV",
                        data=reg_df_all.to_csv().encode("utf-8"),
                        file_name=f"cfua_data_SETUP:{setup}.csv",
                        mime="text/csv",
                        )

