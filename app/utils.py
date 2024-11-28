import streamlit as st
import pandas as pd
import numpy as np
import h3
import h3.api.basic_int as h3_api
import ast
import re
import duckdb

#ols regression
import pingouin as pg
import statsmodels.api as sm
import statsmodels.formula.api as smf

#plots
import plotly.graph_objects as go
import plotly.express as px


def prepare_area_chart_data(df):
    # Extract unique land use classes from index
    lu_classes = set('_'.join(idx.split('_')[1:-1]) for idx in df.index if 'lu_' in idx)
    
    # Create new dataframe for visualization
    plot_data = []
    
    for lu_class in lu_classes:
        # Get rows for this land use class
        class_rows = [idx for idx in df.index if f'lu_{lu_class}_r' in idx]
        
        # Extract radius values and ext_r values
        for row in class_rows:
            radius = int(row.split('_r')[-1])  # Extract the radius number
            plot_data.append({
                'Land Use': lu_class,
                'Radius': f'r{radius}',
                'ext_r': df.loc[row, 'ext_r']
            })
    
    # Convert to DataFrame and sort
    plot_df = pd.DataFrame(plot_data)
    plot_df = plot_df.sort_values(['Land Use', 'Radius'])
    
    # Pivot the data for the area chart
    chart_data = plot_df.pivot(index='Radius', columns='Land Use', values='ext_r')
    
    return chart_data


def prepare_data_for_plotly_chart(df,p_limit=0.01): #, exclude_classes=['nan', 'other']
    # Extract unique land use classes from index
    lu_classes = set('_'.join(idx.split('_')[1:-1]) for idx in df.index if 'lu_' in idx)
    
    # Create new dataframe for visualization
    plot_data = []
    
    for lu_class in lu_classes:
        # Get rows for this land use class
        class_rows = [idx for idx in df.index if f'lu_{lu_class}_r' in idx]
        
        # Extract radius values and both ext_r and ext_p values
        for row in class_rows:
            radius = int(row.split('_r')[-1])  # Extract the radius number
            plot_data.append({
                'Land Use': lu_class,
                'Radius': f'r{radius}',
                'ext_r': df.loc[row, 'ext_r'],
                'ext_p': df.loc[row, 'ext_p']
            })
    
    # Convert to DataFrame and sort
    plot_df_all = pd.DataFrame(plot_data)
    plot_df_all = plot_df_all.sort_values(['Land Use', 'Radius'])
    
    # Filter data based on p-value
    plot_df = plot_df_all[plot_df_all['ext_p'] <= p_limit].copy()
    
    #prep plot
    fig = go.Figure()
    
    # Get unique land use classes
    lu_classes = plot_df['Land Use'].unique()
    
    # Add traces for each land use class
    for lu_class in lu_classes:
        
        #if lu_class.lower() in [exc.lower() for exc in exclude_classes]:
        #    continue
        
        class_data = plot_df[plot_df['Land Use'] == lu_class]
        
        custom_color_map = {
                            "diversity": "violet",
                            "high_diversity": "violet",
                            "urban_fabric": "brown",
                            "suburban_fabric": "burlywood",
                            "shopping_retail":"red",
                            "consumer_services":"red",
                            "food_dining":"orange",
                            "leisure_landuse": "olive",
                            "green_areas":"darkgreen",
                            "green_and_recreation":"darkgreen",
                        }
        fig.add_trace(
            go.Scatter(
                x=class_data['Radius'],
                y=class_data['ext_r'],
                name=lu_class,
                mode='lines',
                line=dict(color=custom_color_map.get(lu_class, 'gray')),
                #stackgroup='one',  # This creates the stacked area effect
                hovertemplate=(
                    f"<b>{lu_class}</b><br>" +
                    "Radius: %{x}<br>" +
                    "ext_r: %{y:.3f}<br>" +
                    "ext_p: %{customdata:.3f}<extra></extra>"
                ),
                customdata=class_data['ext_p']
            )
        )
    
    # Update layout
    fig.update_layout(
        title="Land-use feature´s r by radius",
        xaxis_title="Radius",
        yaxis_title="ext_r Value",
        hovermode='x unified',
        showlegend=True
    )
    
    return fig


@st.cache_data()
def ols_reg_table(df_in, cf_col, base_cols, cat_cols, ext_cols, control_cols):
    df = df_in.copy()

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

def partial_corr(df,cf_cols,corr_target,covar='Income level decile'):
    df_all = pd.DataFrame()
    for c in cf_cols:
        row = pg.partial_corr(data=df, x=c, y=corr_target, x_covar=covar, method='pearson')
        domain_index_name = f"{corr_target} vs {c}"
        row['target_domains'] = domain_index_name
        for col in ['r','p-val']:
            row[col] = round(row[col],3)
        df_all = pd.concat([df_all,row])
    df_all.set_index('target_domains', inplace=True)
    return df_all


# ------- services aggregation and sdi classification --------

def classify_service_diversity(df_in, service_col='services', categories_dict=None):
    
    def parse_and_categorize(service_str):
        # Parse service string to dictionary
        if pd.isna(service_str) or service_str == 'nan':
            return {}
        try:
            service_dict = ast.literal_eval(service_str)
        except:
            return {}
        
        # Categorize services using categories_dict
        categorized_counts = {category: 0 for category in categories_dict.keys()}
        for service, count in service_dict.items():
            service = service.lower().replace(' ', '_')
            for category, items in categories_dict.items():
                if service in items:
                    categorized_counts[category] += count
                    break
        
        return categorized_counts
    
    def calculate_sdi(categorized_counts):
        # Remove 'Other' category and get values
        if 'Other' in categorized_counts:
            del categorized_counts['Other']
        
        values = np.array(list(categorized_counts.values()))
        total = values.sum()
        
        # If no services or only one type, return 0
        if total <= 0:
            return 0
        
        # Calculate proportions and remove zeros
        proportions = values[values > 0] / total
        
        # Calculate Shannon index
        return round(-np.sum(proportions * np.log(proportions)), 1)
    
    # Create a copy of the DataFrame
    df = df_in.copy()
    
    # Calculate SDI for each row
    sdis = []
    for service_str in df[service_col]:
        categorized = parse_and_categorize(service_str)
        sdi = calculate_sdi(categorized)
        sdis.append(sdi)
    
    df['service_sdi'] = sdis
    
    def assign_activity_class(series):
        # Calculate quartiles for non-zero SDI values
        non_zero_sdis = series[series > 0]
        if len(non_zero_sdis) > 0:
            q1, q2, q3, q4 = np.percentile(non_zero_sdis, [25, 50, 75, 90])
            
            # Create the classification function
            def classify(x):
                if x == 0:
                    return 0
                elif x <= q1:
                    return 0
                elif x <= q2:
                    return 1
                elif x <= q3:
                    return 2
                elif x <= q4:
                    return 3
                else:
                    return 4
        else:
            # If no non-zero SDI values, all get class 0
            def classify(x):
                return 0
        
        # Apply the classification function to the series
        return series.apply(classify)
    
    # Apply the activity classification
    df['activity_class'] = assign_activity_class(df['service_sdi']).fillna(0)
    
    return df


def count_landuse_types_in_nd(df_in, r, lu_dict):
    df = df_in.copy().set_index('h3_10')
    
    # extract cf records (those hexes which have footprint values)
    cf_points = df[df['Total footprint'] != 0]
    hex_list = cf_points.index.tolist()
    
    #convert r to k (hex rings)
    nd_size_dict = {1:7,3:20,5:33,7:47,9:60}
    k = nd_size_dict[r]
    
    # gen nd df in distance k
    neighborhoods_as_h3ids = {hex_str: h3.grid_disk(hex_str, k) for hex_str in hex_list}

    #init col for "malls"
    df[f"lu_malls"] = 0
    
    # Loop through each hexagon's neighborhood
    for hex_str, hex_ring in neighborhoods_as_h3ids.items():
        # Filter the dataframe for hexagons in the neighborhood
        neighborhood_df = df[df.index.isin(hex_ring)]
        if neighborhood_df.empty:
            continue
        
        # Replace landuse names from dict
        nd_df = neighborhood_df.copy().replace({'landuse_class': lu_dict})

        # Count landuse_class counts in nd and update df accordingly in center hex
        landuse_counts = nd_df['landuse_class'].value_counts()
        for landuse_class, count in landuse_counts.items():
            if f"lu_{landuse_class}" in df.columns:
                df.at[hex_str, f'lu_{landuse_class}'] += count
            else:
                df[f"lu_{landuse_class}"] = 0
                df.at[hex_str, f'lu_{landuse_class}'] = count
        
        # count "malls" (activity class 3 or 4 = high sdi) in the neighborhood
        if 'activity_class' in nd_df.columns:
            mall_count = len(nd_df[nd_df['activity_class'] > 2])
            df.at[hex_str, 'lu_malls'] = mall_count  # Update only for the specific hexagon
            
    lu_columns = [col for col in df.columns if col.startswith('lu_')]
    for col in lu_columns:
        if col not in df.columns:
            df[col] = 0

    df_out = df[df['Total footprint'] != 0]
    return df_out


def cluster_and_aggregate(df, mean_cols, mode_cols, landuse_col, lu_dict, target_resolution):
    """
    Clusters nearby hexagons with carbon footprint values and aggregates data.
    https://uber.github.io/h3-py/api_verbose.html#h3.cell_to_parent
    Parameters:
    - df: DataFrame containing H3 hexagon indexes at resolution 10 and other feature columns.
    - cf_col: Column containing carbon footprint values to identify relevant hexagons.
    - mean_cols: List of columns to aggregate using mean.
    - sum_cols: List of columns to aggregate using sum.
    - target_resolution: Higher H3 resolution level for clustering (e.g., 9).

    Returns:
    - DataFrame with aggregated values at higher-level H3 resolution.
    """
    if target_resolution <= 9:
        # Filter hexagons with carbon footprint data and replace landuse class names
        cf_hexas = df.dropna(subset=["Total footprint"]).replace({'landuse_class': lu_dict})

        # Get higher-level hex ID for clustering
        cf_hexas['parent_hex'] = cf_hexas["h3_10"].apply(lambda x: h3.cell_to_parent(x, target_resolution))

        # Aggregate mean columns
        mean_agg = cf_hexas.groupby('parent_hex')[mean_cols].mean()

        # Aggregate mode (most common) for categorical columns
        mode_agg = cf_hexas.groupby('parent_hex')[mode_cols].agg(lambda x: x.mode()[0] if not x.mode().empty else None)

        # Count occurrences of each land-use class and pivot into separate columns
        landuse_counts = (cf_hexas.groupby(['parent_hex', landuse_col])
                        .size()
                        .unstack(fill_value=0)
                        .add_prefix('lu_'))

        # Combine all aggregated data
        aggregated_df = mean_agg.join(mode_agg).join(landuse_counts).reset_index()
        aggregated_df.rename(columns={'parent_hex': f'h3_0{target_resolution}'}, inplace=True)

        df_out = aggregated_df[aggregated_df['Total footprint'] != 0]
        
    return df_out

@st.cache_data(max_entries=1) #@st.fragment()
def aggregation(df,nd_size_km,cluster_reso,retail_categories,lu_dict):
    
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

    #we first classify service sdi in 'activity_class' column for all hexas in the df
    df_v2 = classify_service_diversity(df, service_col='services', categories_dict=retail_categories)
    
    mean_cols = ['Housing footprint',
    'Vehicle possession footprint',
    'Public transportation footprint',
    'Leisure travel footprint',
    'Goods and services footprint',
    'Pets footprint',
    'Summer house footprint',
    'Total footprint',
    'Age']
    
    mode_cols = [
        'landuse_class',
        'Education level',
        'Household per capita income decile',
        'Number of persons in household',
        'Household type',
        'Urban degree'
    ]
    landuse_col = "landuse_class"
    
    if cluster_reso <= 9: #use reso if set out of orig = 10
        df_for_reg = cluster_and_aggregate(df_v2,
                                                mean_cols,
                                                mode_cols,
                                                landuse_col,
                                                lu_dict,
                                                target_resolution=cluster_reso)
        df_for_reg = df_for_reg.drop(columns=['landuse_class','Urban degree'])
        lu_cols = [
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
    else:
        df_for_reg = count_landuse_types_in_nd(df_in=df_v2, r=nd_size_km, lu_dict=lu_dict)
        df_for_reg = df_for_reg.drop(columns=['fua_name','services','Urban degree'])
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
        
    return df_for_reg, lu_cols