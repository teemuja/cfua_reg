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
        
        if control_cols is not None:
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

    return df
