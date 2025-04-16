import streamlit as st
import pandas as pd
import h3
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from collections import Counter, defaultdict
from itertools import combinations

#stats
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pingouin as pg


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


def partial_corr_table_v1(df, predictor_cols, target_col, covar_cols=None, delta_raw=10):
    """
    Prepares transformed data and calculates partial correlations with
    estimated % change in original y when predictor increases by x_change units.
    """
    df = df.copy()
    all_predictors = predictor_cols + (covar_cols if covar_cols else [])

    transformations = {col: 'none' for col in all_predictors}
    transformations[target_col] = 'none'

    target = df[target_col]
    target_skew = target.skew()
    has_zeros_target = (target <= 0).any()
    target_is_log = False

    if pg.normality(target)['pval'][0] < 0.05:
        if not has_zeros_target and target_skew > 1:
            df[target_col] = np.log(target)
            transformations[target_col] = 'log'
            target_is_log = True
        elif has_zeros_target:
            df[target_col] = np.sqrt(target + 0.5)
            transformations[target_col] = 'sqrt(+0.5)'

    original_X = {}
    transformed_X = {}
    standardized_X = {}

    for col in all_predictors:
        x = df[col]
        original_X[col] = x.copy()
        skew = x.skew()
        has_zeros = (x <= 0).any()

        if pg.normality(x)['pval'][0] < 0.05:
            if not has_zeros and abs(skew) > 1:
                x = np.log(x)
                transformations[col] = 'log'
            elif has_zeros:
                x = np.sqrt(x + 0.5)
                transformations[col] = 'sqrt(+0.5)'

        transformed_X[col] = x.copy()

        if col in predictor_cols:
            x_mean = x.mean()
            x_std = x.std()
            x = (x - x_mean) / x_std
            standardized_X[col] = (x, x_mean, x_std)

        df[col] = x

    results = []

    for col in predictor_cols:
        pearson = pg.partial_corr(data=df, x=col, y=target_col, covar=covar_cols, method='pearson')
        spearman = pg.partial_corr(data=df, x=col, y=target_col, covar=covar_cols, method='spearman')
        beta = pearson['r'].values[0]

        # change in standardized x
        x_orig = original_X[col]
        x_trans = transformed_X[col]
        x_std, x_mean_trans, x_std_trans = standardized_X[col]

        # impact of delta_raw increase of x in sqrt value
        avg_x = x_orig.mean()
        fx1 = np.sqrt(avg_x + 0.5)
        fx2 = np.sqrt(avg_x + delta_raw + 0.5)
        delta_z = (fx2 - fx1) / x_std_trans

        # ..purge true impact as percent
        if target_is_log:
            delta_log_y = beta * delta_z
            approx_pct_change = (np.exp(delta_log_y) - 1) * 100
        else:
            approx_pct_change = None

        results.append({
            'Variable': col,
            'N': pearson['n'].values[0],
            'Transformation': transformations[col],
            'Pearson_r': round(beta, 3),
            'Pearson_p': round(pearson['p-val'].values[0], 4),
            'Pearson_CI95%': pearson['CI95%'].values[0],
            'Spearman_r': round(spearman['r'].values[0], 3),
            'Spearman_p': round(spearman['p-val'].values[0], 4),
            'Spearman_CI95%': spearman['CI95%'].values[0],
            f'Change%_of_{delta_raw}hex': round(approx_pct_change, 2) if approx_pct_change is not None else None
        })

    return pd.DataFrame(results)


def partial_corr_table_v2(df, predictor_cols, target_col, covar_cols=None, delta_raw=10):
    """
    Prepares transformed data and calculates partial correlations with
    estimated % change in original y when predictor increases by delta_raw units.
    """
    df = df.copy()
    all_predictors = predictor_cols + (covar_cols if covar_cols else [])

    transformations = {col: 'none' for col in all_predictors}
    transformations[target_col] = 'none'

    target = df[target_col]
    target_skew = target.skew()
    has_zeros_target = (target <= 0).any()
    target_is_log = False

    if pg.normality(target)['pval'][0] < 0.05:
        if not has_zeros_target and target_skew > 1:
            df[target_col] = np.log(target)
            transformations[target_col] = 'log'
            target_is_log = True
        elif has_zeros_target:
            df[target_col] = np.sqrt(target + 0.5)
            transformations[target_col] = 'sqrt(+0.5)'

    original_X = {}
    transformed_X = {}
    standardized_X = {}

    for col in all_predictors:
        x = df[col]
        original_X[col] = x.copy()
        skew = x.skew()
        has_zeros = (x <= 0).any()

        if pg.normality(x)['pval'][0] < 0.05:
            if not has_zeros and abs(skew) > 1:
                x = np.log(x)
                transformations[col] = 'log'
            elif has_zeros:
                x = np.sqrt(x + 0.5)
                transformations[col] = 'sqrt(+0.5)'

        transformed_X[col] = x.copy()

        if col in predictor_cols:
            x_mean = x.mean()
            x_std = x.std()
            x = (x - x_mean) / x_std
            standardized_X[col] = (x, x_mean, x_std)

        df[col] = x

    results = []

    for col in predictor_cols:
        pearson = pg.partial_corr(data=df, x=col, y=target_col, covar=covar_cols, method='pearson')
        spearman = pg.partial_corr(data=df, x=col, y=target_col, covar=covar_cols, method='spearman')
        beta = pearson['r'].values[0]

        # Restore transformed x to original scale
        x_std, x_mean_trans, x_std_trans = standardized_X[col]
        x_trans = transformed_X[col]
        x_back = x_trans  # just alias for readability

        transform = transformations[col]

        if transform == 'sqrt(+0.5)':
            before = np.sqrt((x_back * x_std_trans + x_mean_trans) + 0.5)
            after = np.sqrt((x_back * x_std_trans + x_mean_trans + delta_raw) + 0.5)
            delta_z = ((after - before) / x_std_trans).mean()
        elif transform == 'log':
            before = np.log((x_back * x_std_trans + x_mean_trans))
            after = np.log((x_back * x_std_trans + x_mean_trans + delta_raw))
            delta_z = ((after - before) / x_std_trans).mean()
        else:  # no transform
            delta_z = delta_raw / x_std_trans

        if target_is_log:
            delta_log_y = beta * delta_z
            approx_pct_change = (np.exp(delta_log_y) - 1) * 100
        else:
            approx_pct_change = None

        results.append({
            'Variable': col,
            'N': pearson['n'].values[0],
            'Transformation': transformations[col],
            'Pearson_r': round(beta, 3),
            'Pearson_p': round(pearson['p-val'].values[0], 4),
            'Pearson_CI95%': pearson['CI95%'].values[0],
            'Spearman_r': round(spearman['r'].values[0], 3),
            'Spearman_p': round(spearman['p-val'].values[0], 4),
            'Spearman_CI95%': spearman['CI95%'].values[0],
            f'Change%_of_{delta_raw}hex': round(approx_pct_change, 2) if approx_pct_change is not None else None
        })

    return pd.DataFrame(results)


def partial_corr_table_v3(df, predictor_cols, target_col, covar_cols=None):
    
    df = df.copy()

    #log muunnos
    df[target_col] = np.log(df[target_col])
            
    results = []

    for col in predictor_cols:
        #replace 0 with 1 to apply log..
        df.loc[df[col] == 0, col] = 1
        df[col] = np.log(df[col])

        pearson = pg.partial_corr(data=df, x=col, y=target_col, covar=covar_cols, method='pearson')
        spearman = pg.partial_corr(data=df, x=col, y=target_col, covar=covar_cols, method='spearman')
        beta = pearson['r'].values[0]

        results.append({
            'Variable': col,
            'N': pearson['n'].values[0],
            'Pearson_r': round(beta, 3),
            'Pearson_p': round(pearson['p-val'].values[0], 4),
            'Pearson_CI95%': pearson['CI95%'].values[0],
            'Spearman_r': round(spearman['r'].values[0], 3),
            'Spearman_p': round(spearman['p-val'].values[0], 4),
            'Spearman_CI95%': spearman['CI95%'].values[0]
        })

    return pd.DataFrame(results)




def linear_regression_table(df, predictor_cols, target_col, covar_cols=None, delta_raw=10):
    """
    Prepares data for regression analysis and returns results with transformation info.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    predictor_cols (list): Main predictor columns
    target_col (str): Target variable column
    covar_cols (list): Covariate columns (optional)
    
    Returns:
    tuple: (processed_df, results_df_with_transforms)
    """
    
    df = df.copy()
    all_predictors = predictor_cols + (covar_cols if covar_cols else [])
    
    # Store transformations and stats for inverse transform purposes
    transformations = {col: 'none' for col in all_predictors}
    transformations[target_col] = 'none'
    stats = {}

    # --- Transform target ---
    target = df[target_col]
    target_skew = target.skew()
    has_zeros_target = (target <= 0).any()
    
    if pg.normality(target)['pval'][0] < 0.05:
        if not has_zeros_target and target_skew > 1:
            df[target_col] = np.log(target)
            transformations[target_col] = 'log'
        elif has_zeros_target:
            df[target_col] = np.sqrt(target + 0.5)
            transformations[target_col] = 'sqrt(+0.5)'

    # --- Transform predictors ---
    for col in all_predictors:
        pred = df[col]
        skew = pred.skew()
        has_zeros = (pred <= 0).any()

        # Save original stats for later scaling
        transformed = pred.copy()
        if pg.normality(pred)['pval'][0] < 0.05:
            if not has_zeros and abs(skew) > 1:
                transformed = np.log(pred)
                transformations[col] = 'log'
            elif has_zeros:
                transformed = np.sqrt(pred + 0.5)
                transformations[col] = 'sqrt(+0.5)'

        # Standardize
        mean = transformed.mean()
        std = transformed.std()
        stats[col] = {'mean': mean, 'std': std}
        df[col] = (transformed - mean) / std

    # --- Regress ---
    X = df[all_predictors]
    y = df[target_col]
    model = pg.linear_regression(X, y)

    # --- Prepare results ---
    results = model.copy().round(3) #[model['names'] != 'Intercept']
    results.insert(loc=1, column='Transformation', value=results['names'].map(transformations))

    # --- Add Approx_%_change_in_y for N hexes ---
    is_log_target = transformations[target_col] == 'log'
    approx_changes = []

    for _, row in results.iterrows():
        name = row['names']
        coef = row['coef']
        transform = transformations.get(name, 'none')
        stat = stats.get(name, None)

        # If this is a covariate (not land-use), skip approx % change
        if covar_cols and name in covar_cols:
            approx_changes.append(np.nan)
            continue

        if name in stats and stat is not None:
            # Simulate increase of N hexagons 
            if transform == 'sqrt(+0.5)':
                before = np.sqrt((df[name] * stat['std'] + stat['mean']) + 0.5)
                after = np.sqrt((df[name] * stat['std'] + stat['mean'] + delta_raw) + 0.5)
                delta = (after - before).mean() / stat['std']
            elif transform == 'log':
                before = np.log(df[name] * stat['std'] + stat['mean'])
                after = np.log((df[name] * stat['std'] + stat['mean'] + delta_raw))
                delta = (after - before).mean() / stat['std']
            elif transform == 'log1p':
                before = np.log1p(df[name] * stat['std'] + stat['mean'])
                after = np.log1p(df[name] * stat['std'] + stat['mean'] + delta_raw)
                delta = (after - before).mean() / stat['std']
            else:  # no transform
                delta = delta_raw / stat['std']

            delta_log_y = coef * delta
            if is_log_target:
                approx_pct = (np.exp(delta_log_y) - 1) * 100
                approx_changes.append(round(approx_pct, 1))
            else:
                approx_changes.append(np.nan)
        else:
            approx_changes.append(np.nan)

    results[f'Change_of_{delta_raw}hex'] = approx_changes

    # --- Add target info row at top ---
    target_info = pd.DataFrame([{
        col: target_col if col == 'names' else pd.NA for col in results.columns
    }])
    target_info['Transformation'] = transformations[target_col]
    target_info[f'Change_of_{delta_raw}hex'] = np.nan
    results = pd.concat([target_info, results], ignore_index=True)
    
    # --- change ---
    results = results[[*results.columns[:-1].insert(2, results.columns[-1])]]

    return results


def interaction_scan(df, target_col, predictors):
    results = []

    df = df.copy()
    df[f'log_{target_col}'] = np.log1p(df[target_col])
    for col in predictors:
        df[f'log_{col}'] = np.log1p(df[col])

    # build combinations
    interaction_terms = []
    for comb in combinations(predictors, 2):
        new_col = f"{comb[0]}_x_{comb[1]}"
        df[new_col] = df[comb[0]] * df[comb[1]]
        interaction_terms.append(new_col)

    all_predictors = predictors + interaction_terms
    X = df[all_predictors]
    y = df[target_col]

    results = pg.linear_regression(X, y)

    results = results[~results['names'].isin(predictors)]
    return results.round(2)





def gen_reg_table(df_in, cf_col, base_cols, cat_cols, ext_cols,control_cols):
            
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

def gen_reg_table_multitarget(data,target_cf_cols,
                base_cols,
                cat_cols,
                ext_cols,
                control_cols):
    reg_dfs = []
    for r in data['R'].unique().tolist():
        df_r = data[data['R'] == r]
        for target  in target_cf_cols:
            reg_df, base_results, ext_results = gen_reg_table(df_in=df_r.fillna(0),
                                    cf_col=target,
                                    base_cols=base_cols,
                                    cat_cols=cat_cols,
                                    ext_cols=ext_cols,
                                    control_cols=control_cols
                                    )
            reg_df['Radius'] = r
            reg_df['Domain'] = target
            reg_dfs.append(reg_df)
        
    reg_all = pd.concat(reg_dfs)

    return reg_all






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
