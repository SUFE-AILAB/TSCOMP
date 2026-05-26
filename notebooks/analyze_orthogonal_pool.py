
import os
import glob
import re
import numpy as np
import pandas as pd
import logging
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.regression.mixed_linear_model import MixedLM
import statsmodels.api as sm
from scipy import stats
import matplotlib.font_manager as fm
from dimension_couple_analyzer import DimensionCoupleAnalyzer
import warnings
warnings.filterwarnings("ignore")

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("orthogonal_analysis_v3.log", mode='w')
    ]
)
logger = logging.getLogger(__name__)

# Font Setup
cn_fonts = ['SimHei', 'Arial Unicode MS', 'WenQuanYi Micro Hei', 'Microsoft YaHei', 'Heiti TC']
available_fonts = set([f.name for f in fm.fontManager.ttflist])
USE_CN_PLOTS = False
for font in cn_fonts:
    if font in available_fonts:
        plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        USE_CN_PLOTS = True
        break

class ResultLoader:
    """
    Handles robust loading of experiment results from filesystem.
    """
    def __init__(self, result_dir):
        self.result_dir = result_dir

    def _parse_model_string(self, model_str):
        """Extracts components, pred_len, seq_len, loss_func from model folder string."""
        parts = model_str.split('_')
        
        start_idx = -1
        for i, p in enumerate(parts):
            if 'TSGym' in p:
                start_idx = i
                break
        
        if start_idx == -1: return None
        # Filter: only keep TSGym120* IDs (true orthogonal pool)
        tsgym_id = parts[start_idx]
        if not tsgym_id.startswith('TSGym120'):
            return None
        
        pl_match = re.search(r'_pl(\d+)_', model_str)
        sl_match = re.search(r'_sl(\d+)_', model_str)
        lf_match = re.search(r'_lf([A-Za-z]+)_', model_str)
        
        pl_val = int(pl_match.group(1)) if pl_match else None
        sl_val = int(sl_match.group(1)) if sl_match else None
        lf_val = lf_match.group(1) if lf_match else None
        
        try:
            data = {
                'gym_x_mark': parts[start_idx + 1],
                'series_sampling': parts[start_idx + 2],
                'gym_series_norm': parts[start_idx + 3],
                'gym_series_decomp': parts[start_idx + 4],
                'channel_independent': parts[start_idx + 5],
                'gym_input_embed': parts[start_idx + 6],
                'network_architecture': parts[start_idx + 7],
                'attn': parts[start_idx + 8],
                'feature_attn': parts[start_idx + 9],
                'gym_frozen': parts[start_idx + 11],
                'gym_rag': parts[start_idx + 12],
            }
            if pl_val is not None: data['pred_len'] = pl_val
            if sl_val is not None: data['seq_len'] = sl_val
            if lf_val is not None: data['loss_func'] = lf_val
            return data
        except IndexError:
            return None

    def load(self, dataset_filter=None):
        logger.info(f"Scanning directory: {self.result_dir} ...")
        
        # Define exclusion path (absolute)
        exclude_dir = os.path.abspath(os.path.join(self.result_dir, 'results'))
        
        metrics_files = glob.glob(os.path.join(self.result_dir, '**', 'metrics.npy'), recursive=True)
        logger.info(f"Found {len(metrics_files)} metric files.")
        
        parsed_data = []
        for f in metrics_files:
            # Exclude specific 'results' folder
            # Ensure trailing slash to match only directory itself, not 'resultsGym...'
            if os.path.abspath(f).startswith(exclude_dir + os.sep):
                continue
                
            try:
                model_dir = os.path.dirname(f)
                
                # Check for full_config_name.txt for long model names
                config_file = os.path.join(model_dir, 'full_config_name.txt')
                if os.path.exists(config_file):
                    try:
                        with open(config_file, 'r') as cf:
                            model_name = cf.read().strip()
                    except Exception:
                        model_name = os.path.basename(model_dir)
                else:
                    model_name = os.path.basename(model_dir)

                dataset_name = os.path.basename(os.path.dirname(model_dir)).lower()
                
                if dataset_filter and dataset_name not in dataset_filter: continue
                
                metrics = np.load(f)
                # Load all metrics: MAE(0), MSE(1), RMSE(2), MAPE(3), MSPE(4)
                mae = metrics[0]
                mse = metrics[1]
                rmse = metrics[2]
                mape = metrics[3] if len(metrics) > 3 else np.nan
                mspe = metrics[4] if len(metrics) > 4 else np.nan

                features = self._parse_model_string(model_name)
                if features:
                    # Filter invalid prediction lengths (e.g. bugged '9')
                    valid_pls = {24, 36, 48, 60, 96, 192, 336, 720}
                    if features.get('pred_len') not in valid_pls:
                        continue

                    # Store all metrics
                    features['Score'] = float(mse)  # Default to MSE for backward compatibility
                    features['Score_MAE'] = float(mae)
                    features['Score_MSE'] = float(mse)
                    features['Score_RMSE'] = float(rmse)
                    features['Score_MAPE'] = float(mape) if not np.isnan(mape) else np.nan
                    features['Score_MSPE'] = float(mspe) if not np.isnan(mspe) else np.nan
                    features['Model_ID'] = model_name
                    features['dataset_name'] = dataset_name
                    features['Path'] = f
                    parsed_data.append(features)
            except Exception: continue
                
        if not parsed_data: return pd.DataFrame()
        logger.info(f"Successfully loaded {len(parsed_data)} experiment records.")
        return pd.DataFrame(parsed_data)


class DataPreprocessor:
    """Handles data cleaning and standardization."""
    def __init__(self, min_samples_per_group=5):
        self.min_samples = min_samples_per_group

    def standardize_by_group(self, df, group_cols=['dataset_name', 'pred_len'], score_col='Score_MSE'):
        """
        Standardize scores by group using Z-Score transformation.

        与旧代码 analyze_orthogonal_pool_v3_old.py 完全一致的逻辑：
        1. 不过滤NaN行，直接进行分组
        2. 按min_samples过滤有效组
        3. Z-Score标准化

        Args:
            df: DataFrame with metric columns
            group_cols: columns to group by for standardization
            score_col: which metric column to use as 'Score' (default: 'Score_MSE')
                       Options: 'Score_MAE', 'Score_MSE', 'Score_RMSE', 'Score_MAPE', 'Score_MSPE', 'Score_MASE'

        Returns:
            DataFrame with standardized 'Score' column
        """
        if df.empty or not all(c in df.columns for c in group_cols): return df

        # Check if score_col exists
        if score_col not in df.columns:
            logger.warning(f"Score column '{score_col}' not found, using 'Score' instead")
            score_col = 'Score'

        # 与旧代码一致：直接分组过滤，不过滤NaN行
        group_counts = df.groupby(group_cols).size()
        valid_groups = group_counts[group_counts >= self.min_samples].index

        df_filtered = df.set_index(group_cols).loc[valid_groups].reset_index()
        logger.info(f"Performing Z-Score Standardization on {score_col} (N={len(df_filtered)})...")

        # 设置 Score 和 Raw_Score
        df_filtered['Raw_Score'] = df_filtered[score_col]
        def z_score(x): return (x - x.mean()) / x.std() if len(x) > 1 and x.std() > 0 else x - x.mean()

        df_filtered['Score'] = df_filtered.groupby(group_cols)['Raw_Score'].transform(z_score)
        df_filtered['Score'].fillna(0, inplace=True)

        # Convert all categorical columns to strings to avoid type mismatch in OLS formula
        # This ensures consistency when specifying custom reference levels
        # Note: Must convert BEFORE dropping any columns needed for analysis
        excluded = ['Score', 'Raw_Score', 'Model_ID', 'Path', 'Model_Name', 'dataset_name',
                    'Score_MAE', 'Score_MSE', 'Score_RMSE', 'Score_MAPE', 'Score_MSPE', 'Score_MASE']
        categorical_cols = [c for c in df_filtered.columns if c not in excluded and c not in group_cols]
        for c in categorical_cols:
            df_filtered[c] = df_filtered[c].astype(str)
        logger.info(f"Converted {len(categorical_cols)} categorical columns to strings")

        return df_filtered

class Visualizer:
    """Handles all plotting operations."""
    def __init__(self, output_dir='ortho_experiment_report_v3/figures'):
        self.output_dir = output_dir
        for sub in ['distributions', 'radars']:
            os.makedirs(os.path.join(self.output_dir, sub), exist_ok=True)
            
        self.label_mappings = {
            'gym_series_norm': {'None': 'w/o Norm', 'nan': 'w/o Norm', 'null': 'w/o Norm'},
            'gym_series_decomp': {'None': 'w/o Decomp', 'nan': 'w/o Decomp', 'null': 'w/o Decomp'},
            'feature_attn': {'None': 'w/o Feat. Attn.', 'nan': 'w/o Feat. Attn.', 'null': 'w/o Feat. Attn.'},
        }

    def plot_ridgeline(self, df, col, suffix="Global", show_counts=False):
        if df[col].nunique() <= 1 or df[col].nunique() > 50: return
        
        df_viz = df.copy()
        
        # Apply label mappings if available
        if col in self.label_mappings:
            df_viz[col] = df_viz[col].astype(str).replace(self.label_mappings[col])
            
        df_viz['Score'] = df_viz['Score'].clip(upper=4.0, lower=-4.0)
        row_order = df_viz.groupby(col)['Score'].median().sort_values().index.tolist()
        
        counts = df_viz[col].value_counts()
        if show_counts:
            label_map = {val: f"{val}\n(N={count})" for val, count in counts.items()}
        else:
            label_map = {val: f"{val}" for val, count in counts.items()}
        df_viz['Label'] = df_viz[col].map(label_map)
        ordered_labels = [label_map[x] for x in row_order if x in label_map]
        
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
        try:
            g = sns.FacetGrid(df_viz, row='Label', hue='Label', aspect=9, height=0.8,
                              row_order=ordered_labels, hue_order=ordered_labels, palette='viridis')
            g.map(sns.kdeplot, "Score", clip_on=False, shade=True, alpha=0.8, lw=1.5, bw_adjust=0.5)
            g.map(sns.kdeplot, "Score", clip_on=False, color="w", lw=2, bw_adjust=0.5)
            g.map(plt.axhline, y=0, lw=2, clip_on=False)
            
            def label(x, color, label):
                ax = plt.gca()
                ax.text(1.02, 0.2, label, fontweight="bold", color=color, 
                        ha="left", va="center", transform=ax.transAxes, fontsize=16)
            g.map(label, "Score")
            
            g.fig.subplots_adjust(hspace=-0.4, right=0.8)
            g.set_titles("")
            g.set(yticks=[])
            g.set_axis_labels("Score (Z-Score)", "")
            g.despine(bottom=True, left=True)
            
            plt.style.use('ggplot')
            if USE_CN_PLOTS: plt.rcParams['font.sans-serif'] = cn_fonts + ['sans-serif']
            
            safe_col = re.sub(r'[^\w\-]', '_', col)
            g.savefig(os.path.join(self.output_dir, 'distributions', f'ridgeline_{safe_col}_{suffix}.pdf'))
            plt.close()
        except Exception:
            plt.close()
            plt.style.use('ggplot')

    def plot_radar(self, df, col, suffix="Global"):
        if 'dataset_name' not in df.columns or df[col].nunique() <= 1 or df[col].nunique() > 15: return
        
        df_viz = df.copy()
        # Apply label mappings if available
        if col in self.label_mappings:
            df_viz[col] = df_viz[col].astype(str).replace(self.label_mappings[col])

        agg = df_viz.groupby([col, 'dataset_name'])['Score'].median().reset_index()
        pivot = agg.pivot(index=col, columns='dataset_name', values='Score')
        pivot = pivot.dropna(axis=1, thresh=1)
        if pivot.shape[1] < 3: return
        
        # Keep original values for display
        original_values = pivot.copy()
        
        # Normalize scores to [0, 1] for better visualization (position only)
        normalized = pivot.copy()
        for ds in pivot.columns:
            min_s, max_s = pivot[ds].min(), pivot[ds].max()
            if max_s - min_s == 0: 
                normalized[ds] = 0.5  # Mid value for constant
            else: 
                normalized[ds] = 1.0 - (pivot[ds] - min_s) / (max_s - min_s)
        normalized = normalized.fillna(0.0)
        
        # Prepare data
        categories = normalized.columns.tolist()
        num_vars = len(categories)
        
        # Compute angle for each axis (in radians)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        
        # Close the plot by appending the first value
        angles += angles[:1]
        
        # Define marker styles
        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'X', 'P', 'd', '8', 'H']
        
        # Custom color palette - deeper, richer colors
        custom_colors = ['#91C970', '#6B779B', '#F6C952', '#F76560', 
                        '#71C2E2', '#57A686', '#DC9578', '#9F5BB6',
                        '#4A90E2', '#E94B3C', '#50C878', '#FFB347']
        
        # Set up the figure - CRITICAL: use white facecolor
        fig = plt.figure(figsize=(12, 12), facecolor='white')
        ax = plt.subplot(111, polar=True)
        
        # CRITICAL: Set white background, remove gray
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        choices = normalized.index.tolist()
        
        # Use custom colors
        colors = [custom_colors[i % len(custom_colors)] for i in range(len(choices))]
        
        # Plot data
        lines = []
        labels_list = []
        
        for idx, choice in enumerate(choices):
            values = normalized.loc[choice].tolist()
            values += values[:1]  # Close the polygon
            
            # Smart wrapping for legend labels - target "frequency-enhanced-attention" etc.
            label_text = str(choice)
            if len(label_text) > 18:
                if '-' in label_text:
                    # Find hyphen closest to middle
                    hyphens = [i for i, c in enumerate(label_text) if c == '-']
                    if hyphens:
                        mid = len(label_text) // 2
                        best_h = min(hyphens, key=lambda x: abs(x - mid))
                        label_text = label_text[:best_h+1] + '\n' + label_text[best_h+1:]
            
            # Plot with markers - NO FILL
            line = ax.plot(angles, values, 
                          linewidth=2.5, 
                          color=colors[idx],
                          marker=markers[idx % len(markers)],
                          markersize=10,
                          markerfacecolor=colors[idx],
                          markeredgewidth=2,
                          markeredgecolor='white',
                          label=label_text,
                          zorder=10 + idx)[0]
            
            lines.append(line)
            labels_list.append(label_text)
        
        # Configure polar axes
        ax.set_theta_offset(np.pi / 2)  # Start from top
        ax.set_theta_direction(-1)  # Clockwise
        
        # Set the labels for each axis
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=25, weight='bold', color='black')
        
        # Set radial limits and labels
        ax.set_ylim(0, 1.2)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], 
                          fontsize=10, 
                          color='gray')
        
        # Grid styling - lighter and cleaner
        ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.6, color='gray', zorder=1)
        ax.spines['polar'].set_visible(False)
        
        # Add value labels along the radial axes (OUTSIDE the data points)
        # Strategy: For each category axis, annotate all values
        # CRITICAL: Use ORIGINAL values for display, but NORMALIZED values for position
        for cat_idx, (angle, category) in enumerate(zip(angles[:-1], categories)):
            for choice_idx, choice in enumerate(choices):
                norm_value = normalized.loc[choice, category]  # For position
                orig_value = original_values.loc[choice, category]  # For display
                
                # Only show significant values
                if norm_value > 0.05:
                    # Position: slightly beyond the actual value (radially outward)
                    label_radius = norm_value + 0.08
                    
                    # Calculate position
                    x_pos = angle
                    y_pos = label_radius
                    
                    # Adaptive alignment based on angle
                    if angle == 0 or angle == np.pi:
                        ha = 'center'
                    elif 0 < angle < np.pi:
                        ha = 'left'
                    else:
                        ha = 'right'
                    
                    # Add text with ORIGINAL value, larger font, HIGHEST z-order
                    ax.text(x_pos, y_pos, f'{orig_value:.3f}', 
                           fontsize=11,  # Increased from 8
                           ha=ha, 
                           va='center',
                           color=colors[choice_idx],
                           weight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor='white', 
                                   edgecolor=colors[choice_idx],
                                   linewidth=0.8,
                                   alpha=0.95),
                           zorder=1000)  # HIGHEST priority - always on top
        
        # Legend at top with horizontal layout
        ncol = min(len(choices), 5)
        legend = ax.legend(lines, labels_list,
                          loc='upper center',
                          bbox_to_anchor=(0.5, 1.18),
                          ncol=ncol,
                          frameon=False,  # Remove frame
                          fontsize=16,    # Larger font
                          markerscale=1.5,
                          borderpad=0.8)
                          
        # legend.get_frame().set_linewidth(1.5) # No frame needed
        
        # Save with high quality
        safe_col = re.sub(r'[^\w\-]', '_', col)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'radars', f'radar_{safe_col}_{suffix}.pdf'),
                   dpi=300, 
                   bbox_inches='tight',
                   facecolor='white',
                   edgecolor='none')
        plt.close()

    def plot_bar_chart(self, df, x_col, y_col, title, filename):
        with sns.axes_style("white"):
            plt.figure(figsize=(10, 8))
            sns.barplot(data=df, x=x_col, y=y_col, palette='viridis')
            plt.title(title, fontsize=16)
            
            # Enforce larger, bold font for Component labels
            plt.ylabel(y_col, fontsize=16, weight='bold')
            plt.xlabel(x_col, fontsize=14)
            plt.yticks(fontsize=14, weight='bold')
            plt.xticks(fontsize=12)
            
            sns.despine()
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, filename), 
                       dpi=300, 
                       bbox_inches='tight',
                       facecolor='white', 
                       edgecolor='none')
            plt.close()

class OrthogonalAnalyzer:
    """Core Analysis Engine."""
    def __init__(self, df):
        self.df = df
        excluded = ['Model_ID', 'Score', 'Raw_Score', 'HP_Part', 'dataset_name', 'Path', 'Model_Name', 'pred_len',
                    'Score_MAE', 'Score_MSE', 'Score_RMSE', 'Score_MAPE', 'Score_MSPE', 'Score_MASE']
        self.component_cols = [c for c in self.df.columns if c not in excluded]
        
    def _get_reference_level(self, variable_name, levels):
        """
        Determine the appropriate reference level for a variable.
        Uses semantic meaning rather than alphabetical order.
        
        Args:
            variable_name: Name of the variable/dimension
            levels: List of unique levels for this variable
            
        Returns:
            The level to use as reference (baseline) - returns the ACTUAL value from levels
        """
        # Convert to list for easier handling
        levels_list = list(levels)
        levels_str_set = set(str(lvl) for lvl in levels_list)
        
        # Rule 1: Binary True/False -> use False as baseline
        if levels_str_set == {'True', 'False'}:
            # Find and return the actual 'False' value from levels
            for lvl in levels_list:
                if str(lvl) == 'False':
                    return lvl
        
        # Rule 2: Variables with 'None', 'nan', or 'null' -> use as baseline (no treatment)
        # Check for various representations of "no value"
        for null_variant in ['None', 'nan', 'null', 'NaN', 'NULL']:
            for lvl in levels_list:
                if str(lvl) == null_variant:
                    logger.info(f"{variable_name}: Using '{null_variant}' as reference (no treatment baseline)")
                    return lvl
        
        # Rule 3: Specific variable mappings
        reference_preferences = {
            # Series Preprocessing
            'gym_series_norm': ['None', 'nan', 'null'],
            'gym_series_decomp': ['None', 'nan', 'null'],
            'channel_independent': ['False'],
            
            # Series Encoding
            'gym_input_embed': ['series-encoding'],
            'series_sampling': ['False'],
            
            # Network Architecture
            'attn': ['DNN', 'GRU', 'Self-Attention', 'self-attention'],
            'feature_attn': ['None', 'nan', 'null'],
            
            # Network Optimization
            'loss_func': ['MSE'],
            'seq_len': ['24', '48', '96', '192', '336'],  # Prefer smaller values
            
            # Other
            'gym_x_mark': ['False'],
            'gym_rag': ['False'],
            'gym_frozen': ['False'],
            'PL_Group': ['96 (24)', '192 (36)', '336 (48)', '720 (60)'],
            'network_architecture': ['MLP', 'RNN', 'Transformer', 'LLM', 'TSFM'],
        }
        
        # Check if we have preferences for this variable
        if variable_name in reference_preferences:
            for preferred in reference_preferences[variable_name]:
                # Find exact match in actual levels
                for lvl in levels_list:
                    if str(lvl) == preferred:
                        logger.info(f"{variable_name}: Using '{preferred}' as reference (semantic baseline)")
                        return lvl
        
        # Fallback: use alphabetically first (original behavior)
        sorted_levels = sorted(levels_list, key=str)
        logger.info(f"{variable_name}: Using '{sorted_levels[0]}' as reference (alphabetical, no custom rule)")
        return sorted_levels[0]
    
    def run_ols(self, suffix="Global"):
        logger.info(f"\n=== OLS Analysis [{suffix}] ===")
        if not self.component_cols: return None

        # Work with a copy to avoid modifying self.component_cols
        working_cols = self.component_cols.copy()

        # --- Stability Fix 1: Remove constant columns (only 1 unique value) ---
        cols_to_drop = []
        for col in working_cols:
            n_unique = self.df[col].nunique()
            if n_unique <= 1:
                logger.info(f"Dropping constant column '{col}' (only {n_unique} unique value)")
                cols_to_drop.append(col)

        for c in cols_to_drop:
            if c in working_cols:
                working_cols.remove(c)

        if not working_cols:
            logger.warning("No varying columns left after removing constants")
            return None

        # --- Stability Fix 2: Remove Correlated Columns to prevent Singularity ---
        # If network_architecture is present, drop columns that are fixed per architecture (redundant)
        if 'network_architecture' in working_cols:
            for col in working_cols:
                if col == 'network_architecture': continue
                # Check if col is effectively determined by network_architecture
                # (i.e., for every architecture, there is only 1 unique value of this col)
                try:
                    is_fixed_per_arch = self.df.groupby('network_architecture')[col].nunique().eq(1).all()
                    if is_fixed_per_arch and self.df[col].nunique() > 1:
                        logger.info(f"Dropping redundant column '{col}' (perfectly correlated with network_architecture)")
                        cols_to_drop.append(col)
                except Exception:
                    continue

            for c in cols_to_drop:
                if c in working_cols:
                    working_cols.remove(c)
        # ---------------------------------------------------------------------

        # Build formula with explicit reference levels
        terms = []
        for c in working_cols:
            levels = self.df[c].dropna().unique()
            ref_level = self._get_reference_level(c, levels)
            # Use Treatment coding with explicit reference
            terms.append(f"C(Q('{c}'), Treatment(reference='{ref_level}'))")
        
        formula = "Score ~ " + " + ".join(terms)
        logger.info(f"Formula with custom references: {formula[:200]}...")
        
        try:
            model = ols(formula, data=self.df).fit()
            aov = sm.stats.anova_lm(model, typ=3)
            
            # Clean index names for better readability
            clean_index = []
            for idx in aov.index:
                match = re.search(r"Q\('([^']+)'\)", str(idx))
                if match:
                    clean_index.append(match.group(1))
                else:
                    clean_index.append(idx)
            aov.index = clean_index
            # Add Significance Column
            if 'PR(>F)' in aov.columns:
                aov['Significant'] = aov['PR(>F)'] < 0.05
            
            logger.info("--- ANOVA (Type III) ---")
            logger.info(aov)
            
            # 2. Comprehensive Coefficients Construction
            coef_data = []
            params = model.params
            std_err = model.bse
            p_values = model.pvalues
            
            # Intercept
            coef_data.append({
                'Variable': 'Intercept', 'Level': 'Base',
                'Coef': params['Intercept'], 'StdErr': std_err['Intercept'], 'P_Value': p_values['Intercept']
            })

            for col in working_cols:
                levels = sorted(self.df[col].dropna().unique())
                ref_level = self._get_reference_level(col, self.df[col].dropna().unique())

                for lvl in levels:
                    lvl_str = str(lvl)
                    param_name = f"C(Q('{col}'), Treatment(reference='{ref_level}'))[T.{lvl_str}]"

                    if param_name in params:
                        coef_data.append({
                            'Variable': col, 'Level': lvl_str,
                            'Coef': params[param_name],
                            'StdErr': std_err[param_name],
                            'P_Value': p_values[param_name]
                        })
                    else:
                        # This is the reference level
                        coef_data.append({
                            'Variable': col, 'Level': lvl_str,
                            'Coef': 0.0, 'StdErr': 0.0, 'P_Value': 1.0, 'Is_Ref': True
                        })

            coef_df = pd.DataFrame(coef_data)
            # Add Significance Column
            if 'P_Value' in coef_df.columns:
                coef_df['Significant'] = coef_df['P_Value'] < 0.05

            # 3. Calculate Range (Marginal Mean Range)
            range_data = []
            for col in working_cols:
                col_coefs = coef_df[coef_df['Variable'] == col]['Coef']
                if not col_coefs.empty:
                    rng = col_coefs.max() - col_coefs.min()
                    range_data.append({'Component': col, 'Range': rng})
            
            range_df = pd.DataFrame(range_data).sort_values('Range', ascending=False)
            logger.info("\n--- Component Effect Range Ranking ---")
            logger.info(range_df)
            
            return {'anova': aov, 'coef': coef_df, 'range': range_df}

        except Exception as e:
            logger.error(f"OLS Failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def run_glmm(self, suffix="Global"):
        """
        Run GLMM analysis with dataset as random effect.
        Uses the same reference level selection as run_ols.

        Paper claim: "GLMM to isolate the marginal contribution of each component"
        Data: Standardized MSE
        Random effect: dataset_name
        Fixed effects: components + pred_len
        """
        logger.info(f"\n=== GLMM Analysis [{suffix}] ===")
        if not self.component_cols:
            return None

        # Check if dataset_name exists for random effect
        if 'dataset_name' not in self.df.columns:
            logger.warning("dataset_name column not found, cannot run GLMM")
            return None

        # Build formula with explicit reference levels (same as run_ols)
        terms = []
        for c in self.component_cols:
            levels = self.df[c].dropna().unique()
            ref_level = self._get_reference_level(c, levels)
            terms.append(f"C(Q('{c}'), Treatment(reference='{ref_level}'))")

        # Add pred_len as fixed effect
        if 'pred_len' in self.df.columns:
            formula = "Score ~ " + " + ".join(terms) + " + C(pred_len)"
        else:
            formula = "Score ~ " + " + ".join(terms)

        logger.info(f"GLMM Formula: {formula[:200]}...")

        try:
            # Reset index to avoid issues
            df_glmm = self.df.reset_index(drop=True).copy()

            # Fit GLMM
            model = MixedLM.from_formula(
                formula,
                groups="dataset_name",
                data=df_glmm
            )
            result = model.fit(reml=True, disp=False)

            # Extract variance components
            # MixedLM: result.scale = residual variance (sigma2_e)
            # Random effect variance from cov_re (diagonal of random effect covariance)
            sigma2_e = result.scale
            cov_re_diag = np.diag(result.cov_re)
            sigma2_u = cov_re_diag[0] if len(cov_re_diag) > 0 else 0
            ICC = sigma2_u / (sigma2_u + sigma2_e) if (sigma2_u + sigma2_e) > 0 else 0

            logger.info(f"GLMM Log-Likelihood: {result.llf:.2f}")
            logger.info(f"Random effect variance (σ²_u): {sigma2_u:.6f}")
            logger.info(f"Residual variance (σ²_ε): {sigma2_e:.6f}")
            logger.info(f"ICC: {ICC:.4f}")

            # Extract fixed effects (component coefficients)
            fe_params = result.fe_params
            fe_pvalues = result.pvalues
            fe_bse = result.bse

            # Build coefficients DataFrame
            coef_data = []

            # Intercept
            if 'Intercept' in fe_params.index:
                coef_data.append({
                    'Variable': 'Intercept', 'Level': 'Base',
                    'Coef': fe_params['Intercept'],
                    'StdErr': fe_bse['Intercept'],
                    'P_Value': fe_pvalues['Intercept']
                })

            # Component coefficients
            for col in self.component_cols:
                levels = sorted(self.df[col].dropna().unique())
                ref_level = self._get_reference_level(col, self.df[col].dropna().unique())

                for lvl in levels:
                    lvl_str = str(lvl)
                    param_name = f"C(Q('{col}'), Treatment(reference='{ref_level}'))[T.{lvl_str}]"

                    if param_name in fe_params.index:
                        coef_data.append({
                            'Variable': col, 'Level': lvl_str,
                            'Coef': fe_params[param_name],
                            'StdErr': fe_bse[param_name],
                            'P_Value': fe_pvalues[param_name]
                        })
                    else:
                        # Reference level
                        coef_data.append({
                            'Variable': col, 'Level': lvl_str,
                            'Coef': 0.0, 'StdErr': 0.0, 'P_Value': 1.0, 'Is_Ref': True
                        })

            coef_df = pd.DataFrame(coef_data)
            if 'P_Value' in coef_df.columns:
                coef_df['Significant'] = coef_df['P_Value'] < 0.05

            # Calculate range (same as OLS)
            range_data = []
            for col in self.component_cols:
                col_coefs = coef_df[coef_df['Variable'] == col]['Coef']
                if not col_coefs.empty:
                    rng = col_coefs.max() - col_coefs.min()
                    range_data.append({'Component': col, 'Range': rng})

            range_df = pd.DataFrame(range_data).sort_values('Range', ascending=False)

            logger.info("\n--- GLMM Component Effect Range Ranking ---")
            logger.info(range_df)

            return {
                'coef': coef_df,
                'range': range_df,
                'ICC': ICC,
                'sigma2_u': sigma2_u,
                'sigma2_e': sigma2_e,
                'log_likelihood': result.llf,
                'model': result,
            }

        except Exception as e:
            logger.error(f"GLMM Failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _get_pipeline_stage(self, component):
        """Map component to pipeline stage."""
        PIPELINE_MAPPING = {
            'gym_series_norm': 'Series Preprocessing',
            'gym_series_decomp': 'Series Preprocessing',
            'series_sampling': 'Series Preprocessing',
            'channel_independent': 'Series Encoding',
            'gym_input_embed': 'Series Encoding',
            'gym_x_mark': 'Series Encoding',
            'network_architecture': 'Network Architecture',
            'attn': 'Network Architecture',
            'feature_attn': 'Network Architecture',
            'gym_rag': 'Network Architecture',
            'gym_frozen': 'Network Architecture',
            'seq_len': 'Network Optimization',
            'loss_func': 'Network Optimization'
        }
        return PIPELINE_MAPPING.get(component, 'Other')

    def analyze_pipeline_importance(self, imp_df, imp_col='Importance', source=''):
        """
        Aggregates component importances to Pipeline Stage level.
        Generic method works for both OLS (SumSq) and GLMM (Wald Chi-Sq).

        Args:
            imp_df: DataFrame with 'Component' and importance column
            imp_col: column name for importance values
            source: label for logging ('ANOVA SumSq' or 'GLMM Wald')
        """
        if imp_df is None or imp_col not in imp_df.columns: return None

        df_imp = imp_df.copy()

        # Map to Pipeline
        df_imp['Pipeline_Stage'] = df_imp['Component'].map(self._get_pipeline_stage)
        df_imp['Pipeline_Stage'].fillna('Other', inplace=True)

        # Aggregate: 1. Total Importance, 2. Find Top Driver
        stage_imp = df_imp.groupby('Pipeline_Stage')[imp_col].sum().sort_values(ascending=False)

        drivers = []
        for stage in stage_imp.index:
            stage_rows = df_imp[df_imp['Pipeline_Stage'] == stage]
            if not stage_rows.empty:
                top_idx = stage_rows[imp_col].idxmax()
                drivers.append({
                    'Pipeline_Stage': stage,
                    'Total_Importance': stage_imp[stage],
                    'Top_Driver_Component': stage_rows.loc[top_idx, 'Component'],
                    'Driver_Share': stage_rows.loc[top_idx, imp_col]
                })

        stage_df = pd.DataFrame(drivers)

        total = stage_df['Total_Importance'].sum()
        if total > 0:
            stage_df['Share_Pct'] = stage_df['Total_Importance'] / total * 100

        logger.info(f"\n--- Pipeline Stage Importance Analysis ({source}) ---")
        logger.info(stage_df)

        return stage_df

    def _build_pipeline_importance_from_anova(self, anova_df):
        """Extract component importance from OLS ANOVA result, prepare for pipeline aggregation."""
        if anova_df is None or 'sum_sq' not in anova_df.columns: return None

        imp_data = []
        for term, row in anova_df.iterrows():
            if term in ['Intercept', 'Residual']: continue

            match = re.search(r"Q\('([^']+)'\)", str(term))
            comp_name = match.group(1) if match else term

            imp_data.append({
                'Component': comp_name,
                'Importance': row['sum_sq']
            })

        return pd.DataFrame(imp_data)

    def _build_pipeline_importance_from_glmm(self, glmm_result):
        """Calculate Wald Chi-Square importance from GLMM result, prepare for pipeline aggregation."""
        model = glmm_result.get('model')
        if model is None: return None

        fe_params = model.fe_params
        fe_bse = model.bse

        imp_data = []
        for comp in self.component_cols:
            comp_terms = [t for t in fe_params.index if f"Q('{comp}')" in t]
            if comp_terms:
                wald_sum = 0
                valid_terms = 0
                for term in comp_terms:
                    se = fe_bse.get(term, np.nan)
                    if pd.notna(se) and se > 0 and se < 100:
                        wald_sum += (fe_params[term] / se) ** 2
                        valid_terms += 1
                if valid_terms > 0:
                    imp_data.append({
                        'Component': comp,
                        'Importance': wald_sum
                    })

        return pd.DataFrame(imp_data) if imp_data else None


def main():
    parser = argparse.ArgumentParser(description='Statistical Analysis for Pipeline Importance')
    parser.add_argument('--result_dir', type=str, default='results_long_term_forecasting_orthogonal_pool',
                        help='Path to experiment results directory')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Filter by specific datasets')
    parser.add_argument('--output_dir', type=str, default='ortho_experiment_report_v3',
                        help='Output directory for reports and figures')
    parser.add_argument('--method', type=str, default='ols', choices=['ols', 'glmm'],
                        help='Analysis method: ols (default) or glmm')
    args = parser.parse_args()

    loader = ResultLoader(args.result_dir)
    df = loader.load(dataset_filter=args.datasets)
    if df.empty: return

    # Standardize Network Architecture Names
    if 'network_architecture' in df.columns:
        def map_arch(x):
            x_str = str(x).strip()
            if 'TSFM' in x_str: return 'TSFM'
            if 'LLM' in x_str or 'GPT' in x_str.upper(): return 'LLM'
            # Check RNN related (case-insensitive usually good but here data is mixed case?)
            # x_str is usually CamelCase or similar.
            x_upper = x_str.upper()
            if 'RNN' in x_upper or 'GRU' in x_upper or 'LSTM' in x_upper: return 'RNN'
            if 'TRANSFORMER' in x_upper: return 'Transformer'
            if 'MLP' in x_upper or 'DNN' in x_upper: return 'MLP'
            return x_str

        df['network_architecture'] = df['network_architecture'].apply(map_arch)
        # Log unique values to confirm normalization
        if len(df) > 0:
             logger.info(f"Standardized network architectures to: {df['network_architecture'].unique()}")

    # Deduplicate experiments: take mean MSE for duplicate (components, dataset, pred_len)
    logger.info("\n" + "="*40 + "\nDEDUPLICATION\n" + "="*40)
    original_count = len(df)

    def extract_tsgym_id(model_id):
        match = re.search(r'TSGym\d+', model_id)
        return match.group(0) if match else model_id

    df['TSGym_ID'] = df['Model_ID'].apply(extract_tsgym_id)

    # Check for duplicates by (TSGym_ID, dataset_name, pred_len)
    group_cols = ['TSGym_ID', 'dataset_name', 'pred_len']
    dup_counts = df.groupby(group_cols).size()
    n_dups = (dup_counts > 1).sum()
    max_dup = dup_counts.max()

    logger.info(f"Before deduplication: {original_count} records")
    logger.info(f"  Unique (TSGym_ID, dataset, pred_len) groups: {len(dup_counts)}")
    logger.info(f"  Groups with duplicates: {n_dups}")
    logger.info(f"  Max occurrences per group: {max_dup}")

    agg_dict = {col: 'mean' for col in df.columns
                if col.startswith('Score_') and col not in ['Score']}
    if 'Score' in df.columns:
        agg_dict['Score'] = 'mean'
    # Preserve ALL component columns during aggregation (same TSGym_ID = same config)
    component_cols = ['gym_x_mark', 'series_sampling', 'gym_series_norm',
                     'gym_series_decomp', 'channel_independent', 'gym_input_embed',
                     'network_architecture', 'attn', 'feature_attn',
                     'gym_frozen', 'gym_rag', 'seq_len', 'loss_func']
    for col in component_cols:
        if col in df.columns:
            agg_dict[col] = 'first'
    if 'Model_ID' in df.columns:
        agg_dict['Model_ID'] = 'first'

    df = df.groupby(group_cols).agg(agg_dict).reset_index()
    df = df.drop(columns=['TSGym_ID'], errors='ignore')

    logger.info(f"After deduplication: {len(df)} records")
    logger.info(f"  Removed {original_count - len(df)} records total")

    # Standardize after deduplication
    preprocessor = DataPreprocessor(min_samples_per_group=5)
    df_std = preprocessor.standardize_by_group(df)

    # Setup Output Dirs
    FIG_DIR = os.path.join(args.output_dir, 'figures')
    TBL_DIR = os.path.join(args.output_dir, 'tables')
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TBL_DIR, exist_ok=True)

    logger.info("\n" + "="*40 + "\nGLOBAL ANALYSIS (Merged Large Models)\n" + "="*40)

    # Create Global DF with merged backbones
    df_global = df_std.copy()
    if 'network_architecture' in df_global.columns:
        def merge_backbone(arch):
            arch_str = str(arch)
            if arch_str.startswith('LLM-'): return 'LLM'
            if arch_str.startswith('TSFM-'): return 'TSFM'
            return arch_str
        df_global['network_architecture'] = df_global['network_architecture'].apply(merge_backbone)
        logger.info(f"Merged Backbone Architectures: {df_global['network_architecture'].unique()}")

    analyzer = OrthogonalAnalyzer(df_global)

    viz = Visualizer(output_dir=FIG_DIR)

    if args.method == 'ols':
        ols_res = analyzer.run_ols(suffix="Global")
        if ols_res:
            ols_res['anova'].to_csv(os.path.join(TBL_DIR, 'ols_anova_Global.csv'))
            ols_res['coef'].to_csv(os.path.join(TBL_DIR, 'ols_coef_Global.csv'), index=False)
            ols_res['range'].to_csv(os.path.join(TBL_DIR, 'ols_range_Global.csv'), index=False)
            viz.plot_bar_chart(ols_res['range'], 'Range', 'Component', 'Component Effect Range', 'effect_range_Global.pdf')

            # Pipeline Analysis (ANOVA based)
            comp_imp = analyzer._build_pipeline_importance_from_anova(ols_res['anova'])
            pipeline_df = analyzer.analyze_pipeline_importance(comp_imp, imp_col='Importance', source='ANOVA SumSq')
            if pipeline_df is not None:
                pipeline_df.to_csv(os.path.join(TBL_DIR, 'pipeline_importance_Global.csv'), index=False)
                viz.plot_bar_chart(pipeline_df, 'Total_Importance', 'Pipeline_Stage', 'Pipeline Stage Importance (ANOVA SumSq)', 'pipeline_stage_importance_Global.pdf')
    else:
        ols_res = None

    if args.method == 'glmm':
        logger.info("\n" + "="*40 + "\nGLMM ANALYSIS\n" + "="*40)

        glmm_res = analyzer.run_glmm(suffix="Global")
        if glmm_res:
            glmm_res['coef'].to_csv(os.path.join(TBL_DIR, 'glmm_coef_Global.csv'), index=False)
            glmm_res['range'].to_csv(os.path.join(TBL_DIR, 'glmm_range_Global.csv'), index=False)

            # GLMM Pipeline Stage Importance (Wald Chi-Square based)
            comp_imp = analyzer._build_pipeline_importance_from_glmm(glmm_res)
            glmm_pipeline = analyzer.analyze_pipeline_importance(comp_imp, imp_col='Importance', source='GLMM Wald')
            if glmm_pipeline is not None:
                glmm_pipeline.to_csv(os.path.join(TBL_DIR, 'glmm_pipeline_importance_Global.csv'), index=False)
                viz.plot_bar_chart(glmm_pipeline, 'Total_Importance', 'Pipeline_Stage',
                                   'Pipeline Stage Importance (GLMM Wald)', 'glmm_pipeline_stage_importance_Global.pdf')

            # Save GLMM summary
            glmm_summary = pd.DataFrame([{
                'Metric': 'ICC',
                'Value': glmm_res['ICC']
            }, {
                'Metric': 'Random_Effect_Variance',
                'Value': glmm_res['sigma2_u']
            }, {
                'Metric': 'Residual_Variance',
                'Value': glmm_res['sigma2_e']
            }, {
                'Metric': 'Log_Likelihood',
                'Value': glmm_res['log_likelihood']
            }])
            glmm_summary.to_csv(os.path.join(TBL_DIR, 'glmm_summary_Global.csv'), index=False)
    else:
        glmm_res = None

    for col in analyzer.component_cols:
        viz.plot_ridgeline(df_global, col, suffix="Global")
        viz.plot_radar(df_global, col, suffix="Global")
        if col == "attn":
            special_attn_list = ['auto-correlation', 'frequency-enhanced-attention', 'destationary-attention']
            df_special = df_global[df_global[col].isin(special_attn_list)]
            if not df_special.empty:
                viz.plot_radar(df_special, col, suffix="Special_Global")
        
    logger.info("\n" + "="*40 + "\nSTRATIFIED ANALYSIS (By Backbone Family)\n" + "="*40)

    if 'network_architecture' in df_std.columns:
        # Define backbone families for grouped analysis
        backbone_families = {
            'LLM': r'^LLM$',
            'TSFM': r'^TSFM$',
            'RNN': r'^RNN$',
            'Transformer': r'^Transformer$',
            'MLP': r'^MLP$',
        }
        
        for family_name, pattern in backbone_families.items():
            logger.info(f"\n>>> Analyzing Backbone Family: {family_name}")
            mask = df_std['network_architecture'].astype(str).str.match(pattern)
            subset = df_std[mask].copy()
            
            if len(subset) < 20:
                logger.info(f"Skipping {family_name}: insufficient samples (N={len(subset)})")
                continue
            
            # Remove constant columns
            subset = subset.loc[:, (subset != subset.iloc[0]).any()]
            
            safe_name = re.sub(r'[^\w\-]', '_', family_name)
            sub_analyzer = OrthogonalAnalyzer(subset)
            
            # Sub OLS
            sub_ols = sub_analyzer.run_ols(suffix=safe_name)
            if sub_ols:
                sub_ols['anova'].to_csv(os.path.join(TBL_DIR, f'ols_anova_{safe_name}.csv'))
                sub_ols['coef'].to_csv(os.path.join(TBL_DIR, f'ols_coef_{safe_name}.csv'), index=False)
                viz.plot_bar_chart(sub_ols['range'], 'Range', 'Component', f'Range - {family_name}', f'effect_range_{safe_name}.pdf')

                # Pipeline Analysis (ANOVA based)
                sub_comp_imp = sub_analyzer._build_pipeline_importance_from_anova(sub_ols['anova'])
                sub_pipeline_df = sub_analyzer.analyze_pipeline_importance(sub_comp_imp, imp_col='Importance', source='ANOVA SumSq')
                if sub_pipeline_df is not None:
                    sub_pipeline_df.to_csv(os.path.join(TBL_DIR, f'pipeline_importance_{safe_name}.csv'), index=False)
                    viz.plot_bar_chart(sub_pipeline_df, 'Total_Importance', 'Pipeline_Stage', f'Pipeline Stage Importance - {family_name}', f'pipeline_stage_importance_{safe_name}.pdf')

            for col in sub_analyzer.component_cols:
                viz.plot_ridgeline(subset, col, suffix=safe_name)
                viz.plot_radar(subset, col, suffix=safe_name)
                if col == "attn":
                    special_attn_list = ['auto-correlation', 'frequency-enhanced-attention', 'destationary-attention']
                    df_special = subset[subset[col].isin(special_attn_list)]
                    if not df_special.empty:
                        viz.plot_radar(df_special, col, suffix=f"Special_{safe_name}")

    # Dimension Interaction Analysis (disabled for speed)
    # dim_analyzer = DimensionCoupleAnalyzer(df_std, output_dir=args.output_dir)
    # dim_analyzer.analyze_interactions()

if __name__ == "__main__":
    main()

