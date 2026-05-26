
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from itertools import combinations
import re
import math
import warnings
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

class DimensionCoupleAnalyzer:
    """
    Analyzes pairwise interactions between dimensions/components.
    Generates interaction tables and bar charts for every pair of dimensions.
    """
    def __init__(self, df_std, output_dir="ortho_experiment_report_v3"):
        self.df = df_std.copy()
        self.output_dir = output_dir
        self.inter_dir = os.path.join(output_dir, 'interactions')
        os.makedirs(self.inter_dir, exist_ok=True)
        
        # Identify Dimension Columns (excluding metadata and scores)
        # Identify Dimension Columns (excluding metadata and scores)
        excluded = ['Model_ID', 'Score', 'Raw_Score', 'HP_Part', 'dataset_name', 'Path', 'Model_Name', 'pred_len']
        
        # Apply Backbone Categorization if present
        if 'network_architecture' in self.df.columns:
            self.df['network_architecture'] = self.df['network_architecture'].apply(self._map_backbone_to_family)
            
        # Define and Apply Label Mappings
        self.label_mappings = {
            'gym_series_norm': {'None': 'w/o Norm', 'nan': 'w/o Norm', 'null': 'w/o Norm'},
            'gym_series_decomp': {'None': 'w/o Decomp', 'nan': 'w/o Decomp', 'null': 'w/o Decomp'},
            'feature_attn': {'None': 'w/o Feat. Attn.', 'nan': 'w/o Feat. Attn.', 'null': 'w/o Feat. Attn.'},
        }
        
        for col, mapping in self.label_mappings.items():
            if col in self.df.columns:
                # Ensure we capture NaNs/None before replacement
                # Convert to string to ensure 'nan' string matches if present, or NaN float becomes 'nan' string
                self.df[col] = self.df[col].astype(str).replace('nan', 'None').replace('null', 'None')
                self.df[col] = self.df[col].replace(mapping)
                
        self.dimensions = [c for c in self.df.columns if c not in excluded]

    def _map_backbone_to_family(self, arch):
        """Maps specific architectures to broader families."""
        arch_str = str(arch)
        
        # LLM Family
        if arch_str.startswith('LLM-'):
            return 'LLM'
        
        # TSFM (foundation model) Family
        if arch_str.startswith('TSFM-'):
            return 'TSFM'
            
        # RNN Family (GRU, LSTM, xLSTM, RNN)
        if arch_str in ['GRU', 'LSTM', 'RNN', 'xlstm']:
            return 'RNN'
            
        # Transformer
        if arch_str == 'Transformer':
            return 'Transformer'
            
        # MLP Family (MLP, DNN)
        if arch_str in ['MLP', 'DNN']:
            return 'MLP'
            
        return arch_str
        
    def analyze_interactions(self):
        """
        Main method to perform pairwise interaction analysis.
        """
        logger.info("\n" + "="*60)
        logger.info("DIMENSION COUPLE INTERACTION ANALYSIS")
        logger.info("="*60)
        
        if len(self.dimensions) < 2:
            logger.warning("Not enough dimensions for interaction analysis.")
            return

        # 1. Generate Interaction Tables (N CSV files)
        # We iterate through all unique pairs
        pairs = list(combinations(self.dimensions, 2))
        logger.info(f"Analyzing {len(pairs)} dimension pairs...")
        
        all_interactions_list = []
        
        for dim1, dim2 in pairs:
            interaction_df = self._generate_interaction_table(dim1, dim2)
            
            if interaction_df is not None:
                # Append to list for the big CSV
                try:
                    df_copy = interaction_df.copy()
                    df_copy['Dimension1'] = dim1
                    df_copy['Components1'] = df_copy[dim1]
                    df_copy['Dimension2'] = dim2
                    df_copy['Components2'] = df_copy[dim2]
                    
                    # Select and order columns
                    df_copy = df_copy[['Dimension1', 'Components1', 'Dimension2', 'Components2', 'Score']]
                    all_interactions_list.append(df_copy)
                except Exception as e:
                    logger.warning(f"Error preparing interaction data for big CSV ({dim1} vs {dim2}): {e}")

        # Save the big CSV
        if all_interactions_list:
            all_interactions_df = pd.concat(all_interactions_list, ignore_index=True)
            big_csv_path = os.path.join(self.output_dir, 'all_dimension_interactions.csv')
            all_interactions_df.to_csv(big_csv_path, index=False)
            logger.info(f"Saved aggregated interaction table to {big_csv_path}")
            
        # 2. Generate One Big Figure with M Subplots
        # We need to account for directionality: (Dim1=X, Dim2=Hue) AND (Dim2=X, Dim1=Hue) are different views
        # The user requested "any two-two dimension interactions should be analyzed".
        # And "Since bar charts have direction... consider both directions".
        # So for a pair (A, B), we should technically plot A as X and B as Hue, AND B as X and A as Hue?
        # Or just one sufficient to see interaction? 
        # Usually one is enough to see the pattern, but the request says "consider direction".
        # Let's generate plots for all non-trivial pairs.
        
        self._generate_interaction_plots(pairs)
        
        # 3. Synergy Analysis (Chemical Reaction Detection)
        self.analyze_synergy_and_report()
        
        # 4. Generate Heatmaps
        self.generate_paper_heatmaps()

    def _generate_interaction_table(self, dim1, dim2):
        """
        Generates a pivot table for the interaction between dim1 and dim2.
        Values are the mean standardized Score.
        """
        # Group by the two dimensions and calculate mean score
        try:
            interaction_df = self.df.groupby([dim1, dim2])['Score'].mean().reset_index()
            pivot_table = interaction_df.pivot(index=dim2, columns=dim1, values='Score')
            
            # Save to CSV
            safe_dim1 = re.sub(r'[^\w\-]', '_', dim1)
            safe_dim2 = re.sub(r'[^\w\-]', '_', dim2)
            filename = f"interaction_table_{safe_dim1}_vs_{safe_dim2}.csv"
            pivot_table.to_csv(os.path.join(self.inter_dir, filename))
            
            return interaction_df
            
        except Exception as e:
            logger.warning(f"Failed to generate table for {dim1} vs {dim2}: {e}")
            return None

    def _generate_interaction_plots(self, pairs):
        """
        Generates a large figure containing subplots for interactions.
        Since there are many pairs, we might need multiple figures or a very large one.
        We will plot (Dim1 vs Dim2) and optionally (Dim2 vs Dim1) if requested, 
        but usually one plot per pair is sufficient if well designed.
        
        However, the user asked for:
        "Large chart with N subplots... X axis is Dim1 components, Y is Standardized MSE... M bars for Dim2 components".
        "Consider direction... so pair (A,B) might need both A-X/B-Hue and B-X/A-Hue charts?"
        
        Let's generate TWO plots for each pair to fully satisfy "consider directions".
        Total plots = 2 * len(pairs).
        Calculated Example: If 10 dimensions, 45 pairs -> 90 plots. That's too big for one figure file.
        We will group them into pages or just save individual relationship plots?
        User asked for "ONE big chart" (一張大图). 
        If N is large, this is impractical. 
        I will try to pack them, but if too many, I'll split into pages of e.g. 16 plots.
        E.g. "interaction_plots_page1.pdf", "interaction_plots_page2.pdf"...
        """
        
        # Prepare list of plot configurations
        plot_configs = []
        for dim1, dim2 in pairs:
            # Direction 1: X=Dim1, Hue=Dim2
            if self._is_valid_interaction(dim1, dim2):
                plot_configs.append({'x': dim1, 'hue': dim2})
            # Direction 2: X=Dim2, Hue=Dim1
            if self._is_valid_interaction(dim2, dim1):
                plot_configs.append({'x': dim2, 'hue': dim1})
                
        if not plot_configs: return

        # Configuration for subplots
        n_plots = len(plot_configs)
        cols = 3  # 3 columns per row
        rows_per_page = 4
        plots_per_page = cols * rows_per_page
        n_pages = math.ceil(n_plots / plots_per_page)
        
        logger.info(f"Generating {n_plots} interaction plots across {n_pages} pages...")
        
        for page in range(n_pages):
            start_idx = page * plots_per_page
            end_idx = min((page + 1) * plots_per_page, n_plots)
            page_configs = plot_configs[start_idx:end_idx]
            
            # Calculate actual rows needed for this page
            n_in_page = len(page_configs)
            current_rows = math.ceil(n_in_page / cols)
            
            fig, axes = plt.subplots(current_rows, cols, figsize=(20, 5 * current_rows), constrained_layout=True)
            axes_flat = axes.flatten() if n_in_page > 1 else [axes]
            
            for i, config in enumerate(page_configs):
                ax = axes_flat[i]
                self._plot_single_interaction(ax, config['x'], config['hue'])
                
            # Hide unused axes
            for j in range(i + 1, len(axes_flat)):
                axes_flat[j].axis('off')
                
            plt.suptitle(f"Pairwise Dimension Interactions (Page {page+1}/{n_pages})", fontsize=16)
            
            # Save
            filename = f"dimension_interactions_page{page+1}.pdf"
            plt.savefig(os.path.join(self.inter_dir, filename))
            plt.close()
            
    def _is_valid_interaction(self, dim1, dim2):
        """Check if interaction is worth plotting (e.g. not too many levels)."""
        n1 = self.df[dim1].nunique()
        n2 = self.df[dim2].nunique()
        # limit complexity: X axis dim shouldn't have > 20 levels, Hue dim shouldn't have > 10 usually
        if n1 > 30 or n2 > 10: 
            return False
        if n1 < 2 or n2 < 2:
            return False
        return True

    def _plot_single_interaction(self, ax, x_col, hue_col):
        """Plots a single interaction bar chart on the given axes."""
        # Calculate means and standard errors for the plot
        # We use barplot which aggregates automatically
        
        # Sort order? 
        # Ideally sort X by mean score, or semantic order if available.
        # Here we use default or popularity sort?
        # Standard logic: Sort X levels by global mean, Hue levels by global mean
        
        # Order for X
        x_order = self.df.groupby(x_col)['Score'].mean().sort_values().index.tolist()
        # Order for Hue
        hue_order = self.df.groupby(hue_col)['Score'].mean().sort_values().index.tolist()
        
        sns.barplot(
            data=self.df, 
            x=x_col, 
            y='Score', 
            hue=hue_col, 
            ax=ax,
            order=x_order,
            hue_order=hue_order,
            palette='viridis',
            edgecolor='black',
            linewidth=0.5,
            errorbar=None  # Remove error bars for cleaner "Mean" view as requested ("value is ... mean standardized value")
        )
        
        # Formatting
        ax.set_title(f"{x_col} vs {hue_col}", fontsize=12)
        ax.set_xlabel(x_col, fontsize=10)
        ax.set_ylabel("Std. Score (MSE)", fontsize=10)
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        
        # Move legend
        ax.legend(title=hue_col, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

    def analyze_synergy_and_report(self):
        """
        Quantify 'Chemical Reaction' (Synergy): 1+1 > 2
        Synergy = Expected(Additive) - Actual
        Since Lower Score is Better:
        Synergy > 0 implies Actual is Lower (Better) than Expected.
        """
        logger.info("\n=== Synergy Analysis (1+1 > 2 Detection) ===")
        pairs = list(combinations(self.dimensions, 2))
        global_mean = self.df['Score'].mean()
        
        synergy_list = []
        
        for dim1, dim2 in pairs:
            # Group Means
            mean1 = self.df.groupby(dim1)['Score'].mean()
            mean2 = self.df.groupby(dim2)['Score'].mean()
            # Joint Means
            joint = self.df.groupby([dim1, dim2])['Score'].mean().reset_index()
            
            for _, row in joint.iterrows():
                val1, val2 = row[dim1], row[dim2]
                actual = row['Score']
                
                # Check exist
                if val1 not in mean1 or val2 not in mean2: continue
                
                m1 = mean1[val1]
                m2 = mean2[val2]
                
                # Expected (Additive Model)
                expected = m1 + m2 - global_mean
                
                # Synergy: How much BETTER is Actual than Expected?
                # Formula: Expected - Actual
                synergy = expected - actual
                
                synergy_list.append({
                    'Dimension1': dim1,
                    'Value1': val1,
                    'Dimension2': dim2,
                    'Value2': val2,
                    'Global_Mean': global_mean,
                    'Mean_1': m1,
                    'Mean_2': m2,
                    'Expected_Score': expected,
                    'Actual_Score': actual,
                    'Synergy_Score': synergy,
                    'Reaction_Type': 'Positive_Synergy' if synergy > 0 else 'Interference'
                })
        
        if not synergy_list: return
        
        synergy_df = pd.DataFrame(synergy_list).sort_values('Synergy_Score', ascending=False)
        out_path = os.path.join(self.inter_dir, 'synergy_top_findings.csv')
        synergy_df.to_csv(out_path, index=False)
        logger.info(f"Saved Synergy analysis to {out_path}")
        
        # Plot top 10 Positive Synergies
        top_synergies = synergy_df.head(10)
        self._plot_synergy_details(top_synergies, 'Top_Positive_Synergies')
        
        # Plot top 10 Negative Synergies (Interference)
        bottom_synergies = synergy_df.tail(10)
        self._plot_synergy_details(bottom_synergies, 'Top_Interference')

    def _plot_synergy_details(self, report_df, suffix):
        """Generates Interaction Plots (Point Plots) for specific pairs."""
        unique_pairs = report_df[['Dimension1', 'Dimension2']].drop_duplicates()
        
        for _, row in unique_pairs.iterrows():
            d1, d2 = row['Dimension1'], row['Dimension2']
            
            try:
                # Create Interaction Plot (pointplot is best for checking parallelism)
                plt.figure(figsize=(10, 6))
                sns.pointplot(data=self.df, x=d1, y='Score', hue=d2, 
                              markers='o', linestyles='-', errorbar=None, palette='viridis')
                plt.title(f"Interaction Analysis: {d1} vs {d2}\nNon-parallel lines indicate Chemical Reaction", fontsize=12)
                plt.ylabel("Standardized MSE (Lower is Better)")
                plt.xlabel(d1)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                safe_d1 = re.sub(r'[^\w\-]', '_', d1)
                safe_d2 = re.sub(r'[^\w\-]', '_', d2)
                plt.savefig(os.path.join(self.inter_dir, f'interaction_plot_{safe_d1}_{safe_d2}_{suffix}.pdf'))
                plt.close()
            except Exception as e:
                plt.close()
            except Exception as e:
                logger.warning(f"Failed to plot interaction for {d1} vs {d2}: {e}")
                plt.close()

    def fit_all_two_way_interactions(self, valid_interactions=None):
        """
        Fit OLS model with all two-way interaction terms and perform statistical tests.

        This addresses the reviewer's concern that the main-effects-only model
        implicitly assumes additivity, ignoring potential interactions.

        Args:
            valid_interactions: list of (dim1, dim2) tuples to include in the model.
                               If None, uses all dimension pairs.

        Returns:
            model: fitted OLS model with interaction terms
            interaction_results: DataFrame with F-test results and partial η² for each interaction
        """
        from statsmodels.formula.api import ols
        import statsmodels.api as sm

        logger.info("\n" + "="*60)
        logger.info("TWO-WAY INTERACTION STATISTICAL ANALYSIS")
        logger.info("="*60)

        # Get all dimension pairs
        if valid_interactions is None:
            pairs = list(combinations(self.dimensions, 2))
        else:
            pairs = valid_interactions

        # Build formula: main effects + interaction terms
        main_effects = [f"C(Q('{col}'))" for col in self.dimensions]
        interaction_terms = [f"C(Q('{a}')):C(Q('{b}'))" for a, b in pairs]

        formula = "Score ~ " + " + ".join(main_effects + interaction_terms)

        logger.info(f"Fitting model with {len(main_effects)} main effects and {len(interaction_terms)} interactions...")

        try:
            # Fit the model
            model = ols(formula, data=self.df).fit()

            # Perform Type III ANOVA
            anova = sm.stats.anova_lm(model, typ=3)

            # Extract interaction results
            interaction_results = []
            ss_residual = anova.loc['Residual', 'sum_sq']

            for term in anova.index:
                if ':' in term:  # This is an interaction term
                    ss_term = anova.loc[term, 'sum_sq']
                    f_val = anova.loc[term, 'F']
                    p_val = anova.loc[term, 'PR(>F)']

                    # Partial η² = SS_term / (SS_term + SS_residual)
                    partial_eta_sq = ss_term / (ss_term + ss_residual)

                    interaction_results.append({
                        'interaction': term,
                        'sum_sq': ss_term,
                        'F': f_val,
                        'p_value': p_val,
                        'partial_eta_sq': partial_eta_sq,
                        'significant': p_val < 0.05,
                        'effect_size': self._interpret_effect_size(partial_eta_sq)
                    })

            interaction_df = pd.DataFrame(interaction_results)
            interaction_df = interaction_df.sort_values('partial_eta_sq', ascending=False)

            # Save results
            interaction_df.to_csv(os.path.join(self.inter_dir, 'two_way_interaction_anova.csv'), index=False)

            # Summary statistics
            n_significant = interaction_df['significant'].sum()
            logger.info(f"\nFound {n_significant}/{len(interaction_df)} significant interactions (p < 0.05)")

            # Top interactions
            logger.info("\nTop 5 interactions by effect size (partial η²):")
            for _, row in interaction_df.head(5).iterrows():
                sig_marker = "***" if row['p_value'] < 0.001 else ("**" if row['p_value'] < 0.01 else ("*" if row['p_value'] < 0.05 else ""))
                logger.info(f"  {row['interaction']}: η²={row['partial_eta_sq']:.4f}, F={row['F']:.2f}, p={row['p_value']:.4f} {sig_marker}")

            return model, interaction_df

        except Exception as e:
            logger.error(f"Failed to fit interaction model: {e}")
            return None, None

    def test_interactions_f_test(self, model_main, model_interaction):
        """
        Perform F-test comparing main-effects model vs interaction model.

        H0: All interaction coefficients = 0 (no interactions)
        H1: At least one interaction coefficient ≠ 0

        Args:
            model_main: OLS model with only main effects
            model_interaction: OLS model with main effects + interactions

        Returns:
            dict with F-statistic, p-value, and significance
        """
        try:
            f_stat, p_value, df_diff = model_main.compare_f_test(model_interaction)

            result = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'df_num': df_diff,
                'significant': p_value < 0.05
            }

            logger.info("\n" + "-"*40)
            logger.info("F-TEST: Main Effects vs Interaction Model")
            logger.info("-"*40)
            logger.info(f"F-statistic: {f_stat:.4f}")
            logger.info(f"p-value: {p_value:.6f}")
            logger.info(f"Degrees of freedom: {df_diff}")
            logger.info(f"Conclusion: {'Interactions are significant' if p_value < 0.05 else 'No significant interactions'}")

            return result

        except Exception as e:
            logger.error(f"F-test failed: {e}")
            return None

    def compute_interaction_effect_sizes(self, model):
        """
        Compute partial η² (eta-squared) for each interaction term.

        Partial η² = SS_term / (SS_term + SS_residual)

        Interpretation:
        - 0.01: small effect
        - 0.06: medium effect
        - 0.14: large effect

        Args:
            model: fitted OLS model with interaction terms

        Returns:
            DataFrame with effect sizes for each interaction
        """
        import statsmodels.api as sm

        try:
            anova = sm.stats.anova_lm(model, typ=3)

            effect_sizes = []
            ss_residual = anova.loc['Residual', 'sum_sq']

            for term in anova.index:
                if ':' in term:  # Interaction term
                    ss_term = anova.loc[term, 'sum_sq']
                    partial_eta_sq = ss_term / (ss_term + ss_residual)

                    effect_sizes.append({
                        'interaction': term,
                        'sum_sq': ss_term,
                        'F': anova.loc[term, 'F'],
                        'p_value': anova.loc[term, 'PR(>F)'],
                        'partial_eta_sq': partial_eta_sq,
                        'effect_interpretation': self._interpret_effect_size(partial_eta_sq)
                    })

            return pd.DataFrame(effect_sizes).sort_values('partial_eta_sq', ascending=False)

        except Exception as e:
            logger.error(f"Failed to compute effect sizes: {e}")
            return None

    def _interpret_effect_size(self, eta_sq):
        """Interpret partial η² effect size."""
        if eta_sq >= 0.14:
            return 'large'
        elif eta_sq >= 0.06:
            return 'medium'
        elif eta_sq >= 0.01:
            return 'small'
        else:
            return 'negligible'

    def run_full_interaction_analysis(self):
        """
        Complete interaction analysis pipeline:
        1. Fit main-effects model
        2. Fit interaction model
        3. F-test comparing models
        4. Compute effect sizes
        5. Generate report
        """
        from statsmodels.formula.api import ols

        logger.info("\n" + "="*60)
        logger.info("FULL TWO-WAY INTERACTION ANALYSIS")
        logger.info("="*60)

        # Step 1: Fit main effects model
        main_effects = [f"C(Q('{col}'))" for col in self.dimensions]
        formula_main = "Score ~ " + " + ".join(main_effects)

        logger.info("\nStep 1: Fitting main-effects model...")
        model_main = ols(formula_main, data=self.df).fit()
        logger.info(f"Main model R²: {model_main.rsquared:.4f}")

        # Step 2: Fit interaction model
        logger.info("\nStep 2: Fitting interaction model...")
        model_interaction, interaction_results = self.fit_all_two_way_interactions()

        if model_interaction is None:
            logger.error("Failed to fit interaction model. Aborting.")
            return None

        logger.info(f"Interaction model R²: {model_interaction.rsquared:.4f}")
        logger.info(f"R² improvement: {model_interaction.rsquared - model_main.rsquared:.4f}")

        # Step 3: F-test
        logger.info("\nStep 3: Comparing models with F-test...")
        f_test_result = self.test_interactions_f_test(model_main, model_interaction)

        # Step 4: Save comprehensive results
        results = {
            'model_main': model_main,
            'model_interaction': model_interaction,
            'f_test': f_test_result,
            'interaction_effects': interaction_results
        }

        # Generate summary report
        self._generate_interaction_report(results)

        return results

    def _generate_interaction_report(self, results):
        """Generate a summary report for interaction analysis."""
        report_path = os.path.join(self.inter_dir, 'interaction_analysis_report.txt')

        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("TWO-WAY INTERACTION ANALYSIS REPORT\n")
            f.write("="*60 + "\n\n")

            # Model comparison
            f.write("1. MODEL COMPARISON\n")
            f.write("-"*40 + "\n")
            f.write(f"Main-effects R²: {results['model_main'].rsquared:.4f}\n")
            f.write(f"Interaction model R²: {results['model_interaction'].rsquared:.4f}\n")
            f.write(f"R² improvement: {results['model_interaction'].rsquared - results['model_main'].rsquared:.4f}\n\n")

            # F-test
            if results['f_test']:
                f.write("2. F-TEST RESULTS\n")
                f.write("-"*40 + "\n")
                f.write(f"F-statistic: {results['f_test']['f_statistic']:.4f}\n")
                f.write(f"p-value: {results['f_test']['p_value']:.6f}\n")
                f.write(f"Significant: {results['f_test']['significant']}\n\n")

            # Significant interactions
            if results['interaction_effects'] is not None:
                sig_interactions = results['interaction_effects'][results['interaction_effects']['significant']]
                f.write("3. SIGNIFICANT INTERACTIONS (p < 0.05)\n")
                f.write("-"*40 + "\n")
                f.write(f"Count: {len(sig_interactions)}/{len(results['interaction_effects'])}\n\n")

                if len(sig_interactions) > 0:
                    f.write("Top significant interactions by effect size:\n")
                    for _, row in sig_interactions.head(10).iterrows():
                        f.write(f"  - {row['interaction']}\n")
                        f.write(f"    Partial η²: {row['partial_eta_sq']:.4f} ({row['effect_size']})\n")
                        f.write(f"    F: {row['F']:.2f}, p: {row['p_value']:.4f}\n")

        logger.info(f"Saved interaction report to {report_path}")

    def generate_paper_heatmaps(self):
        """
        Generates specific heatmaps requested for the benchmark paper.
        Highlights:
        1. Network Architecture vs Feature Attention
        2. Loss Function vs Series Normalization
        """
        heatmap_dir = os.path.join(self.output_dir, 'paper_heatmaps')
        os.makedirs(heatmap_dir, exist_ok=True)
        
        logger.info("\n=== Generating Paper Heatmaps ===")
        
        # 1. Network Architecture vs Feature Attention
        # Handle 'None' mapping for feature_attn if necessary
        # The user requested specific axes order
        self._plot_custom_heatmap(
            x_dim='network_architecture', 
            y_dim='feature_attn', 
            filter_dict={
                'network_architecture': ['MLP', 'RNN', 'Transformer', 'LLM', 'TSFM'],
                # We will normalize filtering in the method handles different representations of None
            },
            output_dir=heatmap_dir,
            title_suffix="(Model Structure)"
        )
        
        # 2. Loss Function vs Series Normalization
        # Identify correct Series Normalization column. Likely 'gym_series_norm'
        norm_col = 'gym_series_norm' if 'gym_series_norm' in self.df.columns else None
        if not norm_col:
            # Fallback check
            for c in self.df.columns:
                if 'series_norm' in c:
                    norm_col = c
                    break
        
        if norm_col:
            self._plot_custom_heatmap(
                x_dim='loss_func', 
                y_dim=norm_col,
                output_dir=heatmap_dir,
                title_suffix="(Optimization)"
            )
        else:
            logger.warning("Could not find Series Normalization column for heatmap.")

    def _plot_custom_heatmap(self, x_dim, y_dim, output_dir, filter_dict=None, title_suffix=""):
        """
        Plots two heatmaps: 
        1. Actual Score (Performance)
        2. Synergy Score (Interaction Effect)
        """
        try:
            # Prepare Data
            # Fill NaNs with 'None' for categorical matching
            plot_df = self.df.copy()
            if x_dim in plot_df.columns:
                plot_df[x_dim] = plot_df[x_dim].fillna('None').astype(str)
            if y_dim in plot_df.columns:
                plot_df[y_dim] = plot_df[y_dim].fillna('None').astype(str)
                
            # Filter if requested
            if filter_dict:
                for dim, allowed_values in filter_dict.items():
                    if dim in plot_df.columns:
                        # Normalize allowed values to string to match
                        allowed_str = [str(v) for v in allowed_values]
                        # Handling 'None' variations
                        if 'None' in allowed_str:
                            allowed_str.extend(['nan', 'null', 'NaN'])
                        
                        plot_df = plot_df[plot_df[dim].isin(allowed_str)]
            
            if plot_df.empty:
                logger.warning(f"No data found for heatmap {x_dim} vs {y_dim} after filtering.")
                return

            # Calculate Aggregate Scores (Actual)
            pivot_actual = plot_df.groupby([y_dim, x_dim])['Score'].mean().unstack()
            
            # Calculate Synergy Matrix (Expected - Actual)
            # We need marginal means for the filtered subset or global? 
            # Ideally Synergy is global property, but for this plot we might want 'local synergy' 
            # relative to the subset mean to highlight differences within the subset.
            # However, strictly Synergy should use the Global Mean from the full dataset.
            global_mean = self.df['Score'].mean()
            mean_x = self.df.groupby(x_dim)['Score'].mean()
            mean_y = self.df.groupby(y_dim)['Score'].mean()
            
            # Calculate Synergy for each cell in the pivot
            pivot_synergy = pd.DataFrame(index=pivot_actual.index, columns=pivot_actual.columns)
            
            for r in pivot_actual.index:
                for c in pivot_actual.columns:
                    actual = pivot_actual.loc[r, c]
                    if pd.isna(actual): continue
                    
                    # Get marginal means (handle string conversion lookup)
                    # Note: mean_x index might not be stringified yet if original df had mixed types
                    # Try to lookup safely
                    try:
                        mx = mean_x.loc[c] if c in mean_x.index else mean_x.get(c, np.nan)
                        my = mean_y.loc[r] if r in mean_y.index else mean_y.get(r, np.nan)
                    except:
                        mx, my = np.nan, np.nan
                        
                    if pd.isna(mx) or pd.isna(my):
                        pivot_synergy.loc[r, c] = np.nan
                    else:
                        expected = mx + my - global_mean
                        pivot_synergy.loc[r, c] = expected - actual

            pivot_synergy = pivot_synergy.astype(float)

            # Define plot helper
            def save_heatmap(data, title, filename, cmap, center=None):
                plt.figure(figsize=(10, 8))
                sns.heatmap(data, annot=True, fmt=".2f", cmap=cmap, center=center,
                           linewidths=.5, square=True, cbar_kws={"shrink": .5},
                           annot_kws={"size": 18, "weight": "bold"})
                # plt.title(title, fontsize=14)
                plt.xlabel("")
                plt.ylabel("")
                plt.xticks(fontsize=16, weight='bold', rotation=45)
                plt.yticks(fontsize=16, weight='bold', rotation=0)
                plt.tight_layout(pad=0.2)
                plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', pad_inches=0.02)
                plt.close()

            # Plot 1: Actual Performance (Std MSE)
            # Blue is better (lower) for MSE, but usually heatmap uses Red=High, Blue=Low.
            # If metric is MSE, Low (Blue) is Good.
            save_heatmap(pivot_actual, 
                         f"Performance (Std. MSE): {x_dim} vs {y_dim}\n(Lower/Blue is Better)", 
                         f"heatmap_score_{x_dim}_vs_{y_dim}.pdf", 
                         "coolwarm", center=0) # Centered at 0 (Mean)

            # Plot 2: Synergy Score
            # Positive (Red/Green?) is Good. 
            # Synergy > 0 means Better than Expected. 
            # Usually Red is Hot/Active. Let's use a divergent palette.
            # If Synergy > 0 (Good), maybe we want Green? 
            # Standard 'coolwarm': Red is Positive, Blue is Negative.
            # Synergy > 0 (Red) = "Active Synergy" (1+1>2).
            # Synergy < 0 (Blue) = "Interference".
            save_heatmap(pivot_synergy, 
                         f"Synergy (Interactive Effect): {x_dim} vs {y_dim}\n(Red > 0 is Synergistic 1+1>2)", 
                         f"heatmap_synergy_{x_dim}_vs_{y_dim}.pdf", 
                         "coolwarm", center=0)
                         
            logger.info(f"Generated heatmaps for {x_dim} vs {y_dim}")
            
        except Exception as e:
            logger.warning(f"Failed to generate heatmap for {x_dim} vs {y_dim}: {e}")
            import traceback
            traceback.print_exc() 

