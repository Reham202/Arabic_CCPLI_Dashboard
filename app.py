# ==============================================================================
# CCPLI V4 OFFICIAL ENGINE
# Comprehensive Composite Power Language Index
# Production / Statistical Protocol
# ==============================================================================

import numpy as np
import pandas as pd

from scipy.stats import spearmanr, kendalltau

import warnings

warnings.filterwarnings("ignore")


class CCPLIV4OfficialEngine:
    """
    CCPLI V4 Official Framework Engine

    Framework:
    - 10 Strategic Dimensions weighted by AHP.
    - 50 Sub-Indicators.
    - Equal allocation within each dimension:
          w_ij = W_i / 5
    - Linear weighted aggregation as the Base Model.
    - Zero-safe geometric aggregation as the Robustness Model.
    - Calibrated Dirichlet Monte Carlo simulation.
    - Weight perturbation sensitivity analysis.
    - Leave-One-Dimension-Out (LODO).
    - Spearman's rho.
    - Kendall's tau.
    - MARC.
    - MaxRC.
    - Content Validity Index (CVI).

    Important:
    The default dimension weights are the officially specified
    10-dimension weights in the current CCPLI V4 framework.
    """

    # ==========================================================================
    # INITIALIZATION
    # ==========================================================================

    def __init__(self, raw_ahp_weights=None):

        # ----------------------------------------------------------------------
        # 1. Official AHP Dimension Weights
        # ----------------------------------------------------------------------

        if raw_ahp_weights is not None:

            if not isinstance(raw_ahp_weights, dict):
                raise TypeError(
                    "raw_ahp_weights must be provided as a dictionary."
                )

            self.dimension_weights = raw_ahp_weights.copy()

        else:

            self.dimension_weights = {

                "D1_Digital_Algorithmic": 0.12,

                "D2_TAFL_Education": 0.10,

                "D3_Linguistic_Structure": 0.10,

                "D4_Religious_Sacred": 0.10,

                "D5_Civilizational_Historical": 0.08,

                "D6_Knowledge_Scientific_Publishing": 0.10,

                "D7_Economic_Trade_Exposure": 0.12,

                "D8_Demographic_Spread": 0.10,

                "D9_Media_Public_Digital_Presence": 0.08,

                "D10_Geopolitical_Language_Policies": 0.10
            }

        # ----------------------------------------------------------------------
        # Validate Dimension Weights
        # ----------------------------------------------------------------------

        if len(self.dimension_weights) != 10:

            raise ValueError(
                "CCPLI V4 requires exactly 10 strategic dimensions."
            )

        for key, value in self.dimension_weights.items():

            if value < 0:

                raise ValueError(
                    f"Weight for {key} cannot be negative."
                )

        total_w = sum(self.dimension_weights.values())

        if total_w <= 0:

            raise ValueError(
                "The total dimension weight must be greater than zero."
            )

        # ----------------------------------------------------------------------
        # Normalize weights so total = 1.0
        # ----------------------------------------------------------------------

        self.dimension_weights = {

            key: value / total_w

            for key, value in self.dimension_weights.items()
        }

        # ----------------------------------------------------------------------
        # 2. Create 50 Sub-Indicators
        # ----------------------------------------------------------------------

        self.indicators = []

        self.indicator_weights = {}

        for dim_code, dim_weight in self.dimension_weights.items():

            # Equal distribution among five indicators
            sub_weight = dim_weight / 5.0

            for i in range(1, 6):

                indicator_code = f"{dim_code}_I{i}"

                self.indicators.append(indicator_code)

                self.indicator_weights[indicator_code] = sub_weight

        # Final structural validation
        if len(self.indicators) != 50:

            raise RuntimeError(
                "Internal error: CCPLI V4 must contain exactly 50 indicators."
            )

    # ==========================================================================
    # 1. AHP CONSISTENCY CHECK
    # ==========================================================================

    def calculate_ahp_consistency(self, pairwise_matrix):
        """
        Calculate AHP Consistency Ratio (CR).

        Guideline:
            CR <= 0.10  -> acceptable consistency

        Parameters
        ----------
        pairwise_matrix : numpy.ndarray
            Square reciprocal AHP pairwise comparison matrix.

        Returns
        -------
        dict
            Lambda_Max
            CI
            CR
            Guideline_Pass
        """

        pairwise_matrix = np.asarray(pairwise_matrix, dtype=float)

        if pairwise_matrix.ndim != 2:

            raise ValueError(
                "AHP pairwise matrix must be two-dimensional."
            )

        rows, cols = pairwise_matrix.shape

        if rows != cols:

            raise ValueError(
                "AHP pairwise matrix must be square."
            )

        n = rows

        if n < 3:

            raise ValueError(
                "AHP consistency analysis requires at least 3 criteria."
            )

        # Saaty's Random Index
        ri_dict = {

            1: 0.00,
            2: 0.00,
            3: 0.58,
            4: 0.90,
            5: 1.12,
            6: 1.24,
            7: 1.32,
            8: 1.41,
            9: 1.45,
            10: 1.49
        }

        # Eigenvalues
        eigenvalues, _ = np.linalg.eig(pairwise_matrix)

        real_eigenvalues = np.real(eigenvalues)

        max_eigenvalue = np.max(real_eigenvalues)

        # Consistency Index
        ci = (max_eigenvalue - n) / (n - 1)

        # Random Index
        ri = ri_dict.get(n, 1.49)

        # Consistency Ratio
        if ri > 0:

            cr = ci / ri

        else:

            cr = 0.0

        return {

            "Lambda_Max": float(max_eigenvalue),

            "CI": float(ci),

            "CR": float(cr),

            "Guideline_Pass": bool(cr <= 0.10)
        }

    # ==========================================================================
    # 2. NORMALIZATION
    # ==========================================================================

    def normalize_min_max(self, df, directions=None):
        """
        Standard Min-Max normalization to [0, 100].

        Positive direction:
            ((x - min) / (max - min)) * 100

        Negative direction:
            ((max - x) / (max - min)) * 100

        If max == min:
            score = 50

        Parameters
        ----------
        df : pandas.DataFrame
        directions : dict, optional
            1  = positive direction
            -1 = negative direction

        Returns
        -------
        pandas.DataFrame
        """

        if not isinstance(df, pd.DataFrame):

            raise TypeError(
                "df must be a pandas DataFrame."
            )

        directions = directions or {

            column: 1

            for column in df.columns
        }

        df_norm = pd.DataFrame(index=df.index)

        for column in df.columns:

            min_val = df[column].min()

            max_val = df[column].max()

            direction = directions.get(column, 1)

            if direction not in (1, -1):

                raise ValueError(
                    f"Direction for {column} must be 1 or -1."
                )

            if max_val == min_val:

                df_norm[column] = 50.0

            elif direction == 1:

                df_norm[column] = (

                    (df[column] - min_val)

                    / (max_val - min_val)

                ) * 100.0

            else:

                df_norm[column] = (

                    (max_val - df[column])

                    / (max_val - min_val)

                ) * 100.0

        return df_norm

    # ==========================================================================
    # 3. VALIDATE INDICATOR DATA
    # ==========================================================================

    def _validate_indicator_dataframe(self, df):

        if not isinstance(df, pd.DataFrame):

            raise TypeError(
                "Input must be a pandas DataFrame."
            )

        missing = [

            indicator

            for indicator in self.indicators

            if indicator not in df.columns
        ]

        if missing:

            raise ValueError(

                "The following required CCPLI indicators are missing:\n"

                + "\n".join(missing)
            )

        if df[self.indicators].isnull().any().any():

            raise ValueError(
                "The dataset contains missing values in CCPLI indicators."
            )

        return True

    # ==========================================================================
    # 4. LINEAR AGGREGATION
    # ==========================================================================

    def aggregate_linear(self, df_norm, weights=None):
        """
        Base Model:

            LPI_linear = Σ(w_j * I_j)

        where:
            I_j = normalized indicator score
            w_j = indicator weight
        """

        self._validate_indicator_dataframe(df_norm)

        if weights is None:

            weights = self.indicator_weights

        weight_series = pd.Series(weights, dtype=float)

        missing_weights = [

            indicator

            for indicator in self.indicators

            if indicator not in weight_series.index
        ]

        if missing_weights:

            raise ValueError(
                "Weights are missing for one or more indicators."
            )

        # Use only CCPLI indicators in correct order
        weight_series = weight_series.reindex(self.indicators)

        return df_norm[self.indicators].dot(weight_series)

    # ==========================================================================
    # 5. ZERO-SAFE GEOMETRIC AGGREGATION
    # ==========================================================================

    def aggregate_geometric_zero_safe(self, df_norm, weights=None):
        """
        Robustness Model:

            LPI_G =
                exp[ Σ(w_j * ln(1 + I_j)) ] - 1

        log1p is used to ensure numerical stability
        when an indicator equals zero.
        """

        self._validate_indicator_dataframe(df_norm)

        if weights is None:

            weights = self.indicator_weights

        weight_series = pd.Series(weights, dtype=float)

        weight_series = weight_series.reindex(self.indicators)

        log_term = np.log1p(
            df_norm[self.indicators]
        )

        weighted_log = log_term.dot(weight_series)

        return np.expm1(weighted_log)

    # ==========================================================================
    # 6. CALIBRATED DIRICHLET MONTE CARLO
    # ==========================================================================

    def run_calibrated_dirichlet_monte_carlo(
        self,
        df_norm,
        num_simulations=10000,
        precision_k=100,
        random_seed=42
    ):
        """
        Calibrated Dirichlet Monte Carlo Simulation.

        alpha_i = K * W_i

        where:

            W_i = official dimension weight
            K   = precision/concentration parameter

        Default:
            10,000 simulations
            K = 100
            random seed = 42
        """

        self._validate_indicator_dataframe(df_norm)

        if num_simulations <= 0:

            raise ValueError(
                "num_simulations must be greater than zero."
            )

        if precision_k <= 0:

            raise ValueError(
                "precision_k must be greater than zero."
            )

        # ----------------------------------------------------------------------
        # Base model
        # ----------------------------------------------------------------------

        base_scores = self.aggregate_linear(df_norm)

        base_ranks = base_scores.rank(
            ascending=False,
            method="average"
        )

        # ----------------------------------------------------------------------
        # Dimension weights
        # ----------------------------------------------------------------------

        dim_names = list(
            self.dimension_weights.keys()
        )

        base_dim_weights = np.array(

            [
                self.dimension_weights[d]

                for d in dim_names
            ],

            dtype=float
        )

        # ----------------------------------------------------------------------
        # Dirichlet alpha parameters
        # ----------------------------------------------------------------------

        alpha_params = base_dim_weights * precision_k

        # ----------------------------------------------------------------------
        # Random generator
        # ----------------------------------------------------------------------

        rng = np.random.default_rng(
            random_seed
        )

        # ----------------------------------------------------------------------
        # Storage
        # ----------------------------------------------------------------------

        sim_scores = []

        sim_ranks = []

        # ----------------------------------------------------------------------
        # Monte Carlo loop
        # ----------------------------------------------------------------------

        for _ in range(num_simulations):

            sampled_dim_weights = rng.dirichlet(
                alpha_params
            )

            sampled_indicator_weights = {}

            for idx, dim_code in enumerate(dim_names):

                sub_weight = (
                    sampled_dim_weights[idx] / 5.0
                )

                for i in range(1, 6):

                    indicator_code = (
                        f"{dim_code}_I{i}"
                    )

                    sampled_indicator_weights[
                        indicator_code
                    ] = sub_weight

            scores = self.aggregate_linear(

                df_norm,

                weights=sampled_indicator_weights
            )

            ranks = scores.rank(

                ascending=False,

                method="average"
            )

            sim_scores.append(scores)

            sim_ranks.append(ranks)

        # ----------------------------------------------------------------------
        # Convert simulation results to DataFrames
        # ----------------------------------------------------------------------

        scores_df = pd.DataFrame(
            sim_scores
        )

        ranks_df = pd.DataFrame(
            sim_ranks
        )

        # ----------------------------------------------------------------------
        # 95% Confidence Intervals
        # ----------------------------------------------------------------------

        score_ci_low = scores_df.quantile(
            0.025,
            axis=0
        )

        score_ci_up = scores_df.quantile(
            0.975,
            axis=0
        )

        # ----------------------------------------------------------------------
        # Rank displacement
        # ----------------------------------------------------------------------

        rank_difference = (

            ranks_df

            .sub(base_ranks, axis="columns")

        )

        marc = (
            rank_difference
            .abs()
            .mean(axis=0)
        )

        max_rc = (
            rank_difference
            .abs()
            .max(axis=0)
        )

        # ----------------------------------------------------------------------
        # Rank correlations
        # ----------------------------------------------------------------------

        spearmans = []

        kendalls = []

        for i in range(num_simulations):

            rho = spearmanr(

                base_ranks,

                ranks_df.iloc[i]

            ).statistic

            tau = kendalltau(

                base_ranks,

                ranks_df.iloc[i]

            ).statistic

            # Avoid possible NaN values
            if np.isfinite(rho):

                spearmans.append(rho)

            if np.isfinite(tau):

                kendalls.append(tau)

        avg_spearman = (

            float(np.mean(spearmans))

            if spearmans

            else np.nan
        )

        avg_kendall = (

            float(np.mean(kendalls))

            if kendalls

            else np.nan
        )

        return {

            "Base_Scores": base_scores,

            "Base_Ranks": base_ranks,

            "Score_95_CI_Lower": score_ci_low,

            "Score_95_CI_Upper": score_ci_up,

            "MARC_per_Language": marc,

            "MaxRC_per_Language": max_rc,

            "Avg_Spearman_Rho": avg_spearman,

            "Avg_Kendall_Tau": avg_kendall,

            "Simulation_Count": num_simulations,

            "Precision_K": precision_k
        }

    # ==========================================================================
    # 7. WEIGHT PERTURBATION
    # ==========================================================================

    def weight_perturbation_scenario(
        self,
        df_norm,
        factor=1.20
    ):
        """
        Weight Perturbation Scenario.

        One dimension is multiplied by the specified factor
        and all dimension weights are subsequently renormalized.

        Default:
            factor = 1.20 (+20%)
        """

        self._validate_indicator_dataframe(df_norm)

        if factor <= 0:

            raise ValueError(
                "Perturbation factor must be greater than zero."
            )

        base_scores = self.aggregate_linear(
            df_norm
        )

        base_ranks = base_scores.rank(
            ascending=False,
            method="average"
        )

        results = {}

        for dim_code in self.dimension_weights.keys():

            perturbed_dimensions = (
                self.dimension_weights.copy()
            )

            perturbed_dimensions[dim_code] *= factor

            # Renormalization
            total_weight = sum(
                perturbed_dimensions.values()
            )

            normalized_dimensions = {

                key: value / total_weight

                for key, value
                in perturbed_dimensions.items()
            }

            # Convert dimension weights
            # into 50 indicator weights
            new_indicator_weights = {}

            for dimension, dimension_weight in (
                normalized_dimensions.items()
            ):

                for i in range(1, 6):

                    indicator_code = (
                        f"{dimension}_I{i}"
                    )

                    new_indicator_weights[
                        indicator_code
                    ] = dimension_weight / 5.0

            # New scores
            new_scores = self.aggregate_linear(

                df_norm,

                weights=new_indicator_weights
            )

            new_ranks = new_scores.rank(

                ascending=False,

                method="average"
            )

            rho = spearmanr(

                base_ranks,

                new_ranks

            ).statistic

            tau = kendalltau(

                base_ranks,

                new_ranks

            ).statistic

            marc = (

                base_ranks

                .sub(new_ranks)

                .abs()

                .mean()
            )

            max_rc = (

                base_ranks

                .sub(new_ranks)

                .abs()

                .max()
            )

            results[dim_code] = {

                "Spearman_Rho": rho,

                "Kendall_Tau": tau,

                "MARC": marc,

                "MaxRC": max_rc
            }

        return pd.DataFrame(
            results
        ).T

    # ==========================================================================
    # 8. LEAVE-ONE-DIMENSION-OUT ANALYSIS
    # ==========================================================================

    def leave_one_dimension_out_analysis(
        self,
        df_norm
    ):
        """
        Leave-One-Dimension-Out (LODO).

        Each dimension is removed once.

        The remaining indicator weights are renormalized
        to maintain a total weight of 1.0.
        """

        self._validate_indicator_dataframe(df_norm)

        base_scores = self.aggregate_linear(
            df_norm
        )

        base_ranks = base_scores.rank(

            ascending=False,

            method="average"
        )

        results = {}

        for dim_code in self.dimension_weights.keys():

            # Remaining indicators
            remaining_indicators = [

                indicator

                for indicator in self.indicators

                if not indicator.startswith(dim_code)
            ]

            # Original weights
            raw_weights = np.array(

                [
                    self.indicator_weights[indicator]

                    for indicator
                    in remaining_indicators
                ],

                dtype=float
            )

            # Renormalize
            new_weights = (
                raw_weights
                / np.sum(raw_weights)
            )

            weight_dict = dict(

                zip(
                    remaining_indicators,
                    new_weights
                )
            )

            scores_lodo = (

                df_norm[
                    remaining_indicators
                ]

                .dot(
                    pd.Series(weight_dict)
                )
            )

            ranks_lodo = scores_lodo.rank(

                ascending=False,

                method="average"
            )

            rho = spearmanr(

                base_ranks,

                ranks_lodo

            ).statistic

            tau = kendalltau(

                base_ranks,

                ranks_lodo

            ).statistic

            marc = (

                base_ranks

                .sub(ranks_lodo)

                .abs()

                .mean()
            )

            max_rc = (

                base_ranks

                .sub(ranks_lodo)

                .abs()

                .max()
            )

            results[dim_code] = {

                "Spearman_Rho": rho,

                "Kendall_Tau": tau,

                "MARC": marc,

                "MaxRC": max_rc
            }

        return pd.DataFrame(
            results
        ).T

    # ==========================================================================
    # 9. CONTENT VALIDITY INDEX
    # ==========================================================================

    def calculate_cvi_expert_matrix(
        self,
        expert_ratings_df
    ):
        """
        Content Validity Index (CVI).

        Expert rating scale:
            1 = Not relevant
            2 = Somewhat relevant
            3 = Relevant
            4 = Highly relevant

        I-CVI:
            Number of experts rating item >= 3
            divided by total number of experts.

        S-CVI/Ave:
            Mean of all I-CVI values.

        Current validation threshold:
            S-CVI/Ave >= 0.80
        """

        if not isinstance(
            expert_ratings_df,
            pd.DataFrame
        ):

            raise TypeError(
                "expert_ratings_df must be a pandas DataFrame."
            )

        if expert_ratings_df.empty:

            raise ValueError(
                "Expert ratings matrix cannot be empty."
            )

        # ----------------------------------------------------------------------
        # Validate rating values
        # ----------------------------------------------------------------------

        if (

            expert_ratings_df
            .isnull()
            .any()
            .any()
        ):

            raise ValueError(
                "Expert ratings contain missing values."
            )

        invalid_values = (

            (expert_ratings_df < 1)

            | (expert_ratings_df > 4)
        ).any().any()

        if invalid_values:

            raise ValueError(
                "CVI ratings must be between 1 and 4."
            )

        # ----------------------------------------------------------------------
        # Check expected number of indicators
        # ----------------------------------------------------------------------

        if len(expert_ratings_df.index) != 50:

            raise ValueError(

                "CVI matrix should contain exactly "
                "50 indicators for CCPLI V4."
            )

        # ----------------------------------------------------------------------
        # I-CVI
        # ----------------------------------------------------------------------

        relevant_ratings = (

            expert_ratings_df >= 3

        ).sum(axis=1)

        total_experts = (
            expert_ratings_df.shape[1]
        )

        i_cvi = (
            relevant_ratings
            / total_experts
        )

        # ----------------------------------------------------------------------
        # S-CVI/Ave
        # ----------------------------------------------------------------------

        s_cvi_ave = i_cvi.mean()

        return {

            "Item_CVI_per_Indicator": i_cvi,

            "Overall_S_CVI_Average": float(
                s_cvi_ave
            ),

            "Pass_Validation": bool(
                s_cvi_ave >= 0.80
            ),

            "Number_of_Experts": int(
                total_experts
            )
        }

    # ==========================================================================
    # 10. WEIGHT SUMMARY
    # ==========================================================================

    def get_dimension_weight_table(self):
        """
        Return the official 10-dimension weight table.
        """

        rows = []

        for dimension, weight in (
            self.dimension_weights.items()
        ):

            rows.append({

                "Dimension": dimension,

                "Dimension_Weight": weight,

                "Dimension_Weight_Percent":
                    weight * 100,

                "Sub_Indicator_Weight":
                    weight / 5.0,

                "Sub_Indicator_Weight_Percent":
                    weight / 5.0 * 100
            })

        return pd.DataFrame(rows)

    # ==========================================================================
    # 11. 50-INDICATOR WEIGHT TABLE
    # ==========================================================================

    def get_indicator_weight_table(self):
        """
        Return all 50 indicator weights.
        """

        rows = []

        for indicator in self.indicators:

            dimension = indicator.rsplit(
                "_I",
                1
            )[0]

            rows.append({

                "Indicator": indicator,

                "Dimension": dimension,

                "Weight":
                    self.indicator_weights[indicator],

                "Weight_Percent":
                    self.indicator_weights[indicator] * 100
            })

        return pd.DataFrame(rows)


# ==============================================================================
# TEST / DEMONSTRATION PROTOCOL
# ==============================================================================

def run_demo():
    """
    Complete operational demonstration
    for CCPLI V4 Official Engine.
    """

    print("=" * 78)

    print(
        "        CCPLI V4 OFFICIAL ENGINE"
    )

    print(
        "        Final Statistical Protocol"
    )

    print("=" * 78)

    # ==========================================================================
    # Initialize Engine
    # ==========================================================================

    engine = CCPLIV4OfficialEngine()

    # ==========================================================================
    # 1. Dimension Weights
    # ==========================================================================

    print("\n")
    print("-" * 78)

    print(
        "1. OFFICIAL DIMENSION WEIGHTS"
    )

    print("-" * 78)

    dimension_table = (
        engine
        .get_dimension_weight_table()
    )

    print(
        dimension_table.to_string(
            index=False
        )
    )

    print(
        "\nTotal dimension weight:",
        round(
            sum(
                engine.dimension_weights.values()
            ),
            6
        )
    )

    # ==========================================================================
    # 2. 50 Indicator Weights
    # ==========================================================================

    print("\n")
    print("-" * 78)

    print(
        "2. THE 50 SUB-INDICATOR WEIGHTS"
    )

    print("-" * 78)

    indicator_table = (
        engine
        .get_indicator_weight_table()
    )

    print(
        indicator_table.to_string(
            index=False
        )
    )

    print(
        "\nTotal indicator weight:",
        round(
            sum(
                engine.indicator_weights.values()
            ),
            6
        )
    )

    # ==========================================================================
    # 3. Generate Demonstration Data
    # ==========================================================================

    print("\n")
    print("-" * 78)

    print(
        "3. GENERATING DEMONSTRATION DATA"
    )

    print("-" * 78)

    languages = [

        "Arabic",

        "English",

        "French",

        "Chinese",

        "Spanish",

        "German"
    ]

    # Reproducible demonstration data
    rng = np.random.default_rng(42)

    raw_data = rng.uniform(

        10,

        90,

        size=(
            len(languages),
            50
        )
    )

    df_raw = pd.DataFrame(

        raw_data,

        index=languages,

        columns=engine.indicators
    )

    print(
        f"Number of languages: {len(languages)}"
    )

    print(
        f"Number of indicators: {len(engine.indicators)}"
    )

    # ==========================================================================
    # 4. Normalize
    # ==========================================================================

    print("\n")
    print("-" * 78)

    print(
        "4. MIN-MAX NORMALIZATION"
    )

    print("-" * 78)

    df_norm = engine.normalize_min_max(
        df_raw
    )

    print(
        "Normalization completed successfully."
    )

    # ==========================================================================
    # 5. Linear Aggregation
    # ==========================================================================

    print("\n")
    print("-" * 78)

    print(
        "5. LINEAR BASE MODEL"
    )

    print("-" * 78)

    linear_scores = (
        engine.aggregate_linear(
            df_norm
        )
    )

    # ==========================================================================
    # 6. Geometric Aggregation
    # ==========================================================================

    print("\n")
    print("-" * 78)

    print(
        "6. ZERO-SAFE GEOMETRIC MODEL"
    )

    print("-" * 78)

    geometric_scores = (

        engine
        .aggregate_geometric_zero_safe(
            df_norm
        )
    )

    # ==========================================================================
    # 7. Compare Models
    # ==========================================================================

    comparison_table = pd.DataFrame({

        "Linear_Score":
            linear_scores,

        "Linear_Rank":
            linear_scores.rank(
                ascending=False
            ),

        "Geometric_Score":
            geometric_scores,

        "Geometric_Rank":
            geometric_scores.rank(
                ascending=False
            )
    })

    print(
        comparison_table.round(4)
    )

    # ==========================================================================
    # 8. Monte Carlo
    # ==========================================================================

    print("\n")
    print("-" * 78)

    print(
        "7. CALIBRATED DIRICHLET MONTE CARLO"
    )

    print("-" * 78)

    print(
        "Running 10,000 simulations..."
    )

    mc_results = (

        engine
        .run_calibrated_dirichlet_monte_carlo(

            df_norm,

            num_simulations=10000,

            precision_k=100,

            random_seed=42
        )
    )

    print(
        "\nAverage Spearman Rho:",
        f"{mc_results['Avg_Spearman_Rho']:.6f}"
    )

    print(
        "Average Kendall Tau:",
        f"{mc_results['Avg_Kendall_Tau']:.6f}"
    )

    # ==========================================================================
    # 9. Monte Carlo Table
    # ==========================================================================

    mc_table = pd.DataFrame({

        "Base_Rank":
            mc_results["Base_Ranks"],

        "Score_95_CI_Low":
            mc_results[
                "Score_95_CI_Lower"
            ],

        "Score_95_CI_Upper":
            mc_results[
                "Score_95_CI_Upper"
            ],

        "MARC":
            mc_results[
                "MARC_per_Language"
            ],

        "MaxRC":
            mc_results[
                "MaxRC_per_Language"
            ]
    })

    print("\n")
    print(
        mc_table.round(4)
    )

    # ==========================================================================
    # 10. Weight Perturbation
    # ==========================================================================

    print("\n")
    print("-" * 78)

    print(
        "8. WEIGHT PERTURBATION (+20%)"
    )

    print("-" * 78)

    perturbation_results = (

        engine
        .weight_perturbation_scenario(

            df_norm,

            factor=1.20
        )
    )

    print(
        perturbation_results.round(4)
    )

    # ==========================================================================
    # 11. LODO
    # ==========================================================================

    print("\n")
    print("-" * 78)

    print(
        "9. LEAVE-ONE-DIMENSION-OUT (LODO)"
    )

    print("-" * 78)

    lodo_results = (

        engine
        .leave_one_dimension_out_analysis(
            df_norm
        )
    )

    print(
        lodo_results.round(4)
    )

    # ==========================================================================
    # 12. Final Ranking
    # ==========================================================================

    print("\n")
    print("-" * 78)

    print(
        "10. FINAL BASE-MODEL RANKING"
    )

    print("-" * 78)

    final_ranking = pd.DataFrame({

        "Language":
            linear_scores.index,

        "CCPLI_Score":
            linear_scores.values,

        "Rank":
            linear_scores.rank(
                ascending=False,
                method="min"
            ).astype(int)
    })

    final_ranking = (
        final_ranking
        .sort_values("Rank")
        .reset_index(drop=True)
    )

    print(
        final_ranking.round(4)
    )

    # ==========================================================================
    # Completion
    # ==========================================================================

    print("\n")
    print("=" * 78)

    print(
        "CCPLI V4 EXECUTION COMPLETED SUCCESSFULLY."
    )

    print("=" * 78)


# ==============================================================================
# WINDOWS ENTRY POINT
# ==============================================================================

if __name__ == "__main__":

    try:

        run_demo()

    except Exception as error:

        print("\n")
        print("=" * 78)

        print(
            "CCPLI V4 EXECUTION ERROR"
        )

        print("=" * 78)

        print(
            f"\nError type: {type(error).__name__}"
        )

        print(
            f"Error message: {error}"
        )

        print("\n")

        # Keep the Windows console open
        # so the error can be read.
        input(
            "Press ENTER to close..."
        )
