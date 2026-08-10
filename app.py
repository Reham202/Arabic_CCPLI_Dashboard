import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
import warnings
warnings.filterwarnings('ignore')

class CCPLIV4OfficialEngine:
    """
    CCPLI V4 Official Framework Engine
    -----------------------------------
    - Dimensions: 10 Strategic Dimensions weighted by AHP.
    - Indicators: 50 Sub-Indicators with w_ij = W_i / 5.
    - Aggregation: Linear (Base Model) & Zero-Safe Geometric (Robustness Model).
    - Uncertainty: Calibrated Dirichlet Monte Carlo Simulation (10,000 runs).
    - Sensitivity: Weight Perturbation (+20% with Renormalization), LOIO, LODO.
    - Rank Stability: Spearman, Kendall's Tau, MARC, and MaxRC.
    - Validity: Content Validity Index (CVI) framework & Convergent Validity tests.
    """

    def __init__(self, raw_ahp_weights=None):
        # 1. أوزان الأبعاد العشرة المعتمدة رسمياً (AHP)
        if raw_ahp_weights is not None:
            self.dimension_weights = raw_ahp_weights
        else:
            self.dimension_weights = {
                'D1_Digital_Algorithmic': 0.12,
                'D2_TAFL_Education': 0.10,
                'D3_Linguistic_Structure': 0.10,
                'D4_Religious_Sacred': 0.10,
                'D5_Civilizational_Historical': 0.08,
                'D6_Knowledge_Scientific_Publishing': 0.10,
                'D7_Economic_Trade_Exposure': 0.12,
                'D8_Demographic_Spread': 0.10,
                'D9_Media_Public_Digital_Presence': 0.08,
                'D10_Geopolitical_Language_Policies': 0.10
            }
        
        # التأكد من إعادة المعايرة لضمان المجموع = 1.0
        total_w = sum(self.dimension_weights.values())
        self.dimension_weights = {k: v / total_w for k, v in self.dimension_weights.items()}

        # 2. حساب أوزان المؤشرات الـ 50 بدقة: w_ij = W_i / 5
        self.indicators = []
        self.indicator_weights = {}
        
        for dim_code, dim_w in self.dimension_weights.items():
            sub_w = dim_w / 5.0  # خُمس وزن البُعد
            for i in range(1, 6):
                ind_code = f"{dim_code}_I{i}"
                self.indicators.append(ind_code)
                self.indicator_weights[ind_code] = sub_w

    # --------------------------------------------------------------------------
    # 1. AHP Consistency Check
    # --------------------------------------------------------------------------
    def calculate_ahp_consistency(self, pairwise_matrix):
        """حساب نسبة الاتساق (CR) مع اعتماد المعيار الإرشادي (0.10)"""
        n = pairwise_matrix.shape[0]
        ri_dict = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
        
        eigenvalues, _ = np.linalg.eig(pairwise_matrix)
        max_eigenvalue = np.max(np.real(eigenvalues))
        
        ci = (max_eigenvalue - n) / (n - 1)
        ri = ri_dict.get(n, 1.49)
        cr = ci / ri if ri > 0 else 0.0
        
        return {
            'Lambda_Max': float(np.real(max_eigenvalue)),
            'CI': float(np.real(ci)),
            'CR': float(np.real(cr)),
            'Guideline_Pass': bool(cr <= 0.10)
        }

    # --------------------------------------------------------------------------
    # 2. Normalization & Aggregation Models
    # --------------------------------------------------------------------------
    def normalize_min_max(self, df, directions=None):
        """التطبيع القياسي Min-Max [0, 100]"""
        df_norm = pd.DataFrame(index=df.index)
        directions = directions or {col: 1 for col in df.columns}
        
        for col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            direction = directions.get(col, 1)
            
            if max_val == min_val:
                df_norm[col] = 50.0
            elif direction == 1:
                df_norm[col] = ((df[col] - min_val) / (max_val - min_val)) * 100.0
            else:
                df_norm[col] = ((max_val - df[col]) / (max_val - min_val)) * 100.0
        return df_norm

    def aggregate_linear(self, df_norm, weights=None):
        """النموذج الأساسي: التجميع الخطي الموزون"""
        w = weights if weights else self.indicator_weights
        return df_norm.dot(pd.Series(w))

    def aggregate_geometric_zero_safe(self, df_norm, weights=None):
        """النموذج البديل: التجميع الهندسي الآمن عند الصفر: LPI_G = exp[ sum(w_j * ln(1 + I_j)) ] - 1"""
        w = weights if weights else self.indicator_weights
        log_term = np.log1p(df_norm)
        weighted_log = log_term.dot(pd.Series(w))
        return np.expm1(weighted_log)

    # --------------------------------------------------------------------------
    # 3. Uncertainty & Sensitivity Suite
    # --------------------------------------------------------------------------
    def run_calibrated_dirichlet_monte_carlo(self, df_norm, num_simulations=10000, precision_k=100):
        """
        محاكاة مونت كارلو (10,000 تكرار) مع معايرة توزيع ديريكليه
        alpha_i = K * W_i (حيث K يحدد مدى المحافظة على بنية AHP)
        """
        base_scores = self.aggregate_linear(df_norm)
        base_ranks = base_scores.rank(ascending=False)
        
        dim_names = list(self.dimension_weights.keys())
        base_dim_w = np.array([self.dimension_weights[d] for d in dim_names])
        alpha_params = base_dim_w * precision_k  # Calibration
        
        sim_scores, sim_ranks = [], []
        np.random.seed(42)
        
        for _ in range(num_simulations):
            sampled_dim_w = np.random.dirichlet(alpha_params)
            
            sampled_ind_w = {}
            for idx, dim_code in enumerate(dim_names):
                sub_w = sampled_dim_w[idx] / 5.0
                for i in range(1, 6):
                    sampled_ind_w[f"{dim_code}_I{i}"] = sub_w
                    
            scores = self.aggregate_linear(df_norm, weights=sampled_ind_w)
            ranks = scores.rank(ascending=False)
            
            sim_scores.append(scores)
            sim_ranks.append(ranks)
            
        scores_df = pd.DataFrame(sim_scores)
        ranks_df = pd.DataFrame(sim_ranks)
        
        # حساب المقاييس
        score_ci_low = scores_df.quantile(0.025, axis=0)
        score_ci_up = scores_df.quantile(0.975, axis=0)
        marc = np.abs(ranks_df - base_ranks).mean(axis=0)
        max_rc = np.abs(ranks_df - base_ranks).max(axis=0)
        
        spearmans = [spearmanr(base_ranks, ranks_df.iloc[i]).statistic for i in range(num_simulations)]
        kendalls = [kendalltau(base_ranks, ranks_df.iloc[i]).statistic for i in range(num_simulations)]
        
        return {
            'Base_Ranks': base_ranks,
            'Score_95_CI_Lower': score_ci_low,
            'Score_95_CI_Upper': score_ci_up,
            'MARC_per_Language': marc,
            'MaxRC_per_Language': max_rc,
            'Avg_Spearman_Rho': np.mean(spearmans),
            'Avg_Kendall_Tau': np.mean(kendalls)
        }

    def weight_perturbation_scenario(self, df_norm, factor=1.20):
        """اضطراب الأوزان (Scenario Analysis ±20%) مع إعادة التطبيع"""
        base_ranks = self.aggregate_linear(df_norm).rank(ascending=False)
        results = {}
        
        for dim_code in self.dimension_weights.keys():
            p_dims = self.dimension_weights.copy()
            p_dims[dim_code] *= factor
            
            # إعادة التطبيع الكلية ليكون المجموع 1.0
            sum_w = sum(p_dims.values())
            norm_dims = {k: v / sum_w for k, v in p_dims.items()}
            
            new_ind_w = {f"{d}_I{i}": norm_dims[d] / 5.0 for d in norm_dims for i in range(1, 6)}
            
            new_scores = self.aggregate_linear(df_norm, weights=new_ind_w)
            new_ranks = new_scores.rank(ascending=False)
            
            rho = spearmanr(base_ranks, new_ranks).statistic
            tau = kendalltau(base_ranks, new_ranks).statistic
            marc = np.abs(base_ranks - new_ranks).mean()
            max_rc = np.abs(base_ranks - new_ranks).max()
            
            results[dim_code] = {'Spearman_Rho': rho, 'Kendall_Tau': tau, 'MARC': marc, 'MaxRC': max_rc}
            
        return pd.DataFrame(results).T

    def leave_one_dimension_out_analysis(self, df_norm):
        """تحليل استبعاد بُعد كامل (Leave-One-Dimension-Out)"""
        base_ranks = self.aggregate_linear(df_norm).rank(ascending=False)
        results = {}
        
        for dim_code in self.dimension_weights.keys():
            rem_inds = [ind for ind in self.indicators if not ind.startswith(dim_code)]
            raw_w = np.array([self.indicator_weights[ind] for ind in rem_inds])
            new_w = raw_w / np.sum(raw_w)
            
            scores_lodo = df_norm[rem_inds].dot(pd.Series(dict(zip(rem_inds, new_w))))
            ranks_lodo = scores_lodo.rank(ascending=False)
            
            rho = spearmanr(base_ranks, ranks_lodo).statistic
            tau = kendalltau(base_ranks, ranks_lodo).statistic
            marc = np.abs(base_ranks - ranks_lodo).mean()
            max_rc = np.abs(base_ranks - ranks_lodo).max()
            
            results[dim_code] = {'Spearman_Rho': rho, 'Kendall_Tau': tau, 'MARC': marc, 'MaxRC': max_rc}
            
        return pd.DataFrame(results).T

    # --------------------------------------------------------------------------
    # 4. Validity Diagnostics (Content & Construct Support)
    # --------------------------------------------------------------------------
    def calculate_cvi_expert_matrix(self, expert_ratings_df):
        """
        حساب Content Validity Index (CVI) بناءً على تقييمات الخبراء للمؤشرات الـ 50
        (4-point scale: 1=Not relevant, 4=Highly relevant -> CVI = ratings >= 3 / total experts)
        """
        relevant_ratings = (expert_ratings_df >= 3).sum(axis=1)
        total_experts = expert_ratings_df.shape[1]
        i_cvi = relevant_ratings / total_experts
        s_cvi_ave = i_cvi.mean()
        
        return {
            'Item_CVI_per_Indicator': i_cvi,
            'Overall_S_CVI_Average': s_cvi_ave,
            'Pass_Validation': bool(s_cvi_ave >= 0.80)
        }


# ==============================================================================
# اختبار بروتوكول التشغيل الكامل
# ==============================================================================
if __name__ == "__main__":
    engine = CCPLIV4OfficialEngine()
    
    print("======================================================================")
    print("   CCPLI V4 Production Engine - Final Statistical Protocol")
    print("======================================================================")
    
    # طباعة توزيع أوزان المؤشرات الفرعية الحقيقي
    print("\n--- 1. الأوزان الدقيقة للمؤشرات الفرعية (w_ij = W_i / 5) ---")
    sub_w_summary = pd.DataFrame({
        'Dimension_Weight': engine.dimension_weights,
        'Sub_Indicator_Weight (w_ij)': {d: w/5.0 for d, w in engine.dimension_weights.items()}
    })
    print(sub_w_summary)
    
    # إنشاء بيانات محاكاة لـ 6 لغات
    languages = ['Arabic', 'English', 'French', 'Chinese', 'Spanish', 'German']
    np.random.seed(42)
    raw_data = np.random.uniform(10, 90, size=(len(languages), 50))
    df_raw = pd.DataFrame(raw_data, index=languages, columns=engine.indicators)
    df_norm = engine.normalize_min_max(df_raw)
    
    # التجميع الخطي vs الهندسي الآمن
    lin_scores = engine.aggregate_linear(df_norm)
    geom_scores = engine.aggregate_geometric_zero_safe(df_norm)
    
    comp_df = pd.DataFrame({
        'Linear_Score': lin_scores,
        'Linear_Rank': lin_scores.rank(ascending=False),
        'Geom_Score': geom_scores,
        'Geom_Rank': geom_scores.rank(ascending=False)
    })
    print("\n--- 2. النموذج الأساسي (الخطي) vs التجميع الهندسي البديل ---")
    print(comp_df.round(2))
    
    # محاكاة مونت كارلو المعايرة
    print("\n--- 3. محاكاة مونت كارلو المعايرة (10,000 Runs) ---")
    mc_res = engine.run_calibrated_dirichlet_monte_carlo(df_norm, num_simulations=10000, precision_k=100)
    print(f"متوسط Spearman Rho عبر المحاكاة: {mc_res['Avg_Spearman_Rho']:.4f}")
    print(f"متوسط Kendall Tau عبر المحاكاة:  {mc_res['Avg_Kendall_Tau']:.4f}")
    
    mc_table = pd.DataFrame({
        'Base_Rank': mc_res['Base_Ranks'],
        'Score_95%_CI_Low': mc_res['Score_95_CI_Lower'],
        'Score_95%_CI_Up': mc_res['Score_95_CI_Upper'],
        'MARC': mc_res['MARC_per_Language'],
        'MaxRC': mc_res['MaxRC_per_Language']
    })
    print(mc_table.round(2))
