import numpy as np
import pandas as pd
from scipy import stats


class Test:
    def _run_test(self, test, diff, alpha, config, col_key, df):
        if test == 'shapiro'    : return self._shapiro(diff, alpha)
        if test == 'wilcoxon'   : return self._wilcoxon(diff)
        if test == 'permutation': return self._permutation(diff)
        if test == 'kruskal'    : return self._kruskal(diff, config, col_key, df)
        if test == 'ci'         : return self._bootstrap_ci(diff, alpha)
        raise ValueError(f"Unknown test: '{test}'")

    def _shapiro(self, diff, alpha):
        _, p = stats.shapiro(diff)
        return {'SW p': f"{p:.4e}", 'Normal': p > alpha}

    def _wilcoxon(self, diff):
        _, p = stats.wilcoxon(diff, alternative='greater')
        return {'W p': f"{p:.6e}", 'Significant W p': p < 0.05}

    def _permutation(self, diff):
        res = stats.permutation_test(
            (diff,), lambda x: x.mean(),
            permutation_type='samples', alternative='greater', n_resamples=9999
        )
        p = res.pvalue
        return {'Perm p': f"{p:.4e}", 'Significant Perm': p < 0.05}

    def _kruskal(self, diff, config, col_key, df):
        extra_cols = config.get('kruskal_groups', {}).get(col_key, [])
        extra_groups = [df[col] for col in extra_cols]
        groups = [diff] + extra_groups
        _, p = stats.kruskal(*groups)
        return {'Kruskal p': f"{p:.4e}", 'Significant Kruskal p': p < 0.05}
    
    def _bootstrap_ci(self, diff, alpha, n_resamples=9999):
        confidence_level = 1 - alpha
        res = stats.bootstrap(
            (diff,),
            statistic=lambda x: x.mean(),
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            method='percentile',
        )
        lo, hi = res.confidence_interval
        return {
            'CI Lower'       : round(lo, 6),
            'CI Upper'       : round(hi, 6),
            'CI Level'       : f"{confidence_level * 100:.1f}%",
            'Favors Proposed': bool(lo > 0),
        }

    def run_tests(
        self,
        groups     : dict,                    # {'GroupName': {'df': df, ...col keys}}
        metrics    : list[tuple[str, str]],   # [('Display Name', 'col_key'), ...]
        tests      : list[str],               # ['shapiro', 'wilcoxon', 't-test', 'permutation']
        group_col  : str  = 'Group',          # column header for the group name
        alpha      : float = 0.05,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        groups    : the BASELINES or MODELS dict — values must have a 'df' key
                    plus whatever column keys are referenced in `metrics`
        metrics   : list of (display_name, config_key) pairs, where config_key
                    is the key inside each group's config that holds the diff column name
        tests     : which statistical tests to run, from: 'shapiro', 'wilcoxon',
                    'kruskal', 'permutation'
        group_col : the label for the grouping column in the output DataFrame
        alpha     : significance threshold
        """
        results = []

        for group_name, config in groups.items():
            df = config['df']
            for display_name, col_key in metrics:
                diff = df[config[col_key]]
                
                row = {
                    'Metric'    : display_name,
                    group_col   : group_name,
                    'Mean Diff' : round(diff.mean(), 6),
                }

                for test in tests:
                    row.update(self._run_test(test, diff, alpha, config, col_key, df))

                results.append(row)


        return pd.DataFrame(results)