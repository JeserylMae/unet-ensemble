import numpy as np
from scipy import stats


class Test:
    TESTS = {
        'shapiro'    : lambda diff, alpha: _shapiro(diff, alpha),
        'wilcoxon'   : lambda diff, alpha: _wilcoxon(diff),
        'permutation': lambda diff, alpha: _permutation(diff),
        'kruskal': lambda diff, alpha, config, col_key, df: _kruskal(diff, config, col_key, df),     
    }

    def _shapiro(diff, alpha):
        _, p = stats.shapiro(diff)
        return {'SW p': f"{p:.4e}", 'Normal': p > alpha}

    def _wilcoxon(diff):
        _, p = stats.wilcoxon(diff, alternative='greater')
        return {'W p': f"{p:.6e}", 'Significant W p': p < 0.05}

    def _permutation(diff):
        res = stats.permutation_test(
            (diff,), lambda x: x.mean(),
            permutation_type='samples', alternative='greater', n_resamples=9999
        )
        p = res.pvalue
        return {'Perm p': f"{p:.4e}", 'Significant Perm': p < 0.05}

    def _kruskal(diff, config, col_key, df):
        extra_cols = config.get('kruskal_groups', [])
        extra_groups = [df[col] for col in extra_cols]   # ← direct, not config[k]
        groups = [diff] + extra_groups
        _, p = stats.kruskal(*groups)
        return {'Kruskal p': f"{p:.4e}", 'Significant Kruskal p': p < 0.05}
    
    def run_tests(
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

                if 'kruskal' not in tests:
                    for test in tests:
                        row.update(TESTS[test](diff, alpha))
                else: 
                    for test in tests:
                        row.update(
                            TESTS[test](diff, alpha, config, col_key, df) if test == 'kruskal'
                            else TESTS[test](diff, alpha)
                        )

                results.append(row)


        return pd.DataFrame(results)