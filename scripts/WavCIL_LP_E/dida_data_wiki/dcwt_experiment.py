import os
from sild.utils import *
import pandas as pd
from libwon import ParallelerGrid

grid_list = [
    {
        'dataset': ['wiki'],
        'backbone': ['dida'],
        'spec_filt': ['dcwt'],
        'post_temporal': [1],
        'intervene_lambda': [1e-4],
        'intervene_times': [100],
        'min_epoch': [200],
        'heads': [2],
        'norm': [1],
        'temporature': [0.45],
        'dropout': [0.35],
        'nfeat': [64],
        'dcwt_scales': [8],
        'dcwt_omega0': [6.9],
        'seed': range(3)
    },

]

gpus = [4,5]


cmd = 'python main.py'
readme = 'WavCIL (LP): DCWT-based wavelet coefficient-domain invariant learning.'
phone_notice = ""
base_dir = "../../../sild/"


class DCWTParallel(ParallelerGrid):
    def show(self, c):
        exp_dir = self.exp_dir
        ana_dir = self.ana_dir
        cols = ParallelerGrid.collect_keys(grid_list)
        print(cols)
        import json
        import datetime
        from libwon.utils.collector import COLLECTOR
        files = os.listdir(exp_dir)
        res = {}
        for fname in files:
            try:
                folder = os.path.join(exp_dir, fname)
                info = json.load(open(os.path.join(folder, 'args.json')))
                hp = tuple([info[k] for k in cols])

                COLLECTOR.load(os.path.join(exp_dir, fname, 'collector.json'))
                cache = COLLECTOR.cache
                res[hp] = cache
            except Exception as e:
                print(e)

        if c == 0:
            table = []
            metrics = 'best_test_test_auc'.split()
            for k, v in res.items():
                table.append([*k] + [v[x][0] for x in metrics])
            df = pd.DataFrame(table, columns=cols + metrics)
            df = df.sort_values(by="best_test_test_auc".split())
            g = cols.copy()
            del g[g.index('seed')]
            agg_dict = {f"mean_{x}": (x, "mean") for x in metrics}
            agg_dict.update({f"std_{x}": (x, "std") for x in metrics})

            df = df.drop(columns='seed'.split()).groupby(g).agg(**agg_dict).reset_index()

            def format(x):
                x = x * 100 if x <= 1 else x
                return f"{x:.2f}"

            for m in metrics:
                df[f"{m}"] = df[f"mean_{m}"].apply(format) + "+-" + df[f"std_{m}"].apply(format)
            df = df.drop(columns=[f"mean_{m}" for m in metrics] + [f"std_{m}" for m in metrics])

            print("\nDCWT experiment results:")
            print("dataset spec_filt backbone dcwt_scales dcwt_omega0 temporature dropout post_temporal best_test_test_auc")
            result_df = df[['dataset', 'spec_filt','backbone','dcwt_scales', 'dcwt_omega0', 'temporature', 'dropout', 'post_temporal', 'best_test_test_auc']]
            print(result_df.to_string(index=False))

            result_df.to_csv(os.path.join(self.ana_dir, f"dcwt_experiment-{c}.csv"), index=False)

            summary_df = df.groupby('dataset')['best_test_test_auc'].first().reset_index()
            print("dataset best_test_test_auc")
            print(summary_df.to_string(index=False))
            summary_df.to_csv(os.path.join(self.ana_dir, f"dcwt_summary-{c}.csv"), index=False)


gpu_arg, log_arg = "device_id", "log_dir"
dir, fname = os.path.abspath(__file__).split(os.sep)[-2:]
log_dir = os.path.join('../../../logs/', 'dcwt_experiment_ablation_wiki')

parallel = DCWTParallel(
    gpus=gpus,
    log_dir=log_dir,
    grid_list=grid_list,
    cmd=cmd,
    gpu_arg=gpu_arg,
    log_arg=log_arg,
    epoch_arg="max_epoch",
    exp_script_dir=os.path.dirname(os.path.abspath(__file__)),
    base_script_dir=base_dir,
    readme=readme,
    phone_notice=phone_notice
).execute()
