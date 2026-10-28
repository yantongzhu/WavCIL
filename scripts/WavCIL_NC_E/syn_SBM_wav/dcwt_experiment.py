import os
from sild.utils import *
import pandas as pd
from libwon import ParallelerGrid


grid_list = [
    {
        'dataset': ['ali-syn21'],
        'backbone': ['GAT'],
        'spec_filt': ['dcwt'],
        'post_temporal': [0],
        'intervene_lambda': [1e-2],
        'intervene_times': [100],
        'learns': [1],
        'heads': [2],
        'norm': [0],
        'use_RTE': [1],
        'spec_RTE': [1],
        'spec_res': [0],
        'temporature': [1],
        'dropout': [0.4],
        'lr': [1e-2],
        'weight_decay': [5e-7],
        'nhid': [8],
        'min_epoch': [0],
        'patience': [100],
        'dcwt_scales': [32],
        'dcwt_omega0': [6.0],
        'dcwt_pool_mode': ['mean'],
        'dcwt_scale_type': ['linear'],
        'seed': range(3)
    },
    {
        'dataset': ['ali-syn22'],
        'backbone': ['GAT'],
        'spec_filt': ['dcwt'],
        'post_temporal': [0],
        'intervene_lambda': [1e-2],
        'intervene_times': [100],
        'learns': [1],
        'heads': [2],
        'norm': [0],
        'use_RTE': [1],
        'spec_RTE': [1],
        'spec_res': [0],
        'temporature': [0.1],
        'dropout': [0.4],
        'lr': [1e-2],
        'weight_decay': [5e-7],
        'nhid': [8],
        'min_epoch': [0],
        'patience': [100],
        'dcwt_scales': [16],
        'dcwt_omega0': [10.0],
        'dcwt_pool_mode': ['mean'],
        'dcwt_scale_type': ['linear'],
        'seed': range(3)
    },
    {
        'dataset': ['ali-syn23'],
        'backbone': ['GAT'],
        'spec_filt': ['dcwt'],
        'post_temporal': [0],
        'intervene_lambda': [1e-2],
        'intervene_times': [100],
        'learns': [1],
        'heads': [2],
        'norm': [0],
        'use_RTE': [1],
        'spec_RTE': [1],
        'spec_res': [0],
        'temporature': [0.45],
        'dropout': [0.4],
        'lr': [1e-2],
        'weight_decay': [5e-7],
        'nhid': [8],
        'min_epoch': [0],
        'patience': [100],
        'dcwt_scales': [16],
        'dcwt_omega0': [10.0],
        'dcwt_pool_mode': ['mean'],
        'dcwt_scale_type': ['linear'],
        'seed': range(3)
    },
]
gpus = [4,5]
cmd = 'python main.py'
readme = ''
phone_notice = ""
base_dir = "../../../sild/"

class Parallel(ParallelerGrid):
    def show(self,c):
        exp_dir=self.exp_dir
        ana_dir=self.ana_dir
        cols = ParallelerGrid.collect_keys(grid_list)
        print(cols)
        import json
        import datetime
        from libwon.utils.collector import COLLECTOR
        files=os.listdir(exp_dir)
        res={}
        for fname in files:
            try:
                folder=os.path.join(exp_dir,fname)
                info=json.load(open(os.path.join(folder,'args.json')))
                hp=tuple([info[k] for k in cols])
                COLLECTOR.load(os.path.join(exp_dir,fname,'collector.json'))
                cache=COLLECTOR.cache
                res[hp]=cache
            except Exception as e:
                print(e)
        if c==0:
            table=[]
            metrics = 'best_test_auc'.split()
            for k,v in res.items():
                table.append([*k]+[v[x][0] for x in metrics])
            df = pd.DataFrame(table,columns=cols+metrics)
            df = df.sort_values(by="best_test_auc".split())
            g = cols.copy()
            del g[g.index('seed')]
            agg_dict = {f"mean_{x}":(x,"mean") for x in metrics}
            agg_dict.update({f"std_{x}":(x,"std") for x in metrics})
            df = df.drop(columns='seed'.split()).groupby(g).agg(**agg_dict).reset_index()
            def format(x):
                x = x * 100 if x<=1 else x
                return f"{x:.2f}"
            for m in metrics:
                df[f"{m}"] = df[f"mean_{m}"].apply(format) +"+-" + df[f"std_{m}"].apply(format)
            df = df.drop(columns = [f"mean_{m}" for m in metrics] + [f"std_{m}" for m in metrics])
            print("dataset spec_filt backbone dcwt_scales dcwt_omega0 temporature dropout dcwt_pool_mode dcwt_scale_type best_test_auc")
            result_df = df[['dataset', 'spec_filt','backbone','dcwt_scales', 'dcwt_omega0','temporature', 'dropout','dcwt_pool_mode','dcwt_scale_type', 'best_test_auc']]
            print(result_df.to_string(index=False))
            result_df.to_csv(os.path.join(self.ana_dir,f"{os.path.basename(__file__)}-{c}.csv"), index=False)


gpu_arg, log_arg = "device_id", "log_dir"
dir, fname = os.path.abspath(__file__).split(os.sep)[-2:]
log_dir = os.path.join('../../../logs/', 'dcwt_experiment_ablation_syn_SBM')

parallel=Parallel(
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
