from config import get_args
import warnings
from sild.utils import *
import os
import json
from os import path as osp
from pathlib import Path
import sild.utils.data_util as data_util
warnings.simplefilter("ignore")

                
args = get_args()
setup_seed(args.seed)
                                                                           
data_util.DATA_dir = str(Path(__file__).resolve().parents[3] / "data")
args, data = load_synthetic_data(args)
data = move_to(data, args.device)

          
log_dir = args.log_dir
info_dict = get_arg_dict(args)
json.dump(info_dict,open(os.path.join(log_dir,'args.json'),'w'),indent=2)
print(args)

        
from model import SGCNNet, Trainer
model = SGCNNet(args, data).to(args.device)
trainer = Trainer(model, data, args)
results = trainer.train_till_end()

           
measure_dict=results
info_dict.update(measure_dict)
json.dump(info_dict, open(osp.join(log_dir, 'info.json'), 'w'))
COLLECTOR.add_GPU_MEM(args.device,id=False)
COLLECTOR.save_all_time()
COLLECTOR.save(os.path.join(log_dir,'collector.json'))
