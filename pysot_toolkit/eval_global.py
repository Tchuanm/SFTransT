from __future__ import absolute_import
import os
import sys

env_path = os.path.join(os.path.dirname(__file__), '..')
print(env_path)
if env_path not in sys.path:
    sys.path.append(env_path)
import argparse
from got10k.experiments import *
from trackers.got10k_tracker_global import got10k_Tracker
from pysot_toolkit.trackers.net_wrappers import NetWithBackbone
import torch
import shutil

"""
cd tcm/SFTransT/pysot_toolkit
conda activate transt
python eval_got10k_global.py --cuda 0  --begin 95 --end 100 --interval 1 --folds sftranst_train_cfa_smca_attnscale_mlp --subset test
"""

parser = argparse.ArgumentParser(description='got10k tracking test or val subset')
parser.add_argument('--begin', default=98, type=int, help='choose which epoch to begin')
parser.add_argument('--end', default=100, type=int, help='choose which epoch to end')
parser.add_argument('--interval', default=1, type=int, help='interval epoches to test')
parser.add_argument('--cuda', default='2', type=str, help='cuda number')
parser.add_argument('--folds', default='sftranst_train_cfa_smca_attnscale_mlp', type=str, help='epoch folders')
parser.add_argument('--subset', default='test', type=str, help='vel or test')
parser.add_argument('--vis', default=False, type=bool, help='visualization')
parser.add_argument('--win', default='0.5', type=float, help='window penaly')
parser.add_argument('--thr', default='0.98', type=float, help='local global thr')
parser.add_argument('--fail_count', default='10', type=int, help='fail frame count')
args = parser.parse_args()
torch.set_num_threads(1)


# used for test and eval  for:  GOT-10k_eval GOT-10k_test UAV123 TC128  UAV20L


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    for epoch in range(args.begin, args.end + 1, args.interval):
        currrent_path = sys.path[0]
        save_epoch_dir = currrent_path + '/../ltr/checkpoints/ltr/sftranst/' + args.folds
        net_path = os.path.join(save_epoch_dir, 'sftranst_ep%d.pth.tar' % epoch)
        name = '%s_ep%d_win%.2f_global%.2f_count%d' % (args.folds, epoch, args.win, args.thr, args.fail_count)
        net = NetWithBackbone(net_path=net_path)
        tracker = got10k_Tracker(name=name, net=net, instance_size=256, search_factor=4,  # 320 5
                                 temp_factor=2, exemplar_size=128, window_penalty=args.win,
                                 global_thr=args.thr, fail_count=args.fail_count)
        tracker.name = name

        ### for got10k
        root_dir = os.path.expanduser('testing_dataset/GOT10k_got/')
        e = ExperimentGOT10k(root_dir, args.subset, result_dir='results', report_dir='reports')
        e.run(tracker, args.vis)
        if e.subset == 'val':
            performance = e.report([tracker.name])
            ao = performance[name]['overall']['ao']
            print('val valure ao is ', ao)
            with open("./results/eval_got10k.txt", "a") as f:
                f.write(tracker.name + "  AO:{:.3f}".format(ao) + '\n')
        else:
            e.report([tracker.name])
        shutil.rmtree('./results/GOT-10k/' + name)  # delete the fold

        # for uav123/uav20L val;
        root_dir = os.path.expanduser('testing_dataset/UAV/')
        e = ExperimentUAV123(root_dir, version='UAV123', result_dir='results', report_dir='reports')
        e.run(tracker, args.vis)
        performance = e.report([tracker.name])
        ss = performance[name]['overall']['success_score']
        ps = performance[name]['overall']['precision_score']
        fps = performance[name]['overall']['speed_fps']
        print('UAV123 success, precision, speed:', ss, ps, fps)
        with open("./results/eval_UAV.txt", "a") as f:
            f.write(tracker.name + "  success_score:{:.4f}".format(ss) + '\n')

        # # for TC128 val;
        root_dir = os.path.expanduser('testing_dataset/TC128/')
        e = ExperimentTColor128(root_dir, result_dir='results', report_dir='reports')
        e.run(tracker, args.vis)
        performance = e.report([tracker.name])
        ss = performance[name]['overall']['success_score']
        ps = performance[name]['overall']['precision_score']
        fps = performance[name]['overall']['speed_fps']
        with open("./results/eval_TC128.txt", "a") as f:
            f.write(tracker.name + "  success_score:{:.4f}".format(ss) + '\n')
        print('success, precision, speed:', ss, ps, fps)
