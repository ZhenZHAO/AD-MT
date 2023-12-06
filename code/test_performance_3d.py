import os
import argparse
import torch

from networks.net_factory import net_factory
from val_3D import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./data/LA/', help='Name of Experiment')
parser.add_argument('--dataset', type=str, default='LA', help='Name of dataset')
parser.add_argument('--res_path', type=str, default='./results', help='Path to results')
parser.add_argument('--exp', type=str, default='LA/POST', help='experiment_name')
parser.add_argument('--model', type=str, default='vnet', help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--detail', type=int,  default=1, help='print metrics for every samples?')
parser.add_argument("--flag_check_with_best_stu", default=False, action='store_true', help="")
parser.add_argument('--nms', type=int, default=1, help='apply NMS post-procssing?')
parser.add_argument('--labeled_num', type=int, default=4, help='labeled data')

FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "{}/{}_{}_labeled/{}".format(FLAGS.res_path, FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
# test_save_path = "{}/{}_{}_labeled/{}_predictions/".format(FLAGS.res_path, FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
if FLAGS.flag_check_with_best_stu:
    test_save_path = "{}/{}_{}_labeled/{}_predictions_stu/".format(FLAGS.res_path, FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
else:
    test_save_path = "{}/{}_{}_labeled/{}_predictions_tea/".format(FLAGS.res_path, FLAGS.exp, FLAGS.labeled_num, FLAGS.model)

num_classes = 2

if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

with open(FLAGS.root_path + '/test.list', 'r') as f:
    image_list = f.readlines()
# image_list = [FLAGS.root_path + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]

flag_pancreas = 'pancreas' in FLAGS.dataset.lower()
if flag_pancreas:
    image_list = [item.replace('\n', '') for item in image_list]
    image_list = [os.path.join(FLAGS.root_path, "data", f"{item}_norm.h5") for item in image_list]
else:
    image_list = [FLAGS.root_path + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]


def test_calculate_metric():
    model = net_factory(net_type=FLAGS.model, in_chns=1, class_num=num_classes)
    
    if FLAGS.flag_check_with_best_stu:
        save_model_path = os.path.join(snapshot_path, '{}_best_stu_model.pth'.format(FLAGS.model))
    else:
        save_model_path = os.path.join(snapshot_path, '{}_best_tea_model.pth'.format(FLAGS.model))

    model.load_state_dict(torch.load(save_model_path))
    print("init weight from {}".format(save_model_path))
    model.eval()

    if not flag_pancreas:
        avg_metric = test_all_case(model, image_list, num_classes=num_classes,
                            patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                            save_result=True, test_save_path=test_save_path,
                            metric_detail=FLAGS.detail, nms=FLAGS.nms)
    else:
        avg_metric = test_all_case(model, image_list, num_classes=num_classes,
                           patch_size=(96, 96, 96), stride_xy=16, stride_z=16,
                           save_result=True, test_save_path=test_save_path,
                           metric_detail=FLAGS.detail, nms=FLAGS.nms)

    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric()
    print(metric)

