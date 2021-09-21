# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif
import warnings
warnings.filterwarnings('default')

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--dataset1', type=str, required=True)
parser.add_argument('--dataset2', type=str, required=True)
# parser.add_argument('--save_path', type=str, required=True)
parser.add_argument('--epoch1', type=int, required=True)
# parser.add_argument('--num_epochs1', type=int, required=True)
parser.add_argument('--epoch2', type=int, required=True)
parser.add_argument('--num_epochs', type=int, required=True)
parser.add_argument('--train_path1', type=str, required=True)
parser.add_argument('--train_path2', type=str, required=True)
args = parser.parse_args()


if __name__ == '__main__':
    dataset1 = args.dataset1  # 数据集
    dataset2 = args.dataset2  # 数据集
    # save_path = args.save_path
    train_path1 = args.train_path1
    train_path2 = args.train_path2
    # val_path = args.val_path
    model_name = args.model  # bert
    start_epoch = args.epoch1
    mid_epoch = args.epoch2
    end_epoch = args.num_epochs
    # model_name = 'bert'
    x = import_module('models.' + model_name)
    config = x.Config(dataset1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    save_path1 = dataset1 + '/saved_dict/' + config.model_name + '-model-{}.pth'.format(start_epoch)
    save_path2 = dataset1 + '/saved_dict/' + config.model_name + '-model-{}.pth'.format(mid_epoch)

    config.train_path = dataset1 + '/data/' + train_path1   # 训练集
    config.dev_path = dataset2 + '/data/' + train_path2# 测试集
    start_time = time.time()
    print("Loading data...")
    train_data1, train_data2 = build_dataset(config)
    train_iter1 = build_iterator(train_data1, config)
    train_iter2 = build_iterator(train_data2, config)
#     dev_iter = build_iterator(dev_data, config)
#     test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    train(config, model, train_iter1, start_epoch, mid_epoch, save_path1)
    config.checkpoint_path = dataset2 + '/saved_dict/'
    train(config, model, train_iter2, mid_epoch, end_epoch, save_path2)
# # coding: UTF-8
# import time
# import torch
# import numpy as np
# from train_eval import train, init_network
# from importlib import import_module
# import argparse
# from utils import build_dataset, build_iterator, get_time_dif

# parser = argparse.ArgumentParser(description='Chinese Text Classification')
# parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
# args = parser.parse_args()


# if __name__ == '__main__':
#     dataset = 'data'  # 数据集

#     model_name = args.model  # bert
#     # model_name = 'bert'
#     x = import_module('models.' + model_name)
#     config = x.Config(dataset)
#     np.random.seed(1)
#     torch.manual_seed(1)
#     torch.cuda.manual_seed_all(1)
#     torch.backends.cudnn.deterministic = True  # 保证每次结果一样

#     start_time = time.time()
#     print("Loading data...")
#     train_data, dev_data = build_dataset(config)
#     train_iter = build_iterator(train_data, config)
#     dev_iter = build_iterator(dev_data, config)
# #     test_iter = build_iterator(test_data, config)
#     time_dif = get_time_dif(start_time)
#     print("Time usage:", time_dif)

#     # train
#     model = x.Model(config).to(config.device)
#     train(config, model, train_iter, dev_iter)
