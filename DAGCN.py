import os
import shutil
from time import time
from datetime import datetime
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

from data.lib.utils import compute_val_loss, evaluate, predict
from data.lib.preprocess import read_and_generate_dataset
from data.lib.utils import scaled_Laplacian, get_adjacency_matrix

from model.evolution_KL_ST import setup_features_tuple, setup_Adj_matrix
from model.core import ActivateGCN

from model.optimize import Lookahead

np.seterr(divide='ignore', invalid='ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--max_epoch', type=int, default=70, help='Epoch to run [default: 40]')
parser.add_argument('--batch_size', type=int, default=18, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='Initial learning rate [default: 0.0005]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adamW', help='adam or momentum [default: adam]')
parser.add_argument('--length', type=int, default=24, help='Size of temporal : 12')
parser.add_argument("--force", type=str, default=True, help="remove params dir")
parser.add_argument("--data_name", type=str, default=4, help="the number of data documents [8/4]", required=False)
parser.add_argument('--num_point', type=int, default=307, help='road Point Number [170/307] ', required=False)
parser.add_argument('--seed', type=int, default=31150, help='', required=False)
parser.add_argument('--decay', type=float, default=0.99, help='decay rate of learning rate [0.97/0.92]')

FLAGS = parser.parse_args()
decay = FLAGS.decay
dataname = FLAGS.data_name
adj_filename = 'data/PEMS0%s/distance.csv' % dataname
graph_signal_matrix_filename = 'data/PEMS0%s/pems0%s.npz' % (dataname, dataname)
Length = FLAGS.length
num_nodes = FLAGS.num_point
epochs = FLAGS.max_epoch
optimizer = FLAGS.optimizer
num_of_vertices = FLAGS.num_point
seed = FLAGS.seed

num_of_features = 3
points_per_hour = 12
num_for_predict = 12
num_of_weeks = 2
num_of_days = 1
num_of_hours = 2

merge = False
model_name = 'Karl_ActiveGCN_cnt_params_%s' % dataname
params_dir = 'Karl_ActiveGCN_cnt_params'
prediction_path = 'Karl_ActiveGCN_cnt_params_0%s' % dataname
device = torch.device(FLAGS.device)
wdecay = 0.001

theta, gamma = 0.00001, 0.  # adj_value[adj_value < theta] = gamma
learning_rate = 0.001

batch_size = FLAGS.batch_size
mt_mem_adj_value = 0.000001
lt_mem_adj_value = 0.000001
eq_mem_adj_value = 0.0001
is_axis_mean_max_norm = True
scd = -1

data_file = '4'
method = 'KL'
load_matrix = False
KMD = 0.000001
add_A_and_Diag = False
mat_A_and_Diag = False

AMFile = f'AM_D8_Conv_Harry_Karl_norm.npy'
writedown = f'DAGCN_%s_%s.txt' % (dataname, datetime.now(), )

adj = get_adjacency_matrix(adj_filename, num_nodes)
Vout, Vin = np.sum(adj, axis=0), np.sum(adj, axis=1)  # Calculate:  In Degree, Out Degree
Diag = torch.diag_embed(torch.Tensor((Vout + Vin) / 2))  # Set the Diag matrix
adjs = scaled_Laplacian(adj)
supports = (torch.tensor(adjs)).type(torch.float32).to(device)

print("mat_A_and_Diag : ", mat_A_and_Diag)
print("batch_size : ", batch_size)
print("mt_mem_adj_value : ", mt_mem_adj_value)
print("lt_mem_adj_value : ", lt_mem_adj_value)
print("eq_mem_adj_value : ", eq_mem_adj_value)
print("Symmetric Correlation Degree : ", scd)
print("is_axis_mean_max_norm : ", is_axis_mean_max_norm)
print("Learning rate : ", learning_rate)

print('Model is %s' % (model_name,))
timestamp_s = datetime.now()
print("\nWorking start at ", timestamp_s, '\n')

if params_dir != "None":
    params_path = os.path.join(params_dir, model_name)
else:
    params_path = 'params/%s_%s/' % (model_name, timestamp_s)

if os.path.exists(params_path) and not FLAGS.force:
    raise SystemExit("Params folder exists! Select a new params path please!")
else:
    if os.path.exists(params_path):
        shutil.rmtree(params_path)
    os.makedirs(params_path)
    print('Create params directory %s, reading data...' % (params_path,))


def generate_all_data(batch_size_):
    all_data = read_and_generate_dataset(graph_signal_matrix_filename,
                                         num_of_weeks,
                                         num_of_days,
                                         num_of_hours,
                                         num_for_predict,
                                         points_per_hour,
                                         merge)

    # test set ground truth
    true_value = all_data['test']['target']
    print(true_value.shape)

    # training set data loader
    train_loader = DataLoader(
        TensorDataset(
            torch.Tensor(all_data['train']['week']),
            torch.Tensor(all_data['train']['day']),
            torch.Tensor(all_data['train']['recent']),
            torch.Tensor(all_data['train']['target'])
        ),
        batch_size=batch_size_,
        shuffle=True
    )

    # validation set data loader
    val_loader = DataLoader(
        TensorDataset(
            torch.Tensor(all_data['val']['week']),
            torch.Tensor(all_data['val']['day']),
            torch.Tensor(all_data['val']['recent']),
            torch.Tensor(all_data['val']['target'])
        ),
        batch_size=batch_size_,
        shuffle=False
    )

    # testing set data loader
    test_loader = DataLoader(
        TensorDataset(
            torch.Tensor(all_data['test']['week']),
            torch.Tensor(all_data['test']['day']),
            torch.Tensor(all_data['test']['recent']),
            torch.Tensor(all_data['test']['target'])
        ),
        batch_size=batch_size_,
        shuffle=False
    )
    return all_data, true_value, train_loader, val_loader, test_loader


if __name__ == "__main__":
    # read all data from graph signal matrix file. Input: train / valid  / test : length x 3 x NUM_POINT x 12

    all_data, true_value, train_loader, val_loader, test_loader = generate_all_data(batch_size)

    # save Z-score mean and std
    stats_data = {}
    for type_ in ['week', 'day', 'recent']:
        stats = all_data['stats'][type_]
        stats_data[type_ + '_mean'] = stats['mean']
        stats_data[type_ + '_std'] = stats['std']
    np.savez_compressed(
        os.path.join(params_path, 'stats_data'),
        **stats_data
    )

    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    """ Loading Data Above """

    loss_function = nn.MSELoss()
    net = ActivateGCN(c_in=1, c_out=64, num_nodes=num_nodes, recent=24, K=2, Kt=3)
    net.to(device)
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=wdecay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, decay)
    optimizer = Lookahead(optimizer=optimizer)

    print("\n\n")

    his_loss = []
    validation_loss_lst = []
    train_time = []
    A0_ = np.zeros((num_nodes, num_nodes))
    # A = adjs

    A_lst = []

    if not load_matrix:
        print("Constructing Global A-matrix...  By KL method. for Data4-307p. ")
        _, _, train_loader_local, _, _ = generate_all_data(16)

        for train_w, train_d, train_r, train_t in tqdm(train_loader_local, ncols=80, smoothing=0.9):
            nodes_features_all = setup_features_tuple(train_r)
            A = setup_Adj_matrix(nodes_features_all, num_nodes)
            A_lst.append(A)


        save_A_matrix_filename = AMFile
        A_lst = np.array(A_lst)
        np.save(save_A_matrix_filename, A_lst)

        print("Saved.", save_A_matrix_filename)
        # exit(0)
    else:
        print("Loading Adjacency matrix...  ", './AM_D4_Conv_Harry_Karl_norm.npy')

        A = np.load('./AM_D4_Conv_Harry_Karl_norm.npy')
        A[np.isnan(A)] = 0.
        A[np.isinf(A)] = 0.
        A = A.mean(0) * KMD
        # if mat_A_and_Diag:
        #     A = torch.matmul(torch.Tensor(A), torch.Tensor(Diag))
        #
        # if add_A_and_Diag:
        #     A = torch.Tensor(A) + torch.Tensor(Diag) * 1e-14
        #     print(A)
    A = A_lst
    A[np.isnan(A)] = 0.
    A[np.isinf(A)] = 0.
    A = A_lst.mean(0) * KMD
    supports = (torch.tensor(A)).type(torch.float32).to(device)

    with open(writedown, mode='a', encoding='utf-8') as f:
        f.write(f"seed,epoch,train_loss,valid_loss,learning_rate,_MAE,_MAPE,_RMSE,datetime\n")

    print("ActiveGCN have {} paramerters in total.".format(sum(x.numel() for x in net.parameters())))

    # exit(7)

    watch = True
    for epoch in range(1, epochs + 1):
        train_loss = []
        start_time_train = time()

        if not watch:
            break
        for train_w, train_d, train_r, train_t in tqdm(train_loader, ncols=80, smoothing=0.9):

            train_r = train_r.to(device)
            train_t = train_t.to(device)
            net.train()
            optimizer.zero_grad()

            output, _, A1 = net(train_r, supports)
            loss = loss_function(output, train_t)
            loss.backward()
            optimizer.step()
            training_loss = loss.item()
            train_loss.append(training_loss)

        scheduler.step()
        end_time_train = time()
        train_loss = np.mean(train_loss)
        print('Epoch step: %s, t-loss: %.4f, time: %.2fs' % (epoch, train_loss, end_time_train - start_time_train))

        train_time.append(end_time_train - start_time_train)

        valid_loss = compute_val_loss(net, val_loader, loss_function, supports, device, epoch)

        his_loss.append(valid_loss)

        _MAE, _RMSE, _MAPE = evaluate(net, test_loader, true_value, supports, device, epoch_=epoch)

        with open(writedown, mode='a', encoding='utf-8') as f:
            f.write(
                f"{seed},{epoch},{train_loss},{valid_loss},{scheduler.get_last_lr()[0]},{_MAE},{_MAPE},{_RMSE},{datetime.now()}\n")

        params_filename = os.path.join(params_path,
                                       '%s_epoch_%s_%s.params' % (model_name, epoch, str(round(valid_loss, 2))))
        torch.save(net.state_dict(), params_filename)
        # print('save parameters to file: %s' % (params_filename, ))

        validation_loss_lst.append(float(valid_loss))
        watch_early_stop = np.array(validation_loss_lst)
        arg = np.argmin(watch_early_stop)
        print(
            f"\t >>> Lowest v-loss in {epoch} :  epoch_{arg + 1}  {validation_loss_lst[arg]}  lr = {scheduler.get_last_lr()}\n\n")
        if validation_loss_lst[arg] < 710.0 and learning_rate == 0.001:
            learning_rate = 0.0001
            optimizer = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=wdecay)
            print("Optim changed. ")

    print("\n\nTraining finished.")
    print("Training time/epoch: %.4f secs/epoch" % np.mean(train_time))

    bestId = np.argmin(his_loss)
    print("The valid loss on best model is epoch%s, value is %s" % (str(bestId + 1), str(round(his_loss[bestId], 4))))
    best_params_filename = os.path.join(params_path, '%s_epoch_%s_%s.params' % (
    model_name, str(bestId + 1), str(round(his_loss[bestId], 2))))
    net.load_state_dict(torch.load(best_params_filename))
    start_time_test = time()
    prediction, spatial_at, parameter_adj = predict(net, test_loader, supports, device)
    end_time_test = time()

    evaluate(net, test_loader, true_value, supports, device, epoch)
    test_time = (end_time_test - start_time_test)

    print("Test time: %.2f" % test_time)
    print("Total time: %f s" % (datetime.now() - timestamp_s).seconds)
