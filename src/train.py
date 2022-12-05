import os

from utils import load_model, get_obj_from_str, load_dataset

from sklearn.cluster import KMeans, SpectralClustering
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm


def train_model(args):
    # load the model
    model = load_model(args)
    if "pretrained_path" in args["model"]:
        model.load_state_dict(args["model"]['pretrained_path'])

    # load the dataset
    train_dataset = load_dataset(args["dataset"]["train"])
    test_dataset = load_dataset(args["dataset"]["test"])

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args['dataset']['train']['batch_size'],
                                                   shuffle=args['dataset']['train']['shuffle'],
                                                   num_workers=args['dataset']['train']['num_workers'])
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args['dataset']['test']['batch_size'],
                                                  shuffle=args['dataset']['train']['shuffle'],
                                                  num_workers=args['dataset']['train']['num_workers'])

    # train loop
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['optimizer']['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=125, gamma=0.6)

    # init stuff before loop
    cs_loss = Cluster_loss()
    save_loss, save_similarity_loss, save_gt = [], [], []
    model.to(args['device'])
    best_acc = 0

    # other hyperparameter
    T = 2  # T = 2 ===> do sc each epoch
    STOP_CLUSTERING = 200
    global center_num
    center_num = 16
    lam = 1

    for epoch in range(args["dataset"]["train"]["epochs"] + 1):
        if epoch % T == 0 and epoch < STOP_CLUSTERING:
            with torch.no_grad():
                Ground_true, loss_mask_num, loss_mask_den = offline_spectral_cluster(model, train_dataloader)
            save_gt.append(Ground_true.cpu().numpy())
            continue

        scheduler.step()
        model.train()
        total_loss = 0.0
        similarity_loss = 0.0

        for batch_step, input_data in tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader), smoothing=0.9):
            inputs, labels = input_data
            inputs, labels = inputs.to(args["device"]), labels.long().to(args["device"])

            optimizer.zero_grad()
            output, f_map, corre = model(inputs, eval=False)

            clr_loss = .0
            for attribution in range(args['model']['num_classes'] // 2):
                clr_loss += criterion(output[:, 2 * attribution:2 * attribution + 2], labels[:, attribution])
            labels = None

            loss_ = cs_loss.update(corre, loss_mask_num, loss_mask_den, labels)
            loss = clr_loss + lam * loss_
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            similarity_loss += loss_.item()

        # save epoch losses
        total_loss = float(total_loss) / len(train_dataloader)
        similarity_loss = float(similarity_loss) / len(train_dataloader)
        save_loss.append(total_loss)
        save_similarity_loss.append(similarity_loss)
        # acc = 0
        acc = test_acc(model, test_dataloader, args)
        print('Epoch', epoch, 'loss: %.4f' % total_loss, 'cs_loss: %.4f' % similarity_loss, 'test accuracy:%.4f' % acc)

        if epoch % args["dataset"]["train"]["save_every"] == 0:
            torch.save(model.state_dict(),
                       os.path.join(args['dataset']['train']['weight_save_path'], 'model_%.3d.pth' % epoch))
    torch.save(model.state_dict(), os.path.join(args['dataset']['train']['weight_save_path'], 'model_last.pth'))
    return model


####################################################################################
################################ Helper functions ##################################
####################################################################################

def offline_spectral_cluster(net, train_data):
    net.eval()
    f_map = []
    for inputs, labels in train_data:
        inputs, labels = inputs.cuda(), labels.cuda()

        cur_fmap = net(inputs, eval=True).detach().cpu().numpy()
        f_map.append(cur_fmap)
    # for map in f_map:
    #     print(map.shape)
    f_map = np.concatenate(f_map, axis=0)
    sample, channel, _, _ = f_map.shape
    f_map = f_map.reshape((sample, channel, -1))
    mean = np.mean(f_map, axis=0)
    cov = np.mean(np.matmul(f_map - mean, np.transpose(f_map - mean, (0, 2, 1))), axis=0)
    diag = np.diag(cov).reshape(channel, -1)
    correlation = cov / (np.sqrt(np.matmul(diag, np.transpose(diag, (1, 0)))) + 1e-5) + 1
    ground_true, loss_mask_num, loss_mask_den = spectral_clustering(correlation, n_cluster=center_num)

    return ground_true, loss_mask_num, loss_mask_den


def spectral_clustering(similarity_matrix, n_cluster=8):
    W = similarity_matrix

    sz = W.shape[0]
    sp = SpectralClustering(n_clusters=n_cluster, affinity='precomputed', random_state=21)
    y_pred = sp.fit_predict(W)
    # for i in range(n_cluster):
    #     print(np.sum(y_pred==i))
    del W
    ground_true_matrix = np.zeros((sz, sz))
    loss_mask_num = []
    loss_mask_den = []
    for i in range(n_cluster):
        idx = np.where(y_pred == i)[0]
        cur_mask_num = np.zeros((sz, sz))
        cur_mask_den = np.zeros((sz, sz))
        for j in idx:
            ground_true_matrix[j][idx] = 1
            cur_mask_num[j][idx] = 1
            cur_mask_den[j][:] = 1
        loss_mask_num.append(np.expand_dims(cur_mask_num, 0))
        loss_mask_den.append(np.expand_dims(cur_mask_den, 0))
    loss_mask_num = np.concatenate(loss_mask_num, axis=0)
    loss_mask_den = np.concatenate(loss_mask_den, axis=0)
    return torch.from_numpy(ground_true_matrix).float().cuda(), torch.from_numpy(
        loss_mask_num).float().cuda(), torch.from_numpy(loss_mask_den).float().cuda()


class Cluster_loss():
    def __init__(self):
        pass

    def update(self, correlation, loss_mask_num, loss_mask_den, labels):
        batch, channel, _ = correlation.shape
        c, _, _ = loss_mask_num.shape
        if labels is not None:
            label_mask = (1 - labels).view(batch, 1, 1)
            ## smg_loss if only available for positive sample
            correlation = correlation * label_mask
        correlation = (correlation / batch).view(1, batch, channel, channel).repeat(c, 1, 1, 1)

        new_Num = torch.sum(correlation * loss_mask_num.view(c, 1, channel, channel).repeat(1, batch, 1, 1),
                            dim=(1, 2, 3))
        new_Den = torch.sum(correlation * (loss_mask_den).view(c, 1, channel, channel).repeat(1, batch, 1, 1),
                            dim=(1, 2, 3))
        ret_loss = -torch.sum(new_Num / (new_Den + 1e-5))
        return ret_loss


def mse(model, loader, criterion, args):
    model.eval()

    total_error = 0

    for batch_step, input_data in tqdm(enumerate(loader, 0), total=len(loader), smoothing=0.9):
        inputs, labels = input_data
        inputs, labels = inputs.to(args['device']), labels.long().to(args['device'])
        y = model.predict(inputs)

        clr_loss = .0
        for attribution in range(args['model']['num_classes'] // 2):
            clr_loss += criterion(y[:, 2 * attribution:2 * attribution + 2], labels[:, attribution])

        total_error += clr_loss
    return total_error / len(loader)


def test_acc(model, loader, args):
    model.eval()

    correct = 0
    total = 0

    for batch_step, input_data in tqdm(enumerate(loader, 0), total=len(loader), smoothing=0.9):
        inputs, labels = input_data
        inputs, labels = inputs.to(args['device']), labels.long().to(args['device'])

        outputs = model(inputs, eval=True)
        x = model.layer4(outputs)
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        outputs = model.fc(x)
        out = outputs.data


        for i in range(out.shape[0]):
            if labels[i] == 1 and out[i] >= 0.5:
                correct += 1
            elif labels[i] == 0 and out[i] <= 0.5:
                correct += 1

    model.train()

    return correct / len(loader)
