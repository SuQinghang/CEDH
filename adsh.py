import torch
import torch.optim as optim
import os
import time
import models.alexnet as alexnet
import utils.evaluate as evaluate
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR


from loguru import logger
from models.adsh_loss import ADSH_Loss
from data.data_loader import sample_dataloader


def train(
        query_dataloader,
        train_dataloader,
        retrieval_dataloader,
        code_length,
        device,
        lr,
        max_iter,
        max_epoch,
        num_samples, # hyperparameter: q
        batch_size,
        root,
        dataset,
        gamma, # hyperparameter: γ
        topk,
        eval_epoch,
        save_name,
):
    """
    Training model.

    Args
        query_dataloader, retrieval_dataloader(torch.utils.data.dataloader.DataLoader): Data loader.
        code_length(int): Hashing code length.
        device(torch.device): GPU or CPU.
        lr(float): Learning rate.
        max_iter(int): Number of iterations.
        max_epoch(int): Number of epochs.
        num_train(int): Number of sampling training data points.
        batch_size(int): Batch size.
        root(str): Path of dataset.
        dataset(str): Dataset name.
        gamma(float): Hyper-parameters.
        topk(int): Topk k map.

    Returns
        mAP(float): Mean Average Precision.
    """
    # Initialization
    model = alexnet.load_model(code_length).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5,
    )
    scheduler = CosineAnnealingLR(optimizer, max_iter, 1e-7)

    num_dataset = len(train_dataloader.dataset)
    U = torch.zeros(num_samples, code_length).to(device)
    B = torch.randn(num_dataset, code_length).to(device)
    train_targets = train_dataloader.dataset.get_onehot_targets().to(device)
    start = time.time()

    timestr = time.strftime('%Y-%m-%d-%H:%M', time.gmtime())
    savedir = os.path.join('checkpoints', 'adsh', dataset, save_name+'-'+timestr)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    best_mAP = 0.0
    for it in range(max_iter):

        if it % eval_epoch == 0 or it == max_iter - 1:
            query_code = generate_code(model, query_dataloader, code_length, device)
            retrieval_code = generate_code(model, retrieval_dataloader, code_length, device)
            mAP = evaluate.mean_average_precision(
                query_code.to(device),
                retrieval_code.to(device),
                query_dataloader.dataset.get_onehot_targets().to(device),
                retrieval_dataloader.dataset.get_onehot_targets().to(device),
                device,
                topk,
            )
            #!save best
            if mAP >= best_mAP:
                best_mAP = mAP
                training_code = generate_code(model, train_dataloader, code_length, device)
                torch.save(training_code.cpu(), os.path.join(savedir, 'training_code{}.t'.format(code_length)))
                torch.save(query_code.cpu(), os.path.join(savedir, 'query_code{}.t'.format(code_length)))
                query_targets = query_dataloader.dataset.get_onehot_targets()
                torch.save(query_targets, os.path.join(savedir, 'query_targets{}.t'.format(code_length)))
                torch.save(retrieval_code.cpu(), os.path.join(savedir, 'retrieval_code{}.t'.format(code_length)))
                retrieval_targets = retrieval_dataloader.dataset.get_onehot_targets()
                torch.save(retrieval_targets, os.path.join(savedir, 'retrieval_targets{}.t'.format(code_length)))
                torch.save(model, os.path.join(savedir, 'model-{}.t'.format(code_length)))
            logger.info('[iter:{}/{}][mAP:{:4f}]'.format(it+1, max_iter, mAP))

        iter_start = time.time()
        # Sample training data for cnn learning
        training_dataloader, sample_index = sample_dataloader(train_dataloader, num_samples, batch_size, root, dataset)

        # Create Similarity matrix
        training_targets = training_dataloader.dataset.get_onehot_targets().to(device)
        S = (training_targets @ train_targets.t() > 0).float()
        S = torch.where(S == 1, torch.full_like(S, 1), torch.full_like(S, -1))

        # Soft similarity matrix, benefit to converge
        r = S.sum() / (1 - S).sum()
        S = S * (1 + r) - r

        # Training CNN model
        CNN_time_start = time.time()
        for epoch in range(max_epoch):
            for batch, (data, targets, index) in enumerate(training_dataloader):
                data, targets, index = data.to(device), targets.to(device), index.to(device)
                optimizer.zero_grad()

                F= model(data)
                U[index, :] = F.data
                criterion = ADSH_Loss(code_length, gamma)
                cnn_loss = criterion(F, B, S[index, :], sample_index[index])
                cnn_loss.backward()
                optimizer.step()
            scheduler.step()
        CNN_time_end = time.time()

        # Update B
        B_time_start = time.time()
        expand_U = torch.zeros(B.shape).to(device)
        expand_U[sample_index, :] = U
        B = solve_dcc(B, U, expand_U, S, code_length, gamma)
        B_time_end = time.time()

        # Total loss
        iter_loss = calc_loss(U, B, S, code_length, sample_index, gamma)
        logger.debug(
            '[iter:{}/{}][loss:{:.2f}][iter_time:{:.2f}]'.format(it + 1, max_iter, iter_loss, time.time() - iter_start))
        logger.debug(
            '[iter:{}/{}][CNN_time:{:.2f}][B_time:{:.2f}]'.format(it + 1, max_iter, CNN_time_end-CNN_time_start, B_time_end-B_time_start))
    logger.info('[Training time:{:.2f}]'.format(time.time() - start))
    logger.info('Best checkpoint saved at: {}'.format(savedir))


def solve_dcc(B, U, expand_U, S, code_length, gamma):
    """
    Solve DCC problem.
    """
    Q = (code_length * S).t() @ U + gamma * expand_U

    for bit in range(code_length):
        q = Q[:, bit]
        u = U[:, bit]
        B_prime = torch.cat((B[:, :bit], B[:, bit + 1:]), dim=1)
        U_prime = torch.cat((U[:, :bit], U[:, bit + 1:]), dim=1)

        B[:, bit] = (q.t() - B_prime @ U_prime.t() @ u.t()).sign()

    return B



def calc_loss(U, B, S, code_length, omega, gamma):
    """
    Calculate loss.
    """
    hash_loss = ((code_length * S - U @ B.t()) ** 2).sum()
    quantization_loss = ((U - B[omega, :]) ** 2).sum()
    loss = (hash_loss + gamma * quantization_loss) / (U.shape[0] * B.shape[0])

    return loss.item()


def generate_code(model, dataloader, code_length, device):
    """
    Generate hash code

    Args
        dataloader(torch.utils.data.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): Using gpu or cpu.

    Returns
        code(torch.Tensor): Hash code.
    """
    model.eval()
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length])
        for data, _, index in dataloader:
            data = data.to(device)
            hash_code = model(data)
            code[index, :] = hash_code.sign().cpu()
    model.train()
    return code
