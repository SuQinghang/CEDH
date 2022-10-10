import os
import time

import torch
import torch.optim as optim
import torch.nn.functional as F
from loguru import logger
from torch.optim.lr_scheduler import CosineAnnealingLR

import models.alexnet as alexnet
import utils.evaluate as evaluate
from data.data_loader import sample_dataloader
from utils.sim_matrix import cosine_S, smooth_S
import torch.nn as nn

class CEDH_Loss(nn.Module):
    """
    Loss function of ADSH

    Args:
        code_length(int): Hashing code length.
        gamma(float): Hyper-parameter.
    """
    def __init__(self, code_length, gamma):
        super(CEDH_Loss, self).__init__()
        self.code_length = code_length
        self.gamma = gamma

    def forward(self, F, B, S, omega):

        hash_loss = ((self.code_length * S - F @ B.t()) ** 2).sum()
        quantization_loss = ((F - B[omega, :]) ** 2).sum()

        loss = (hash_loss + self.gamma * quantization_loss) / (F.shape[0] * B.shape[0])

        return loss


def train(
        dataset,
        dataset_root,
        query_dataloader,
        train_dataloader,
        retrieval_dataloader,
        
        original_method,
        original_code_dir,
        original_length,
        target_length,
        
        max_iter,
        max_epoch,
        batch_size,
        num_samples,
        lr,
        W_lambda,
        gamma,
        topk,
        device,
        sim_S,
        alpha,
        eval_epoch,
        save_name,
):
    """
    Training model.

    Args
        dataset(str): name of dataset.
        dataset_root(str): root path of dataset.
        query_dataloader, train_dataloader, retrieval_dataloader(torch.utils.data.dataloader.DataLoader): Data loader.

        original_method(str): original method used to generate Hashing code.
        code_dir(str): dir path of original Hashing code.
        original_length(int): original Hashing code length.
        target_length(int): target Hashing code length.
        num_samples(int): number of data sampled to train cnn.
        batch_size(int): number of images in a batch.

        lr(float): learning rate.
        max_iter(int): number of iterations to train the whole framework.
        max_epoch(int): number of epoch to train cnn.
        W_lambda(float): hyper-parameter to regularize matrix W.
        alpha(float): hyper-parameter to soft the constraint in multi-label scence.
        gamma(float): hyper-parameter to trade-off quantization loss.
        topk(int): Topk k map.
        device(torch.device): GPU or CPU.
        save_name(str): name of checkpoint save dir.

    Returns
        mAP(float): Mean Average Precision.
    """
    # Initialization
    step_size = target_length - original_length 
    

    if original_method == 'cedh':
        B1 = torch.load(os.path.join(original_code_dir, 'training_code{}.t'.format(original_length))).to(device)
        original_retrieval_code = torch.load(os.path.join(original_code_dir, 'retrieval_code{}.t'.format(original_length))).to(device)
        retrieval_targets = torch.load(os.path.join(original_code_dir, 'retrieval_targets{}.t'.format(original_length))).to(device)

    else:
        #* original hashing code of training data
        B1 = torch.load(os.path.join(original_code_dir, 'training_code{}.t'.format(original_length))).to(device)
        #* original hashing code of retrieval data
        original_retrieval_code = torch.load(os.path.join(original_code_dir, 'retrieval_code{}.t'.format(original_length))).to(device)
        #* targets of retrieval data
        retrieval_targets = torch.load(os.path.join(original_code_dir, 'retrieval_targets{}.t'.format(original_length))).to(device)
        # retrieval_targets = retrieval_dataloader.dataset.get_onehot_targets().to(device)
    # import ipdb;ipdb.set_trace()
    model = alexnet.load_model(target_length).to(device)
    criterion = CEDH_Loss(target_length, gamma)
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5,
    )
    scheduler = CosineAnnealingLR(optimizer, max_iter, 1e-7)
    
    U = torch.zeros(num_samples, target_length).to(device)
    W = torch.rand(original_length, step_size).to(device)
    B2 = (B1@W)
    B = torch.cat((B1, B2), 1)

    Z = torch.zeros(B.shape[0], step_size).to(device)
    best_mAP = 0.0
    timestr = time.strftime('%Y-%m-%d-%H:%M', time.gmtime())
    savedir = os.path.join('checkpoints', 'cedh', dataset, save_name+'-'+timestr)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    for it in range(max_iter):
        
        #! evaluate and save best
        # '''
        if (it)%eval_epoch == 0 or it == max_iter-1:
            query_code = generate_code(model, query_dataloader, target_length, device)
            new_retrieval_code = (original_retrieval_code @ W).sign()
            retrieval_code = torch.cat((original_retrieval_code, new_retrieval_code),1)
            mAP = evaluate.mean_average_precision(
                query_code.to(device),
                retrieval_code,
                query_dataloader.dataset.get_onehot_targets().to(device),
                retrieval_targets,
                device,
                topk,
            )
            #!save best
            if mAP >= best_mAP:
                best_mAP = mAP
                training_code = generate_code(model, train_dataloader, target_length, device)
                torch.save(training_code.cpu(), os.path.join(savedir, 'training_code{}.t'.format(target_length)))
                torch.save(query_code.cpu(), os.path.join(savedir, 'query_code{}.t'.format(target_length)))
                query_targets = query_dataloader.dataset.get_onehot_targets().to(device)
                torch.save(query_targets.cpu(), os.path.join(savedir, 'query_targets{}.t'.format(target_length)))
                # retrieval集扩展后新的长度的hashcode
                torch.save(retrieval_code.cpu(), os.path.join(savedir, 'retrieval_code{}.t'.format(target_length)))
                torch.save(retrieval_targets.cpu(), os.path.join(savedir, 'retrieval_targets{}.t'.format(target_length)))
                torch.save(model, os.path.join(savedir, 'model-{}.t'.format(target_length)))
                torch.save(W.cpu(), os.path.join(savedir, 'W-{}.t'.format(target_length)))
            logger.info('[iter:{}/{}][mAP:{:4f}]'.format(it+1, max_iter, mAP))
        #! evaluate end
        # '''
        iter_start = time.time()
        #* Sample training data for cnn learning
        training_dataloader, sample_index = sample_dataloader(train_dataloader, num_samples, batch_size, dataset_root, dataset)

        #* Create Similarity matrix
        #* the targets of the sampled training data
        training_targets = training_dataloader.dataset.get_onehot_targets().to(device)
        #* the targets of the whole training data
        train_targets = train_dataloader.dataset.get_onehot_targets().to(device)

        if sim_S == 'cosine':
            S = cosine_S(training_targets, train_targets)
        elif sim_S == 'smooth':
            S = smooth_S(training_targets, train_targets, alpha)

        # Training CNN model
        CNN_time_start = time.time()
        for epoch in range(max_epoch):
            for batch, (data, targets, index) in enumerate(training_dataloader):
                data, targets, index = data.to(device), targets.to(device), index.to(device)
                optimizer.zero_grad()

                f = model(data)

                U[index, :] = f.data
                
                # cnn_loss = criterion(f, B, S[index, :], sample_index[index])
                hash_loss = ((target_length * S[index, :] - f @ B.t()) **2).sum()
                # import ipdb; ipdb.set_trace()
                quantization_loss = ((f[:, :original_length] - B[sample_index[index]][:,:original_length]) ** 2).sum()+\
                                        ((f[:, original_length:] - Z[sample_index[index]])**2).sum()
                cnn_loss = (hash_loss + gamma * quantization_loss) / (f.shape[0] * B.shape[0])
                cnn_loss.backward()
                optimizer.step()
        
        scheduler.step()
        CNN_time_end = time.time()

        # update W
        U1 = U[:, :original_length]
        U2 = U[:, original_length:]
        W_time_start = time.time()
        temp1 = B1.t() @ B1 + W_lambda * torch.eye(original_length).to(device)
        temp2 = target_length * S - U1 @ B1.t()
        temp3 = U2.t() @ U2 + W_lambda * torch.eye(step_size).to(device)
        W = torch.inverse(temp1) @ (B1.t() @ temp2.t() @ U2  + B1.t() @ Z) @ torch.inverse(temp3)
        W_time_end = time.time()

        # update Z
        expand_U2 = torch.zeros(B2.shape).to(device)
        expand_U2[sample_index] = U2
        Z = (expand_U2 + B1 @ W).sign()

        B2 = (B1 @ W)
        B[:, original_length:] = B2

        # Total loss
        iter_loss = calc_loss(U, B, S, target_length, sample_index, gamma)
        logger.debug(
            '[iter:{}/{}][loss:{:.2f}][iter_time:{:.2f}]'.
                        format(it + 1, max_iter, iter_loss, time.time() - iter_start))
        logger.debug(
            '[iter:{}/{}][CNN_time:{:.2f}][W_time:{:.2f}]'.format(it + 1, max_iter, CNN_time_end-CNN_time_start, W_time_end-W_time_start))

    logger.info('Best checkpoint saved at: {}'.format(savedir))

def solve_dcc(B1, B2, U1, U2, expand_U2, S, code_length, step_size, gamma, alpha, W):
    """
    Solve DCC problem.
    """
    Z = code_length * S - U1 @ (B1.t())
    P = -2*(Z.t() @ U2 + gamma * expand_U2 + alpha * (B1 @ W))

    for bit in range(step_size):
        p        = P[:, bit]
        u2       = U2[:, bit]
        B_prime  = torch.cat((B2[:, :bit], B2[:, bit + 1:]), dim=1)
        U2_prime = torch.cat((U2[:, :bit], U2[:, bit + 1:]), dim=1)

        B2[:, bit] = -(2 * B_prime @ U2_prime.t() @ u2 + p).sign()

    return B2



def calc_loss(U, B, S, code_length, omega, gamma):
    """
    Calculate loss.
    """
    hash_loss         = ((code_length * S - U @ B.t()) ** 2).sum()
    quantization_loss = ((U - B[omega, :]) ** 2).sum()
    loss              = (hash_loss + gamma * quantization_loss) / (U.shape[0] * B.shape[0])

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
