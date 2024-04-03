
import torch
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

SIGNAL_CROP_LEN = 2560
SIGNAL_NON_ZERO_START = 571

def get_inputs(batch, apply = "non_zero", device = "cuda"):
    # (B, C, L)
    if batch.shape[1] > batch.shape[2]:
        batch = batch.permute(0, 2, 1)

    B, n_leads, signal_len = batch.shape

    if apply == "non_zero":
        transformed_data = torch.zeros(B, n_leads, SIGNAL_CROP_LEN)
        for b in range(B):
            start = SIGNAL_NON_ZERO_START
            diff = signal_len - start
            if start > diff:
                correction = start - diff
                start -= correction
            end = start + SIGNAL_CROP_LEN
            for l in range(n_leads):
                transformed_data[b, l, :] = batch[b, l, start:end]

    else:
        transformed_data = batch

    return transformed_data.to(device)

def train(model, loader, optimizer, criterion, device = "cuda"):
    log = []
    model.train()
    for batch in tqdm(loader):
        raw, exam_id, label = batch
        ecg = get_inputs(raw).to(device)
        label = label.to(device).float()

        logits = model.forward(ecg)
        loss = criterion(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log.append(loss.item())
    return log

def eval(model, loader, criterion, device = "cuda"):
    log = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader):
            raw, exam_id, label = batch
            ecg = get_inputs(raw).to(device)
            label = label.to(device).float()

            logits = model.forward(ecg)
            loss = criterion(logits, label)

            log += loss.item()
    return log / len(loader)

def plot_log(log_trn, log_val = None, epoch = None):
    plt.figure();
    plt.plot(log_trn);
    if log_val != None:
        plt.axhline(y = log_val, color = 'tab:orange');
        plt.title('trn: {}, val: {}'.format(np.mean(log_trn), log_val));
    if epoch != None:
        plt.savefig('output/loss_{}.png'.format(epoch));
        plt.close();
    else:
        plt.show();

def export(model, model_label, epoch):
    print('exporting partial model at epoch {}'.format(epoch))
    torch.save(model, 'output/partial_{}.pt'.format(model_label))