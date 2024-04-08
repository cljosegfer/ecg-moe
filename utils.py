
import torch
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
    # log = 0
    log = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader):
            raw, exam_id, label = batch
            ecg = get_inputs(raw).to(device)
            label = label.to(device).float()

            logits = model.forward(ecg)
            loss = criterion(logits, label)

            # log += loss.item()
            log.append(loss.item())
    # return log / len(loader)
    return log

def synthesis(model, loader, best_thresholds = None, device = "cuda"):
    if best_thresholds == None:
        num_classes = len(loader.output_cols)
        print(num_classes)
        thresholds = np.arange(0, 1.01, 0.01)  # Array of thresholds from 0 to 1 with step 0.01
        predictions = {thresh: [[] for _ in range(num_classes)] for thresh in thresholds}
        true_labels_dict = [[] for _ in range(num_classes)]
    else:
        all_binary_results = []
        all_true_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader):
            raw, exam_id, label = batch
            ecg = get_inputs(raw).to(device)
            label = label.to(device).float()

            logits = model(ecg)
            probs = torch.sigmoid(logits)

            if best_thresholds == None:
                for class_idx in range(num_classes):
                    for thresh in thresholds:
                        predicted_binary = (probs[:, class_idx] >= thresh).float()
                        predictions[thresh][class_idx].extend(
                            predicted_binary.cpu().numpy()
                        )
                    true_labels_dict[class_idx].extend(
                        label[:, class_idx].cpu().numpy()
                    )
            else:
                binary_result = torch.zeros_like(probs)
                for i in range(len(best_thresholds)):
                    binary_result[:, i] = (
                        probs[:, i] >= best_thresholds[i]
                    ).float()
                
                all_binary_results.append(binary_result)
                all_true_labels.append(label)
    
    if best_thresholds == None:
        best_f1s, best_thresholds = find_best_thresholds(predictions, true_labels_dict, thresholds)
        return best_f1s, best_thresholds
    else:
        all_binary_results = torch.cat(all_binary_results, dim=0)
        all_true_labels = torch.cat(all_true_labels, dim=0)
        return all_binary_results, all_true_labels, metrics_table(all_binary_results, all_true_labels)

def find_best_thresholds(predictions, true_labels_dict, thresholds):
    num_classes = len(predictions[0])
    print(num_classes)
    best_thresholds = [0.5] * num_classes
    best_f1s = [0.0] * num_classes

    for class_idx in (range(num_classes)):
        for thresh in thresholds:
            f1 = f1_score(
                true_labels_dict[class_idx],
                predictions[thresh][class_idx],
                zero_division=0,
            )

            if f1 > best_f1s[class_idx]:
                best_f1s[class_idx] = f1
                best_thresholds[class_idx] = thresh
    
    return best_f1s, best_thresholds

def metrics_table(all_binary_results, all_true_labels):
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    num_classes = all_binary_results.shape[-1]
    print(num_classes)
    for class_idx in range(num_classes):
        class_binary_results = all_binary_results[:, class_idx].cpu().numpy()
        class_true_labels = all_true_labels[:, class_idx].cpu().numpy()

        accuracy = accuracy_score(class_true_labels, class_binary_results)
        precision = precision_score(
            class_true_labels, class_binary_results, zero_division=0
        )
        recall = recall_score(
            class_true_labels, class_binary_results, zero_division=0
        )
        f1 = f1_score(class_true_labels, class_binary_results, zero_division=0)

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    metrics_dict = {
        "Accuracy": accuracy_scores,
        "Precision": precision_scores,
        "Recall": recall_scores,
        "F1 Score": f1_scores,
    }

    return metrics_dict

def plot_log(log_trn, log_val = None, epoch = None):
    plt.figure();
    plt.plot(log_trn);
    if log_val != None:
        # plt.axhline(y = log_val, color = 'tab:orange');
        # plt.title('trn: {}, val: {}'.format(np.mean(log_trn), log_val));
        plt.plot(log_val);
    plt.title('trn: {}, val: {}'.format(np.mean(log_trn), np.mean(log_val)));
    if epoch != None:
        plt.savefig('output/loss_{}.png'.format(epoch));
        plt.close();
    else:
        plt.show();

def export(model, model_label, epoch):
    print('exporting partial model at epoch {}'.format(epoch))
    torch.save(model, 'output/partial_{}.pt'.format(model_label))