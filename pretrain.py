
# # setup
import torch
import torch.nn as nn


# # train gate setup
model_label = 'gate'

from configs.gate import LoadDataConfig, Downstream_cnn_args
from data.load_data import LoadData
from models.baseline import ResnetBaseline
from utils import train, eval, plot_log, export


# # init
loader_config = LoadDataConfig()
resnet_config = Downstream_cnn_args()

dataloader = LoadData(**loader_config.__dict__)
model = ResnetBaseline(**resnet_config.__dict__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 5


# # train
model = model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

log = []
for epoch in range(EPOCHS):
    train_dl, val_dl = dataloader.get_train_dataloader(), dataloader.get_val_dataloader()

    train_log = train(model, train_dl, optimizer, criterion, device)
    val_log = eval(model, val_dl, criterion, device)
    plot_log(train_log, val_log, epoch = epoch)
    export(model, model_label, epoch)

torch.save(model, 'output/{}.pt'.format(model_label))
