import torch

device = None
nr_gpus = None
if torch.cuda.is_available():
    device = torch.device('cuda')
    nr_gpus = torch.cuda.device_count()
else:
    device = torch.device('cpu')
    
