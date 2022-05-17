import torch
import logging

def get_regress_options():
    return "regression", (1,), 2, [[0]], ["random"]

def toy_y(x):
    # Define a toy regression problem with a non-linear target function
    return torch.sin(x) -0.1 * x +0.1*x**2 + 0.01*x**3 

def get_regress_train_valid_test_loaders(batch_size=300, random=False, level=0, augmentation="", N = 300, start=-6., end=6., noise = 1.0):
    valid_portion = 0.1
    test_portion = 0.1

    train_size = int(N*(1-valid_portion-test_portion))
    valid_size = int(N*valid_portion)
    test_size = int(N*test_portion)

    train_x = (end-start) * torch.rand(train_size) + start
    train_y = toy_y(train_x) + noise * torch.randn(train_size)

    valid_x= torch.linspace(start, end, valid_size) 
    valid_y = toy_y(valid_x)

    test_x= torch.linspace(start, end, test_size) 
    test_y = toy_y(test_x)

    if random:
        test_x = torch.linspace(-12, 12, 1000)
        test_y = toy_y(test_x)

    train_dataset = torch.utils.data.TensorDataset(train_x.unsqueeze(-1), train_y.unsqueeze(-1))
    valid_dataset = torch.utils.data.TensorDataset(valid_x.unsqueeze(-1), valid_y.unsqueeze(-1))
    test_dataset = torch.utils.data.TensorDataset(test_x.unsqueeze(-1), test_y.unsqueeze(-1))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logging.info("### Regress: Train size: %d, Validation size: %d, Test size: %d ###" % (len(train_dataset), len(valid_dataset), len(test_dataset)))
    return train_loader, valid_loader, test_loader
    
    
