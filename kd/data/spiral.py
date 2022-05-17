import logging
import math
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler

def rotate_point(point, angle):
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return rotation_matrix.dot(point)

def generate_spiral(samples, start, end, angle):
    # Generate points from the square root of random data inside an uniform distribution on [0, 1).
    points = math.radians(start) + np.sqrt(np.random.rand(samples, 1)) * math.radians(end)

    # Apply a rotation to the points.
    rotated_x_axis = np.cos(points) * points + np.random.rand(samples, 1) * 0.5
    rotated_y_axis = np.sin(points) * points + np.random.rand(samples, 1) * 0.5

    # Stack the vectors inside a samples x 2 matrix.
    rotated_points = np.column_stack((rotated_x_axis, rotated_y_axis))
    return np.apply_along_axis(rotate_point, 1, rotated_points, math.radians(angle))

def get_spiral_options():
    return "classification", (2,), 3, [[0]], ["random"]

def get_spiral_train_valid_test_loaders(batch_size=300, random=False, level=0, augmentation="", N = 300, arms =3):
    data = np.empty((0, 3))
    angles = [(360 / arms) * i for i in range(arms)]

    for i, angle in enumerate(angles):
        points = generate_spiral(N//arms, 0.0, 360, angle)
        classified_points = np.hstack((points, np.full((N//arms, 1), i)))
        data = np.concatenate((data, classified_points))

    data = torch.from_numpy(data).float()
    if random:
        # Create a cartesian product of the data with x ranging from -20 to 20 and y ranging from -20 to 20
        x,y = torch.linspace(-20, 20, 100), torch.linspace(-20, 20, 100)
        inputs = torch.cartesian_prod(x, y)
        # The labels are not important in this case 
        labels = torch.randint(0, arms, (len(inputs),)).float().unsqueeze(-1)
        data = torch.cat((inputs, labels), dim=1)

    # Split the data into train, validation and test sets.
    indices = list(range(len(data)))
    indices = torch.randperm(len(indices))

    valid_portion = 0.1 
    test_portion = 0.1
    valid_split = int(len(indices) * valid_portion)
    test_split = int(len(indices) * (valid_portion + test_portion))
    valid_idx, test_idx, train_idx = indices[:valid_split], indices[valid_split:test_split], indices[test_split:]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    inps = data[:, :2]
    tgts = data[:, 2].long()
    final_data = torch.utils.data.TensorDataset(inps, tgts)
    
    train_loader = torch.utils.data.DataLoader(final_data, batch_size=batch_size, sampler=train_sampler, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(final_data, batch_size=batch_size, sampler=valid_sampler, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(final_data, batch_size=batch_size, sampler=test_sampler, pin_memory=True)
    if random:
        test_loader = torch.utils.data.DataLoader(final_data, batch_size=batch_size, shuffle=False, pin_memory=True)

    logging.info("### Spiral: Train size: %d, Validation size: %d, Test size: %d ###" % (len(train_idx), len(valid_idx), len(test_idx)))
    
    return train_loader, valid_loader, test_loader
    
    
