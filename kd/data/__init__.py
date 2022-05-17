
from kd.data.cifar import get_cifar_train_valid_test_loaders, get_cifar_options
from kd.data.svhn import get_svhn_train_valid_test_loaders, get_svhn_options
from kd.data.regress import get_regress_train_valid_test_loaders, get_regress_options
from kd.data.spiral import get_spiral_train_valid_test_loaders, get_spiral_options

def data_factory(name, batch_size=128, random=False, level = 0, augmentation=""):
    if name == 'cifar':
        return get_cifar_train_valid_test_loaders(batch_size=batch_size, random=random, level=level, augmentation=augmentation)
    elif name == 'svhn':
        return get_svhn_train_valid_test_loaders(batch_size=batch_size, random=random, level=level, augmentation=augmentation)
    elif name == 'regress':
        return get_regress_train_valid_test_loaders(batch_size=batch_size, random=random, level=level, augmentation=augmentation)
    elif name == 'spiral':
        return get_spiral_train_valid_test_loaders(batch_size=batch_size, random=random, level=level, augmentation=augmentation)
    else:
        raise ValueError("Unknown dataset: %s" % name)

def options_factory(name):
    if name == 'cifar':
        return get_cifar_options()
    elif name == 'svhn':
        return get_svhn_options()
    elif name == 'regress':
        return get_regress_options()
    elif name == 'spiral':
        return get_spiral_options()
    else:
        raise ValueError("Unknown dataset: %s" % name)

