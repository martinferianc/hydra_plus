from kd.data import options_factory

from kd.models.fc import FC
from kd.models.lenet import LeNetFeatureExtractor
from kd.models.lenet import Classifier as ClassifierLe
from kd.models.resnet import ResNet, Blocks 
from kd.models.resnet import Classifier as ClassifierRes

from kd.models.skeletons import MultiTails, BaseModel
from kd.models.operations import DropoutLinear, GaussLinear, DirichletLinear

def model_factory(dataset, method, n_tails=None, models=None, train=True, hyperparameters={}):
    _, input_size, output_size, _, _ = options_factory(dataset)
    model = None
    if dataset in ["regress", "spiral"]:
        planes_len = len(hyperparameters["planes"])
        if method == "ensemble": 
            if train:
                model = FC(input_size=input_size, output_size=output_size, planes=hyperparameters["planes"])
            else:
                tails = MultiTails(tails= models, n_tails=len(models), n_predictions=len(models), method=method)
                model = BaseModel(tail_model=tails, output_size=output_size)

        elif method == "hydra" or method == "hydra+":
            core_hydra_model = FC(input_size=input_size, output_size=hyperparameters["planes"][planes_len-hyperparameters["depth"]], planes=hyperparameters["planes"][:planes_len-hyperparameters["depth"]], activation_end=True)
            tails_hydra = [FC(input_size=hyperparameters["planes"][planes_len-hyperparameters["depth"]], output_size=output_size, planes=hyperparameters["planes"][planes_len-hyperparameters["depth"]+1:]) for _ in range(n_tails)]
            tails = MultiTails(tails=tails_hydra, n_tails=n_tails, n_predictions=n_tails, method=method)
            model = BaseModel(core_model=core_hydra_model, tail_model=tails, output_size=output_size)

        elif method == "drop":
            core_drop_model = FC(input_size=input_size, output_size=hyperparameters["planes"][planes_len-1], planes=hyperparameters["planes"][:planes_len-1], activation_end=True)
            tail_drop_model = FC(input_size=hyperparameters["planes"][planes_len-1], output_size=output_size, planes=[], end_layer= DropoutLinear(hyperparameters["planes"][planes_len-1], output_size, **hyperparameters["layer_kwargs"]))
            tails = MultiTails(tails=[tail_drop_model], n_tails=1, n_predictions=n_tails, method=method)
            model = BaseModel(core_model=core_drop_model, tail_model=tails, output_size=output_size)

        elif method == "gauss":
            core_gauss_model = FC(input_size=input_size, output_size=hyperparameters["planes"][planes_len-1], planes=hyperparameters["planes"][:planes_len-1], activation_end=True)
            tail_gauss_model = FC(input_size=hyperparameters["planes"][planes_len-1], output_size=output_size, planes=[], end_layer= GaussLinear(hyperparameters["planes"][planes_len-1], output_size, **hyperparameters["layer_kwargs"]))
            tails = MultiTails(tails=[tail_gauss_model], n_tails=1, n_predictions=n_tails, method=method)
            model = BaseModel(core_model=core_gauss_model, tail_model=tails, output_size=output_size)

        elif method == "endd":
            core_dirichlet_model = FC(input_size=input_size, output_size=hyperparameters["planes"][planes_len-1], planes=hyperparameters["planes"][:planes_len-1], activation_end=True)
            tail_dirichlet_model = FC(input_size=hyperparameters["planes"][planes_len-1], output_size=output_size, planes=[], end_layer= DirichletLinear(hyperparameters["planes"][planes_len-1], output_size, **hyperparameters["layer_kwargs"]))
            if train:
                tails = MultiTails(tails=[tail_dirichlet_model], n_tails=1, n_predictions=1, method=method)
            else:
                tails = MultiTails(tails=[tail_dirichlet_model], n_tails=1, n_predictions=n_tails, method=method)
            model = BaseModel(core_model=core_dirichlet_model, tail_model=tails, output_size=output_size)
        else:
            raise ValueError("Method not supported")
    
    elif dataset == "cifar":
        if method == "ensemble":
            if train:
                model = ResNet(layers=hyperparameters["n_layers"], planes = hyperparameters["n_planes"], strides=hyperparameters["strides"], output_size=output_size, end=True)
            else:
                tails = MultiTails(tails=models, n_tails=len(models), n_predictions=len(models), method="ensemble")
                model = BaseModel(tail_model=tails, output_size=output_size)

        elif method == "hydra" or method == "hydra+":
            core_hydra_model = ResNet(layers=hyperparameters["n_layers"][:-1], planes = hyperparameters["n_planes"][:-1], strides=hyperparameters["strides"][:-1], output_size=output_size, end=False)
            tails_hydra= [Blocks(hyperparameters["n_planes"][-2], hyperparameters["n_planes"][-1], hyperparameters["depth"], end=True, output_size=output_size, stride=hyperparameters["strides"][-1]) for _ in range(n_tails)]
            tails = MultiTails(tails=tails_hydra, n_tails=n_tails, n_predictions=n_tails, method=method)
            model = BaseModel(core_model=core_hydra_model, tail_model=tails, output_size=output_size)

        elif method == "drop":
            core_drop_model = ResNet(layers=hyperparameters["n_layers"], planes = hyperparameters["n_planes"], strides=hyperparameters["strides"], output_size=output_size, end=False)
            tail_drop_model = ClassifierRes(hyperparameters["n_planes"][-1], output_size=output_size, end_layer = DropoutLinear(hyperparameters["n_planes"][-1], output_size, **hyperparameters["layer_kwargs"]))
            tails = MultiTails(tails=[tail_drop_model], n_tails=1, n_predictions=n_tails, method=method)
            model = BaseModel(core_model=core_drop_model, tail_model=tails, output_size=output_size)

        elif method == "gauss":
            core_gauss_model = ResNet(layers=hyperparameters["n_layers"], planes = hyperparameters["n_planes"], strides=hyperparameters["strides"], output_size=output_size, end=False)
            tail_gauss_model = ClassifierRes(hyperparameters["n_planes"][-1], output_size=output_size, end_layer = GaussLinear(hyperparameters["n_planes"][-1], output_size, **hyperparameters["layer_kwargs"]))
            tails = MultiTails(tails=[tail_gauss_model], n_tails=1, n_predictions=n_tails, method=method)
            model = BaseModel(core_model=core_gauss_model, tail_model=tails, output_size=output_size)

        elif method == "endd":
            core_dirichlet_model = ResNet(layers=hyperparameters["n_layers"], planes = hyperparameters["n_planes"], strides=hyperparameters["strides"], output_size=output_size, end=False)
            tail_dirichlet_model = ClassifierRes(hyperparameters["n_planes"][-1], output_size=output_size, end_layer = DirichletLinear(hyperparameters["n_planes"][-1], output_size, **hyperparameters["layer_kwargs"]))
            if train:
                tails = MultiTails(tails=[tail_dirichlet_model], n_tails=1, n_predictions=1, method=method)
            else:
                tails = MultiTails(tails=[tail_dirichlet_model], n_tails=1, n_predictions=n_tails, method=method)
            model = BaseModel(core_model=core_dirichlet_model, tail_model=tails, output_size=output_size)
        else:
            raise ValueError("Method not supported")

    elif dataset == "svhn":
        if method == "ensemble":
            if train:
                model = LeNetFeatureExtractor(planes=hyperparameters["planes"], output_size=output_size, end=True, end_planes=hyperparameters["end_planes"])
            else:
                tails = MultiTails(tails=  models, n_tails=len(models), n_predictions=len(models), method="ensemble")
                model = BaseModel(tail_model=tails, output_size=output_size)

        elif method == "hydra" or method == "hydra+":
            core_hydra_model = LeNetFeatureExtractor(planes=hyperparameters["planes"], output_size=output_size, end=False)
            tails_hydra = [ClassifierLe(planes=hyperparameters["end_planes"]) for _ in range(n_tails)]
            tails = MultiTails(tails=tails_hydra, n_tails=n_tails, n_predictions=n_tails, method=method)
            model = BaseModel(core_model=core_hydra_model, tail_model=tails, output_size=output_size)

        elif method == "drop":
            core_drop_model = LeNetFeatureExtractor(planes=hyperparameters["planes"], output_size=output_size, end=False)
            tail_drop_model = ClassifierLe(planes=hyperparameters["end_planes"], end_layer=DropoutLinear(hyperparameters["end_planes"][-2], output_size, **hyperparameters["layer_kwargs"]))
            tails = MultiTails(tails=[tail_drop_model], n_tails=1, n_predictions=n_tails, method=method)
            model = BaseModel(core_model=core_drop_model, tail_model=tails, output_size=output_size)

        elif method == "gauss":
            core_gauss_model = LeNetFeatureExtractor(planes=hyperparameters["planes"], output_size=output_size, end=False)
            tail_gauss_model = ClassifierLe(planes=hyperparameters["end_planes"], end_layer=GaussLinear(hyperparameters["end_planes"][-2], output_size, **hyperparameters["layer_kwargs"]))
            tails = MultiTails(tails=[tail_gauss_model], n_tails=1, n_predictions=n_tails, method=method)
            model = BaseModel(core_model=core_gauss_model, tail_model=tails, output_size=output_size)

        elif method == "endd":
            core_dirichlet_model = LeNetFeatureExtractor(planes=hyperparameters["planes"], output_size=output_size, end=False)
            tail_dirichlet_model = ClassifierLe(planes=hyperparameters["end_planes"], end_layer=DirichletLinear(hyperparameters["end_planes"][-2], output_size, **hyperparameters["layer_kwargs"]))
            if train:
                tails = MultiTails(tails=[tail_dirichlet_model], n_tails=1, n_predictions=1, method=method)
            else:
                tails = MultiTails(tails=[tail_dirichlet_model], n_tails=1, n_predictions=n_tails, method=method)
            model = BaseModel(core_model=core_dirichlet_model, tail_model=tails, output_size=output_size)
        else:
            raise ValueError("Method not supported")
    else:
        raise ValueError("Dataset not supported")
    
    name_children(model, "model")
    return model

# Recursively go through all the children of the given module and name them with respect to their ancestors
def name_children(module, name_parent=""):
    for name, child in module.named_children():
        name_child =  name_parent + "_" + name
        child.name = name_child
        name_children(child, name_parent=name_child)
