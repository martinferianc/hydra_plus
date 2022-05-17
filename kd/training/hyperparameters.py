from kd.training.utils import LinearScheduler
import logging

def get_hyperparameters(dataset, method, n_tails, overwrite_hyperparameters={}):
    if dataset not in ["cifar", "regress", "spiral", "svhn"]:
        raise ValueError("Dataset not supported")
    if method not in ["endd", "gauss", "drop", "hydra", "hydra+", "ensemble"]:
        raise ValueError("Method not supported")

    hyperparameters = {}
    hyperparameters["layer_kwargs"] = {}
    hyperparameters["lambda_scheduler"] = 0.0 
    hyperparameters["kl_scheduler"] = 0.0
    hyperparameters["alpha_scheduler"] = 0.0
    hyperparameters["beta_scheduler"] = 1.0
    hyperparameters["temperature_mean_scheduler"] = 0.0
    hyperparameters["temperature_individual_scheduler"] = 0.0
    hyperparameters["method"] = method
    hyperparameters["grad_clip"] = 5.0

    if dataset == "spiral":
        hyperparameters["lr"] = 0.01
        hyperparameters["planes"] = [100, 100, 100, 100]
        hyperparameters["epochs"] = 200
        if method == "ensemble":
            hyperparameters["wd"] = 0.0001
        else:
            hyperparameters["wd"] = 0.00000001
            hyperparameters["temperature_mean_scheduler"] = 1.0
            hyperparameters["temperature_individual_scheduler"] = 3.0
            hyperparameters["alpha_scheduler"] = 0.9
            hyperparameters["depth"] = 2
        
        if method == "gauss":
            hyperparameters["kl_scheduler"] = 1.0/240

        elif method == "hydra+":
            if n_tails == 20:
                hyperparameters["lambda_scheduler"] = 4.0
            elif n_tails == 10:
                hyperparameters["lambda_scheduler"] = 7.
            elif n_tails == 5:
                hyperparameters["lambda_scheduler"] = 9.
            hyperparameters["beta_scheduler"] = 0.5


    elif dataset == "regress":
        hyperparameters["epochs"] = 200
        hyperparameters["lr"] = 0.05
        hyperparameters["planes"] = [50, 50, 50]
        if method == "ensemble":
            hyperparameters["wd"] = 0.00001
        else:
            hyperparameters["wd"] = 0.00000001
            hyperparameters["temperature_mean_scheduler"] = 1.0
            hyperparameters["temperature_individual_scheduler"] = 1.0
            hyperparameters["alpha_scheduler"] = 0.9
            hyperparameters["depth"] = 2

        if method == "gauss":
            hyperparameters["kl_scheduler"] = 1.0/240

        elif method == "hydra+":
            hyperparameters["beta_scheduler"] = .5
            if n_tails == 20:
                hyperparameters["lambda_scheduler"] = LinearScheduler(start_epoch=50, end_epoch=150, epochs=hyperparameters["epochs"], start_value =0.0, end_value=0.002)
            elif n_tails == 10:
                hyperparameters["lambda_scheduler"] = LinearScheduler(start_epoch=50, end_epoch=150, epochs=hyperparameters["epochs"], start_value =0.0, end_value=0.02)
            elif n_tails == 5:
                hyperparameters["lambda_scheduler"] = LinearScheduler(start_epoch=50, end_epoch=150, epochs=hyperparameters["epochs"], start_value =0.0, end_value=0.6)

    elif dataset == "cifar":
        hyperparameters["epochs"] = 200
        hyperparameters["lr"] = 0.01
        hyperparameters["n_layers"] = [2, 2, 2, 2]
        hyperparameters["strides"] = [1, 2, 2, 2]

        if method == "ensemble":
            hyperparameters["wd"] = 0.0001
            hyperparameters["n_planes"] = [32, 64, 128, 256]
        else:
            hyperparameters["wd"] = 0.0000001
            hyperparameters["temperature_mean_scheduler"] = 1.0
            hyperparameters["temperature_individual_scheduler"] = 8.0
            hyperparameters["alpha_scheduler"] = 0.95
            hyperparameters["n_planes"] = [96, 192, 256, 128]
            hyperparameters["depth"] = 2

        if method == "gauss":
            hyperparameters["kl_scheduler"] = 1/45000

        elif method == "hydra+":
            if n_tails == 20:
                hyperparameters["lambda_scheduler"] = LinearScheduler(start_epoch=20, end_epoch=150, epochs=hyperparameters["epochs"], start_value =0.0, end_value=0.002)
            elif n_tails == 10:
                hyperparameters["lambda_scheduler"] = LinearScheduler(start_epoch=20, end_epoch=150, epochs=hyperparameters["epochs"], start_value =0.0, end_value=0.05)
            elif n_tails == 5:
                hyperparameters["lambda_scheduler"] = LinearScheduler(start_epoch=20, end_epoch=150, epochs=hyperparameters["epochs"], start_value =0.0, end_value=0.1)
            hyperparameters["beta_scheduler"] = 0.5

    elif dataset == "svhn":
        hyperparameters["epochs"] = 200
        hyperparameters["lr"] = 0.001

        if method == "ensemble":
            hyperparameters["wd"] = 0.0001
            hyperparameters["planes"] = [32, 64]
            hyperparameters["end_planes"] = [64*4*4, 128, 10]
        else:
            hyperparameters["wd"] = 0.0000001
            hyperparameters["planes"] = [128, 32]
            hyperparameters["end_planes"] = [32*4*4,128,128, 10]
            hyperparameters["temperature_mean_scheduler"] = 1.0
            hyperparameters["temperature_individual_scheduler"] = 5.0
            hyperparameters["alpha_scheduler"] = 0.95

        if method == "gauss":
            hyperparameters["kl_scheduler"] = 1/65932

        elif method == "hydra+":
            if n_tails == 20:
                hyperparameters["lambda_scheduler"] = 0.0005
            elif n_tails == 10:
                hyperparameters["lambda_scheduler"] = 0.001
            elif n_tails == 5:
                hyperparameters["lambda_scheduler"] = 0.005
            hyperparameters["beta_scheduler"] = 0.9
            
    if method == "endd":
        hyperparameters["alpha_scheduler"] = 0.0
        hyperparameters["beta_scheduler"] = 0.0

    if method == "hydra":
        hyperparameters["alpha_scheduler"] = 1.0

    if method == "drop":
        hyperparameters["layer_kwargs"] = {"dropout":0.5}

    if method == "gauss":
        hyperparameters["layer_kwargs"] = {"sigma_prior":1.0}
    
    for key, value in overwrite_hyperparameters.items():
        logging.info("#### Overwriting hyperparameter {} with {}, was {}".format(key, value, hyperparameters[key]))
        hyperparameters[key] = value
    return hyperparameters