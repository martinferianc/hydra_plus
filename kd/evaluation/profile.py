from deeplite.torch_profiler.torch_profiler import *
from deeplite.torch_profiler.torch_profiler import flops_counter
from deeplite.profiler import Device
from collections import OrderedDict
import time
from deeplite.torch_profiler.torch_profiler import TorchProfiler
import torch
import logging
import numpy as np

from kd.models.operations import DirichletLinear, GaussLinear, DropoutLinear
from kd.models.resnet import Add
from kd.data import options_factory

def profile_single_part(part, N, input_size, output_size, device, seed, name):
    if len(input_size) == 1:
        input_size = (1,1) + input_size
    # Create tensor dataset with the given input size and normal distribution
    data = torch.randn(N, *input_size, dtype=torch.float32)
    random_labels = torch.randint(0, output_size, (N,), dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(data, random_labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False)
    loader_train = torch.utils.data.DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False)

    # Create Profiler class and register the profiling functions
    data_loader = TorchProfiler.enable_forward_pass_data_splits({"train":loader_train, "test":loader})
    profiler = TorchProfiler(part, data_loader, name=name)
    profiler.register_profiler_function(ComputeComplexity())

    # Compute the registered profiler metrics for the PyTorch Model
    metrics = profiler.compute_network_status(batch_size=1, device=device,short_print=False,
                                                    include_weights=True)
    return metrics

def dropoutlinear_flops_counter_hook(module, input, output):
    # First dropout is performed than the linear layer
    param_size = 4
    activation_size = 4
    input = input[0]
    output_last_dim = output.shape[-1]
    batch_size = output.shape[0]
    bias_flops = output_last_dim if module.bias is not None else 0

    total_activations, memory_footprint, output_shape = flops_counter.parse_module_output(module, output, activation_size)
    
    module.__hook_variables__ = flops_counter.HookVariables()
    module.__hook_variables__.mac += int(np.prod(input.shape) * output_last_dim + bias_flops) + np.prod(input.shape) # Accounting for both the linear layer and the dropout operation
    module.__hook_variables__.params += sum(p.numel() for p in module.parameters() if p.requires_grad)
    module.__hook_variables__.activations += sum(total_activations)
    module.__hook_variables__.model_size += module.__hook_variables__.params * param_size
    module.__hook_variables__.memory_footprint += sum(memory_footprint)

    m_key = flops_counter.get_m_key(module.__class__)
    summary = OrderedDict()
    summary["m_key"] = m_key
    summary['layer_time'] = time.time()
    summary["input_shape"] = flops_counter.get_input_shape(input)
    summary["input_shape"][0] = batch_size
    summary["output_shape"] = output_shape
    if hasattr(module, "weight") and hasattr(module.weight, "size"):
        summary["layer_weight_size"] = list(module.weight.size())
    if hasattr(module, "bias") and hasattr(module.bias, "size"):
        summary["layer_bias_size"] = list(module.bias.size())
    module.__hook_variables__.summary = summary
    flops_counter.layer_count += 1

def gausslinear_flops_counter_hook(module, input, output):
    # First dropout is performed than the linear layer
    param_size = 4
    activation_size = 4
    input = input[0]
    output_last_dim = output.shape[-1]
    batch_size = output.shape[0]
    bias_flops = output_last_dim if module.bias is not None else 0

    total_activations, memory_footprint, output_shape = flops_counter.parse_module_output(module, output, activation_size)
    
    module.__hook_variables__ = flops_counter.HookVariables()
    module.__hook_variables__.mac += int(np.prod(input.shape) * output_last_dim + bias_flops) * 2 
    module.__hook_variables__.mac += int(np.prod(module.weight.shape)) # To account for the exp on the weights
    if module.bias is not None:
        module.__hook_variables__.mac += int(np.prod(module.bias.shape))
    module.__hook_variables__.mac += int(np.prod(input.shape)) # To account for the pow(2)
    module.__hook_variables__.mac += int(np.prod(output_shape)) # To account for the sqrt
    module.__hook_variables__.params += sum(p.numel() for p in module.parameters() if p.requires_grad)
    module.__hook_variables__.activations += sum(total_activations)
    module.__hook_variables__.model_size += module.__hook_variables__.params * param_size
    module.__hook_variables__.memory_footprint += sum(memory_footprint)

    m_key = flops_counter.get_m_key(module.__class__)
    summary = OrderedDict()
    summary["m_key"] = m_key
    summary['layer_time'] = time.time()
    summary["input_shape"] = flops_counter.get_input_shape(input)
    summary["input_shape"][0] = batch_size
    summary["output_shape"] = output_shape
    if hasattr(module, "weight") and hasattr(module.weight, "size"):
        summary["layer_weight_size"] = list(module.weight.size()*2) # Accounting also for the log variance
    if hasattr(module, "bias") and hasattr(module.bias, "size"):
        summary["layer_bias_size"] = list(module.bias.size()*2) # Accounting also for the log variance
    module.__hook_variables__.summary = summary
    flops_counter.layer_count += 1

@torch.no_grad()
def profile_model(model, results, args, N=1000):
    dataset = args.dataset
    seed = args.seed
    # This is because of Pytorch 1.10.0 where they have decided to remove complex32 (?)
    torch.complex32 = None

    # First define the functions to count flops for the custom modules
    MODULES_MAPPING = flops_counter.MODULES_MAPPING
    MODULES_MAPPING[DropoutLinear] = dropoutlinear_flops_counter_hook
    # Here we just use the default counter, since the linear layer output could be cached
    # And the  sampling from the dirichlet distribution is inexpensive when compared to the rest of the computation
    MODULES_MAPPING[DirichletLinear] = flops_counter.linear_flops_counter_hook
    # Here we assume that the Gauss linear layer is using the local reparameterization trick and we cache the features
    # Again the sampling from the Gauassian is inexpensive when compared to the rest of the computation
    MODULES_MAPPING[GaussLinear] = gausslinear_flops_counter_hook
    MODULES_MAPPING[Add] = flops_counter.relu_flops_counter_hook

    flops_counter.MODULES_MAPPING = MODULES_MAPPING

    # The profiling will be split intwo two parts: one for the feature extractior and the second into tails 
    # We will collect statistrics for a) the whole model b) the feature extractor and c) the tails

    # Begin with the feature extractor
    _, input_size, output_size, _, _ = options_factory(dataset)
    device = Device.GPU if args.gpu >= 0 else Device.CPU

    metrics = profile_single_part(model.core_model, N, input_size, output_size,device, seed, "core")

    core_flops = metrics['flops'] * 10**9
    core_params = metrics['total_params'] * 10**6
    core_zero_params = 0
    for p in model.core_model.parameters():
        core_zero_params+= p.numel() - p.nonzero().size(0)
    core_model_size = metrics['model_size']
    core_model_size_zero_params = core_zero_params * (4 / 10**6)
    core_memory_footprint = metrics['memory_footprint']
    core_layerwise_summary = metrics['layerwise_summary']

    logging.info("#### Core Model FLOPs: {} ####".format(core_flops))
    logging.info("#### Core Model Params: {} ####".format(core_params))
    logging.info("#### Core Model Zero Params: {} ####".format(core_zero_params))
    logging.info("#### Core Model Size in MB: {} ####".format(core_model_size))
    logging.info("#### Core Model Size of Zero Params in MB: {} ####".format(core_model_size_zero_params))
    logging.info("#### Core Model Memory Footprint in MB: {} ####".format(core_memory_footprint))
    logging.info("#### Core Model Layerwise Summary ####")
    logging.info(core_layerwise_summary)

    results['core_model'] = {}
    results['core_model']['flops'] = core_flops
    results['core_model']['params'] = core_params
    results['core_model']['zero_params'] = core_zero_params
    results['core_model']['model_size'] = core_model_size
    results['core_model']['model_size_zero_params'] = core_model_size_zero_params
    results['core_model']['memory_footprint'] = core_memory_footprint
    results['core_model']['layerwise_summary'] = core_layerwise_summary

    # Move to the tails 
    # Create a dummy tensor that gets passed through the base model to find what should be the size for the data loader
    dummy_tensor = torch.randn(1, *input_size,device=next(model.parameters()).device)
    tail_input_shape = tuple(model.core_model(dummy_tensor).shape)[1:]
    # Select just a single tail 
    metrics = profile_single_part(model.tail_model.tails[0], N, tail_input_shape, output_size, device, seed, "tail")

    tail_flops = metrics['flops'] * 10**9
    tail_params = metrics['total_params'] * 10**6
    tail_zero_params = 0
    for p in model.tail_model.tails.parameters():
        tail_zero_params += p.numel() - p.nonzero().size(0)
    tail_model_size = metrics['model_size']
    tail_model_size_zero_params = tail_zero_params * (4 / 10**6)
    tail_memory_footprint = metrics['memory_footprint']
    tail_layerwise_summary = metrics['layerwise_summary']

    logging.info("#### Tail FLOPs: {} ####".format(tail_flops))
    logging.info("#### Tail Params: {} ####".format(tail_params))
    logging.info("#### Tail Zero Params: {} ####".format(tail_zero_params))
    logging.info("#### Tail Size in MB: {} ####".format(tail_model_size))
    logging.info("#### Tail Size of Zero Params in MB: {} ####".format(tail_model_size_zero_params))
    logging.info("#### Tail Memory Footprint in MB: {} ####".format(tail_memory_footprint))
    logging.info("#### Tail Layerwise Summary ####")
    logging.info(tail_layerwise_summary)

    results['tail'] = {}
    results['tail']['flops'] = tail_flops
    results['tail']['params'] = tail_params
    results['tail']['zero_params'] = tail_zero_params
    results['tail']['model_size'] = tail_model_size
    results['tail']['model_size_zero_params'] = tail_model_size_zero_params
    results['tail']['memory_footprint'] = tail_memory_footprint
    results['tail']['layerwise_summary'] = tail_layerwise_summary

    # Compute the total FLOPs and total params, this will depend on the method used
    if args.method in ["hydra+", "hydra", "ensemble"]:
        tail_params*=args.n_tails
        tail_model_size*=args.n_tails

    if args.method not in ["endd", "gauss"]:
        tail_flops*=args.n_tails
        tail_memory_footprint*=args.n_tails

    total_flops = core_flops + tail_flops
    total_params = core_params + tail_params
    total_zero_params = core_zero_params + tail_zero_params
    total_model_size = core_model_size + tail_model_size
    total_model_size_zero_params = core_model_size_zero_params + tail_model_size_zero_params
    total_memory_footprint = core_memory_footprint + tail_memory_footprint

    logging.info("#### Total FLOPs: {} ####".format(total_flops))
    logging.info("#### Total Params: {} ####".format(total_params))
    logging.info("#### Total Zero Params: {} ####".format(total_zero_params))
    logging.info("#### Total Size in MB: {} ####".format(total_model_size))
    logging.info("#### Total Size of Zero Params in MB: {} ####".format(total_model_size_zero_params))
    logging.info("#### Total Memory Footprint in MB: {} ####".format(total_memory_footprint))

    results["flops"] = total_flops
    results["params"] = total_params
    results["zero_params"] = total_zero_params
    results["model_size"] = total_model_size
    results["model_size_zero_params"] = total_model_size_zero_params
    results["memory_footprint"] = total_memory_footprint


