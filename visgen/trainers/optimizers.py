from torch.optim import (ASGD, SGD, Adadelta, Adagrad, Adam, Adamax, AdamW,
                         RMSprop)

key2opt = {
    "sgd": SGD,
    "adam": Adam,
    "adamw": AdamW,
    "asgd": ASGD,
    "adamax": Adamax,
    "adadelta": Adadelta,
    "adagrad": Adagrad,
    "rmsprop": RMSprop,
}


def get_optimizer(opt_dict, model_params):
    optimizer = _get_optimizer_instance(opt_dict)
    params = {k: v for k, v in opt_dict.items() if k != "name"}
    optimizer = optimizer(model_params, **params)
    return optimizer


def _get_optimizer_instance(opt_dict):
    if opt_dict is None:
        return SGD
    opt_name = opt_dict["name"]
    if opt_name not in key2opt:
        raise NotImplementedError(f"Optimizer {opt_name} not implemented")
    return key2opt[opt_name]
