import torch.optim as optim


class OptimizerRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(builder_cls):
            cls._registry[name] = builder_cls
            return builder_cls

        return decorator

    @classmethod
    def build(cls, config, params):
        name = config.optim.optimizer
        if name not in cls._registry:
            raise NotImplementedError(f"Optimizer {name} not supported yet!")
        return cls._registry[name].build(config, params)


@OptimizerRegistry.register("Adam")
class AdamOptimizer:

    @staticmethod
    def build(config, params):
        return optim.Adam(
            params,
            lr=config.optim.lr,
            betas=(config.optim.beta1, config.optim.beta2),
            eps=config.optim.eps,
            weight_decay=config.optim.weight_decay,
        )


@OptimizerRegistry.register("AdamW")
class AdamWOptimizer:

    @staticmethod
    def build(config, params):
        return optim.AdamW(
            params,
            lr=config.optim.lr,
            betas=(config.optim.beta1, config.optim.beta2),
            eps=config.optim.eps,
            weight_decay=config.optim.weight_decay,
        )
