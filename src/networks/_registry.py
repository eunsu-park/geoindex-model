"""Model registry with decorator-based registration."""

from typing import Dict, Callable, List

_MODEL_REGISTRY: Dict[str, Callable] = {}


def register_model(name: str):
    """Decorator to register a model factory function."""
    def decorator(factory_fn):
        _MODEL_REGISTRY[name] = factory_fn
        return factory_fn
    return decorator


def create_model(config):
    """Create model based on configuration using the registry."""
    model_type = config.model.model_type
    if model_type not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model_type: '{model_type}'. Available: {available}")
    return _MODEL_REGISTRY[model_type](config)


def list_models() -> List[str]:
    """Return sorted list of registered model names."""
    return sorted(_MODEL_REGISTRY.keys())
