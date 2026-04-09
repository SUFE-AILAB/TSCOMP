"""
Model Factory Module.

This module provides a unified model creation interface, managing instantiation of various meta-learning
models through the factory pattern.

Main Components:
    - ModelConfig: Model configuration dataclass, encapsulates all parameters needed for model creation
    - ModelFactory: Model factory class, provides model registration and creation functionality

Model Type Design:
    The model_type parameter contains two parts of information: architecture type + mask strategy, format is "{architecture}-{mask}"

    Supported architecture types:
        - mlp: Basic MLP meta-predictor
        - icl: Standard ICL model (multi-head attention)
        - icl-simple: Simplified ICL model (no Q/K/V projection)
        - icl-frozencomp: Frozen component embedding ICL model
        - icl-addcomp: Additive component embedding ICL model
        - icl-labelencoder: Label encoder ICL model (no embedding layer)
        - icl-deepinput: Deep input projection ICL model
        - icl-tabpfn: TabPFN ICL model

    Supported mask strategies (ICL models only):
        - nomask: No mask, fully connected attention
        - simplemask: Simple mask, test points cannot see each other
        - mask-similar-meta: Mask training samples with similar meta-features (formerly icl-mq)
        - mask-train-self: Training samples can only see themselves/diagonal (formerly icl-hls)
        - mask-test-train: Test samples cannot see training samples (formerly icl-ls)
        - mask-train-peers: Training samples cannot see each other (formerly icl-yx)

    Combination examples:
        - "icl-nomasktrain": Standard ICL + no mask
        - "icl-simplemask": Standard ICL + simple mask
        - "icl-simple-mask-train-self": Simplified ICL + training sees only self
        - "icl-frozencomp-mask-similar-meta": Frozen component ICL + mask similar meta-features

Usage:
    Register model builder functions using the @ModelFactory.register decorator,
    then create model instances through ModelFactory.create_model.

Author: TSGym
"""
# core/model_factory.py
from typing import Dict, Any, Optional
from dataclasses import dataclass
import torch.nn as nn

@dataclass
class ModelConfig:
    """Model configuration class - uses dataclass for type safety"""
    n_col: list  # List of category counts for each component
    meta_feature_dim: int  # Meta feature dimension
    d_model: int  # Model hidden layer dimension
    dropout: float  # Dropout ratio
    n_layers: int  # Number of layers
    nhead: Optional[int] = None  # Number of attention heads (only needed for ICL models)
    k: float = 0.0  # Attention truncation parameter (only for ICL models)
    temporal: float = 1.0  # Temperature parameter (only for ICL models)
    model_type: str = 'mlp'  # Model type identifier
    add_embed_dim: int = 256  # Additive embedding dimension (only for icl-addcomp)
    tabpfn_model_path: Optional[str] = None  # TabPFN model path

class ModelFactory:
    """Model factory class - uniformly creates various meta-learning models"""

    _registry = {}  # Model registry: {model_type: builder_func}

    @classmethod
    def register(cls, model_type: str):
        """Decorator: Register model builder function"""
        def decorator(builder_func):
            cls._registry[model_type] = builder_func
            return builder_func
        return decorator

    @classmethod
    def create_model(cls, model_type: str, config: ModelConfig, device) -> nn.Module:
        """
        Create model instance

        Args:
            model_type: Model type (e.g., 'mlp', 'icl-simple', 'icl-frozencomp')
            config: Model configuration object
            device: Target device

        Returns:
            Created model instance (already moved to specified device)

        Raises:
            ValueError: If model type is not registered
        """
        # Support fuzzy matching (e.g., 'icl-simple-xxx' matches 'icl-simple')
        builder = cls._find_builder(model_type)
        if builder is None:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available types: {list(cls._registry.keys())}"
            )

        model = builder(config)
        return model.to(device)

    @classmethod
    def _find_builder(cls, model_type: str):
        """
        Find model builder function (supports prefix matching)

        Matching rules:
        1. Exact match takes priority
        2. Prefix matches are sorted by length in descending order to ensure longest prefix matches first
           For example: "icl-simple-nomasktrain" should match "icl-simple" not "icl"
        """
        # Exact match
        if model_type in cls._registry:
            return cls._registry[model_type]

        # Prefix match: sort by registered key length in descending order to ensure longest prefix takes priority
        sorted_registry = sorted(cls._registry.items(), key=lambda x: len(x[0]), reverse=True)
        for registered_type, builder in sorted_registry:
            if model_type.startswith(registered_type + '-') or model_type == registered_type:
                return builder

        return None

    @classmethod
    def list_models(cls):
        """List all registered model types"""
        return list(cls._registry.keys())

# ============================================================================
# Model Registration
# Naming convention: registration key is the architecture type (without mask strategy),
# mask strategy is passed through config.model_type
# ============================================================================

@ModelFactory.register('mlp')
def build_mlp(config: ModelConfig):
    """Build MLP model (does not use ICL architecture, no mask strategy)"""
    from core.networks import meta_predictor
    return meta_predictor(
        n_col=config.n_col,
        embed_dim_meta_feature=config.meta_feature_dim,
        d_model=config.d_model,
        dropout=config.dropout,
        n_layers=config.n_layers
    )

@ModelFactory.register('icl-simple')
def build_icl_simple(config: ModelConfig):
    """Build Simple ICL model (simplified attention, no Q/K/V projection)"""
    from core.networks import MetaSimpleICL
    return MetaSimpleICL(
        n_col=config.n_col,
        embed_dim_meta_feature=config.meta_feature_dim,
        d_model=config.d_model,
        dropout=config.dropout,
        num_layers=config.n_layers,
        model_type=config.model_type,  # Pass complete model_type (with mask strategy)
        k=config.k,
        temporal=config.temporal
    )

@ModelFactory.register('icl-frozencomp')
def build_icl_frozencomp(config: ModelConfig):
    """Build Frozen Component ICL model (frozen component embedding)"""
    from core.networks import MetaICLFrozenComp
    return MetaICLFrozenComp(
        n_col=config.n_col,
        embed_dim_meta_feature=config.meta_feature_dim,
        d_model=config.d_model,
        dropout=config.dropout,
        nhead=config.nhead,
        num_layers=config.n_layers,
        model_type=config.model_type,  # Pass complete model_type (with mask strategy)
        k=config.k,
        temporal=config.temporal
    )

@ModelFactory.register('icl-addcomp')
def build_icl_addcomp(config: ModelConfig):
    """Build Add Component ICL model (additive component embedding)"""
    from core.networks import MetaICLAddComp
    return MetaICLAddComp(
        n_col=config.n_col,
        embed_dim_meta_feature=config.meta_feature_dim,
        d_model=config.d_model,
        dropout=config.dropout,
        nhead=config.nhead,
        num_layers=config.n_layers,
        model_type=config.model_type,  # Pass complete model_type (with mask strategy)
        k=config.k,
        temporal=config.temporal,
        add_embed_dim=config.add_embed_dim
    )

@ModelFactory.register('icl-labelencoder')
def build_icl_labelencoder(config: ModelConfig):
    """Build Label Encoder ICL model (no embedding layer, uses normalized label values)"""
    from core.networks import MetaICLLabelEncoder
    return MetaICLLabelEncoder(
        n_col=config.n_col,
        embed_dim_meta_feature=config.meta_feature_dim,
        d_model=config.d_model,
        dropout=config.dropout,
        nhead=config.nhead,
        num_layers=config.n_layers,
        model_type=config.model_type,  # Pass complete model_type (with mask strategy)
        k=config.k,
        temporal=config.temporal
    )

@ModelFactory.register('icl-deepinput')
def build_icl_deepinput(config: ModelConfig):
    """Build Deep Input ICL model (deep input projection)"""
    from core.networks import MetaICLDeepInput
    return MetaICLDeepInput(
        n_col=config.n_col,
        embed_dim_meta_feature=config.meta_feature_dim,
        d_model=config.d_model,
        dropout=config.dropout,
        nhead=config.nhead,
        num_layers=config.n_layers,
        model_type=config.model_type,  # Pass complete model_type (with mask strategy)
        k=config.k,
        temporal=config.temporal
    )

@ModelFactory.register('icl-tabpfn')
def build_icl_tabpfn(config: ModelConfig):
    """Build TabPFN ICL model (based on pretrained TabPFN, does not use mask strategy)"""
    from networks import MetaICLTabPFN
    if config.tabpfn_model_path is None:
        raise ValueError("tabpfn_model_path must be provided for icl-tabpfn model")
    return MetaICLTabPFN(
        n_col=config.n_col,
        model_path=config.tabpfn_model_path,
        device='cuda'  # Will be moved to correct device in create_model
    )

@ModelFactory.register('icl')
def build_icl(config: ModelConfig):
    """Build standard ICL model (multi-head attention, default ICL implementation)"""
    from core.networks import MetaICL
    return MetaICL(
        n_col=config.n_col,
        embed_dim_meta_feature=config.meta_feature_dim,
        d_model=config.d_model,
        dropout=config.dropout,
        nhead=config.nhead,
        num_layers=config.n_layers,
        model_type=config.model_type,  # Pass complete model_type (with mask strategy)
        k=config.k,
        temporal=config.temporal
    )