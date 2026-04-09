"""
Component Parser Module.

This module is responsible for parsing time series forecasting model component configuration information
from experiment result paths.

Main Components:
    - ComponentInfo: Component information dataclass, stores all component configurations parsed from path
    - ComponentParser: Component parser, provides path parsing functionality
    - ModelIdentifier: Model identifier, identifies model family based on rules

Component Classification:
    Basic components (12, parsed from path by underscore):
        - gym_x_mark: Whether to use time marks
        - gym_series_sampling: Whether to use series sampling
        - gym_series_norm: Series normalization method
        - gym_series_decomp: Series decomposition method
        - gym_channel_independent: Whether channel independent
        - gym_input_embed: Input embed/tokenizer type
        - gym_network_architecture: Network architecture (model type)
        - gym_attn: Attention/backbone network type
        - gym_feature_attn: Feature attention
        - gym_encoder_only: Whether encoder-only
        - gym_frozen: Whether frozen
        - gym_rag: Whether to use RAG

    Hyperparameter components (8, extracted from path using regex):
        - sequence_length: Sequence length (sl)
        - d_model: Model dimension (dm)
        - d_ff: Feedforward network dimension (df)
        - encoder_layers: Number of encoder layers (el)
        - training_epochs: Number of training epochs (epochs)
        - loss_function: Loss function (lf)
        - learning_rate: Learning rate (lr)
        - lradjust: Learning rate scheduler (lrs)

    Optional components:
        - gym_pl: Prediction length (pl)

Supported model families for identification:
    - PatchTST: Based on series-patching input embedding
    - DLinear: MLP + DNN + MA decomposition
    - OLinear: MLP + NormLin + ortho embedding
    - Autoformer: Transformer + auto attention
    - TimeMixer: MLP + series sampling + decomposition

Path format example:
    'LTF_TSGym1000497_True_True_Stat_None_False_series-encoding_MLP_NormLin_...'

Author: TSGym
"""
# data/component_parser.py
import re
from typing import Dict, List, Callable, Tuple, Optional
from dataclasses import dataclass


# Component name list, order matches parsing order in path
# First 12 are obtained by splitting path by underscore, last 8 are extracted from hyperparameter string using regex
BASE_COMPONENT_NAMES = [
    'gym_x_mark',             # 0: Whether to use time marks
    'gym_series_sampling',    # 1: Whether to use series sampling
    'gym_series_norm',        # 2: Series normalization method
    'gym_series_decomp',      # 3: Series decomposition method
    'gym_channel_independent', # 4: Whether channel independent
    'gym_input_embed',        # 5: Input embed/tokenizer type
    'gym_network_architecture',# 6: Network architecture (model type)
    'gym_attn',               # 7: Attention/backbone network type
    'gym_feature_attn',       # 8: Feature attention
    'gym_encoder_only',       # 9: Whether encoder-only
    'gym_frozen',             # 10: Whether frozen
    'gym_rag',                # 11: Whether to use RAG
]

HP_COMPONENT_NAMES = [
    'sequence_length',   # sl
    'd_model',           # dm
    'd_ff',              # df
    'encoder layers',    # el
    'training epochs',   # epochs
    'loss function',     # lf
    'learning rate',     # lr
    'lradjust',          # lrs (lr scheduler)
]

PERIOD_COMPONENT_NAME = 'gym_pl'  # pl (prediction length)


@dataclass
class ComponentInfo:
    """Component information - all components parsed from path"""
    gym_x_mark: str
    gym_series_sampling: str
    gym_series_norm: str
    gym_series_decomp: str
    gym_channel_independent: str
    gym_input_embed: str
    gym_network_architecture: str
    gym_attn: str
    gym_feature_attn: str
    gym_encoder_only: str
    gym_frozen: str
    gym_rag: str
    sequence_length: str
    d_model: str
    d_ff: str
    encoder_layers: str
    training_epochs: str
    loss_function: str
    learning_rate: str
    lradjust: str
    gym_pl: Optional[str] = None
    path: str = ''

    @property
    def model_family(self) -> str:
        """Identify model family (e.g., PatchTST, DLinear, etc.)"""
        return ModelIdentifier.identify(self)

    # Aliases for backward compatibility
    @property
    def gym_CI(self) -> str:
        return self.gym_channel_independent

    @property
    def gym_series_tokenizer(self) -> str:
        return self.gym_input_embed

    @property
    def gym_model(self) -> str:
        return self.gym_network_architecture

    @property
    def gym_backbone(self) -> str:
        return self.gym_attn

    def to_dict(self) -> Dict:
        """Convert to dictionary format (backward compatible)"""
        d = {
            'gym_x_mark': self.gym_x_mark,
            'gym_series_sampling': self.gym_series_sampling,
            'gym_series_norm': self.gym_series_norm,
            'gym_series_decomp': self.gym_series_decomp,
            'gym_channel_independent': self.gym_channel_independent,
            'gym_input_embed': self.gym_input_embed,
            'gym_network_architecture': self.gym_network_architecture,
            'gym_attn': self.gym_attn,
            'gym_feature_attn': self.gym_feature_attn,
            'gym_encoder_only': self.gym_encoder_only,
            'gym_frozen': self.gym_frozen,
            'gym_rag': self.gym_rag,
            'sequence_length': self.sequence_length,
            'd_model': self.d_model,
            'd_ff': self.d_ff,
            'encoder layers': self.encoder_layers,
            'training epochs': self.training_epochs,
            'loss function': self.loss_function,
            'learning rate': self.learning_rate,
            'lradjust': self.lradjust,
            'path': self.path
        }
        if self.gym_pl is not None:
            d['gym_pl'] = self.gym_pl
        return d

    def to_component_list(self, include_pl: bool = False) -> List[str]:
        """Convert to component value list (consistent with original code current_components format)"""
        result = [
            self.gym_x_mark,
            self.gym_series_sampling,
            self.gym_series_norm,
            self.gym_series_decomp,
            self.gym_channel_independent,
            self.gym_input_embed,
            self.gym_network_architecture,
            self.gym_attn,
            self.gym_feature_attn,
            self.gym_encoder_only,
            self.gym_frozen,
            self.gym_rag,
            self.sequence_length,
            self.d_model,
            self.d_ff,
            self.encoder_layers,
            self.training_epochs,
            self.loss_function,
            self.learning_rate,
            self.lradjust,
        ]
        if include_pl and self.gym_pl is not None:
            result.append(self.gym_pl)
        return result


class ModelIdentifier:
    """Model identifier - identifies model family based on rules"""

    _rules: List[Tuple[str, Callable]] = []

    @classmethod
    def register_rule(cls, name: str, predicate: Callable[[ComponentInfo], bool]):
        cls._rules.append((name, predicate))

    @classmethod
    def identify(cls, component: ComponentInfo) -> str:
        for name, predicate in cls._rules:
            if predicate(component):
                return name
        return 'unknown'

    @classmethod
    def list_rules(cls) -> List[str]:
        return [name for name, _ in cls._rules]

# Register all model identification rules
ModelIdentifier.register_rule(
    'PatchTST',
    lambda c: c.gym_input_embed == 'series-patching'
)

ModelIdentifier.register_rule(
    'DLinear',
    lambda c: (c.gym_network_architecture == 'MLP' and
               c.gym_attn == 'DNN' and
               c.gym_series_decomp == 'MA')
)

ModelIdentifier.register_rule(
    'OLinear',
    lambda c: (c.gym_network_architecture == 'MLP' and
               c.gym_attn == 'NormLin' and
               'ortho' in c.gym_input_embed)
)

ModelIdentifier.register_rule(
    'Autoformer',
    lambda c: (c.gym_network_architecture == 'Transformer' and
               'auto' in c.gym_attn)
)

ModelIdentifier.register_rule(
    'TimeMixer',
    lambda c: (c.gym_network_architecture == 'MLP' and
               c.gym_series_sampling == 'True' and
               c.gym_series_decomp != 'None')
)


class ComponentParser:
    """Component parser - parses component information from paths

    Path format example:
    'results_long_term_forecasting/resultsGym_MLP/ECL/LTF_TSGym1000497_True_True_Stat_None_False_series-encoding_MLP_NormLin_null_True_False_False_ECL_ftM_sl512_ll48_pl96_dm64_el2_dl1_df256_fc3_ebtimeF_dtTrue_Exp_epochs30_lfPSLoss_lr0.0001_lrscosine_0'

    Parsing logic (consistent with original run_meta_dl.py):
    1. Remove directory path name
    3. Find part after TSGym\\d*, split by underscore
    4. First 12 are basic components
    5. Part after 12th is concatenated into hyperparameter string, extract sl/dm/df/el/epochs/lf/lr/lrs/pl using regex
    """

    @staticmethod
    def parse_path(path: str, task_name1: str = 'LTF', task_name2: str = 'long_term_forecasting',
                   dataset: Optional[str] = None,
                   include_pl: bool = False) -> ComponentInfo:
        """
        Parse component information from experiment path

        Args:
            path: Filename part of experiment result (e.g., LTF_TSGym1000497_True_True_...)
            task_name: Task name prefix (e.g., 'LTF'), used to remove prefix
            dataset: Dataset name (e.g., 'ECL'), if provided, concatenated to front of path
            include_pl: Whether to parse prediction length (pl)

        Returns:
            ComponentInfo object
        """
        try:
            path = path.split("/")[-1]
            # Find part after TSGym\d*
            tsm = re.search(r'TSGym\d*', path)
            if not tsm:
                raise ValueError(f"Cannot find TSGym pattern in path")

            after_tsgym = path[tsm.end() + 1:]  # +1 skip underscore
            parts = after_tsgym.split('_')

            # First 12 are basic components
            if len(parts) < 12:
                raise ValueError(f"Expected at least 12 base components, got {len(parts)}")
            base_components = parts[:12]

            # Part after 12th is concatenated into hyperparameter string
            k_HP = '_'.join(parts[12:])

            # Extract values from hyperparameter string using regex
            sl = re.search(r'_sl(\d+)_', k_HP) or re.search(r'^sl(\d+)_', k_HP)
            dm = re.search(r'_dm(\d+)_', k_HP)
            df = re.search(r'_df(\d+)_', k_HP)
            el = re.search(r'_el(\d+)_', k_HP)
            epochs = re.search(r'_epochs(\d+)_', k_HP) or re.search(r'epochs(\d+)_', k_HP)
            lf = re.search(r'lf([^_]+)', k_HP)
            lr = re.search(r'_lr([\d.]+)_', k_HP)
            lrs = re.search(r'lrs([^_]+)', k_HP)

            hp_values = [
                sl.group(1) if sl else None,
                dm.group(1) if dm else None,
                df.group(1) if df else None,
                el.group(1) if el else None,
                epochs.group(1) if epochs else None,
                lf.group(1) if lf else None,
                lr.group(1) if lr else None,
                lrs.group(1) if lrs else None,
            ]

            # Validate that all required hyperparameters were extracted
            if any(v is None for v in hp_values):
                missing_indices = [i for i, v in enumerate(hp_values) if v is None]
                hp_names = ['sl', 'dm', 'df', 'el', 'epochs', 'lf', 'lr', 'lrs']
                missing_names = [hp_names[i] for i in missing_indices]
                raise ValueError(f"Failed to extract hyperparameters {missing_names} from path '{path}'")

            # Optional: parse prediction length
            pl_value = None
            if include_pl:
                pl_match = re.search(r'_pl(\d+)_', k_HP)
                if pl_match:
                    pl_value = pl_match.group(1)

            return ComponentInfo(
                gym_x_mark=base_components[0],
                gym_series_sampling=base_components[1],
                gym_series_norm=base_components[2],
                gym_series_decomp=base_components[3],
                gym_channel_independent=base_components[4],
                gym_input_embed=base_components[5],
                gym_network_architecture=base_components[6],
                gym_attn=base_components[7],
                gym_feature_attn=base_components[8],
                gym_encoder_only=base_components[9],
                gym_frozen=base_components[10],
                gym_rag=base_components[11],
                sequence_length=hp_values[0],
                d_model=hp_values[1],
                d_ff=hp_values[2],
                encoder_layers=hp_values[3],
                training_epochs=hp_values[4],
                loss_function=hp_values[5],
                learning_rate=hp_values[6],
                lradjust=hp_values[7],
                gym_pl=pl_value,
                path=path
            )
        except (ValueError, IndexError, AttributeError, TypeError) as e:
            raise ValueError(f"Failed to parse path '{path}': {e}")

    @staticmethod
    def parse_batch(paths: List[str], **kwargs) -> List[ComponentInfo]:
        """Batch parse paths"""
        return [ComponentParser.parse_path(p, **kwargs) for p in paths]


# Compatibility functions (maintain backward compatibility with old code)
def parse_path(path: str, **kwargs) -> Dict:
    """Parse function for backward compatibility with old code"""
    return ComponentParser.parse_path(path, **kwargs).to_dict()

def is_PatchTST(components_dict: Dict) -> bool:
    """Backward compatibility"""
    return components_dict.get('gym_input_embed', components_dict.get('gym_series_tokenizer', '')) == 'series-patching'

def is_DLinear(components_dict: Dict) -> bool:
    """Backward compatibility"""
    return (components_dict.get('gym_network_architecture', components_dict.get('gym_model', '')) == 'MLP' and
            components_dict.get('gym_attn', components_dict.get('gym_backbone', '')) == 'DNN' and
            components_dict['gym_series_decomp'] == 'MA')

def is_OLinear(components_dict: Dict) -> bool:
    """Backward compatibility"""
    return (components_dict.get('gym_network_architecture', components_dict.get('gym_model', '')) == 'MLP' and
            components_dict.get('gym_attn', components_dict.get('gym_backbone', '')) == 'NormLin' and
            'ortho' in components_dict.get('gym_input_embed', components_dict.get('gym_series_tokenizer', '')))

def is_autoformer(components_dict: Dict) -> bool:
    """Backward compatibility"""
    return (components_dict.get('gym_network_architecture', components_dict.get('gym_model', '')) == 'Transformer' and
            'auto' in components_dict.get('gym_attn', components_dict.get('gym_backbone', '')))

def is_timemixer(components_dict: Dict) -> bool:
    """Backward compatibility"""
    return (components_dict.get('gym_network_architecture', components_dict.get('gym_model', '')) == 'MLP' and
            components_dict['gym_series_sampling'] == 'True' and
            components_dict['gym_series_decomp'] != "None")


if __name__ == '__main__':
    # ========== Test Path ==========
    test_filename = ('LTF_TSGym1000497_True_True_Stat_None_False_series-encoding'
                     '_MLP_NormLin_null_True_False_False_ECL_ftM_sl512_ll48_pl96'
                     '_dm64_el2_dl1_df256_fc3_ebtimeF_dtTrue_Exp_epochs30'
                     '_lfPSLoss_lr0.0001_lrscosine_0')

    # ---------- 1. Basic Parsing ----------
    print('=' * 60)
    print('Test 1: Basic parsing (without pl)')
    print('=' * 60)
    info = ComponentParser.parse_path(test_filename, task_name='LTF', dataset='ECL')
    print(f'  gym_x_mark            = {info.gym_x_mark}')
    print(f'  gym_series_sampling   = {info.gym_series_sampling}')
    print(f'  gym_series_norm       = {info.gym_series_norm}')
    print(f'  gym_series_decomp     = {info.gym_series_decomp}')
    print(f'  gym_channel_independent = {info.gym_channel_independent}')
    print(f'  gym_input_embed       = {info.gym_input_embed}')
    print(f'  gym_network_architecture = {info.gym_network_architecture}')
    print(f'  gym_attn              = {info.gym_attn}')
    print(f'  gym_feature_attn      = {info.gym_feature_attn}')
    print(f'  gym_encoder_only      = {info.gym_encoder_only}')
    print(f'  gym_frozen            = {info.gym_frozen}')
    print(f'  gym_rag               = {info.gym_rag}')
    print(f'  sequence_length       = {info.sequence_length}')
    print(f'  d_model               = {info.d_model}')
    print(f'  d_ff                  = {info.d_ff}')
    print(f'  encoder_layers        = {info.encoder_layers}')
    print(f'  training_epochs       = {info.training_epochs}')
    print(f'  loss_function         = {info.loss_function}')
    print(f'  learning_rate         = {info.learning_rate}')
    print(f'  lradjust              = {info.lradjust}')
    print(f'  gym_pl                = {info.gym_pl}')

    # Verify values
    assert info.gym_x_mark == 'True'
    assert info.gym_series_sampling == 'True'
    assert info.gym_series_norm == 'Stat'
    assert info.gym_series_decomp == 'None'
    assert info.gym_channel_independent == 'False'
    assert info.gym_input_embed == 'series-encoding'
    assert info.gym_network_architecture == 'MLP'
    assert info.gym_attn == 'NormLin'
    assert info.gym_feature_attn == 'null'
    assert info.gym_encoder_only == 'True'
    assert info.gym_frozen == 'False'
    assert info.gym_rag == 'False'
    assert info.sequence_length == '512'
    assert info.d_model == '64'
    assert info.d_ff == '256'
    assert info.encoder_layers == '2'
    assert info.training_epochs == '30'
    assert info.loss_function == 'PSLoss'
    assert info.learning_rate == '0.0001'
    assert info.lradjust == 'cosine'
    assert info.gym_pl is None
    print('  [PASS] All basic assertions passed')

    # ---------- 2. Parsing with pl ----------
    print()
    print('=' * 60)
    print('Test 2: Parsing with pl')
    print('=' * 60)
    info_pl = ComponentParser.parse_path(test_filename, task_name='LTF', dataset='ECL', include_pl=True)
    assert info_pl.gym_pl == '96'
    print(f'  gym_pl = {info_pl.gym_pl}')
    print('  [PASS] pl parsing correct')

    # ---------- 3. Backward Compatible Aliases ----------
    print()
    print('=' * 60)
    print('Test 3: Backward compatible field aliases')
    print('=' * 60)
    assert info.gym_CI == 'False'
    assert info.gym_series_tokenizer == 'series-encoding'
    assert info.gym_model == 'MLP'
    assert info.gym_backbone == 'NormLin'
    print(f'  gym_CI                = {info.gym_CI}')
    print(f'  gym_series_tokenizer  = {info.gym_series_tokenizer}')
    print(f'  gym_model             = {info.gym_model}')
    print(f'  gym_backbone          = {info.gym_backbone}')
    print('  [PASS] Alias backward compatibility correct')

    # ---------- 4. to_dict ----------
    print()
    print('=' * 60)
    print('Test 4: to_dict()')
    print('=' * 60)
    d = info.to_dict()
    assert d['gym_network_architecture'] == 'MLP'
    assert d['loss function'] == 'PSLoss'
    assert 'gym_pl' not in d  # Should not be included when pl is not enabled
    d_pl = info_pl.to_dict()
    assert d_pl['gym_pl'] == '96'
    print(f'  dict keys ({len(d)}): {list(d.keys())}')
    print('  [PASS] to_dict correct')

    # ---------- 5. to_component_list ----------
    print()
    print('=' * 60)
    print('Test 5: to_component_list()')
    print('=' * 60)
    cl = info.to_component_list()
    assert len(cl) == 20
    assert cl[:12] == ['True', 'True', 'Stat', 'None', 'False', 'series-encoding',
                       'MLP', 'NormLin', 'null', 'True', 'False', 'False']
    assert cl[12:] == ['512', '64', '256', '2', '30', 'PSLoss', '0.0001', 'cosine']
    cl_pl = info_pl.to_component_list(include_pl=True)
    assert len(cl_pl) == 21
    assert cl_pl[-1] == '96'
    print(f'  component_list (len={len(cl)}): {cl}')
    print(f'  with pl (len={len(cl_pl)}): last = {cl_pl[-1]}')
    print('  [PASS] component_list correct')

    # ---------- 6. Model Identification ----------
    print()
    print('=' * 60)
    print('Test 6: Model family identification')
    print('=' * 60)
    assert info.model_family == 'unknown'  # NormLin + series-encoding does not match registered rules
    print(f'  Current path model family: {info.model_family}')

    # Construct PatchTST path
    patchtst_fn = test_filename.replace('series-encoding', 'series-patching')
    info_patch = ComponentParser.parse_path(patchtst_fn, task_name='LTF', dataset='ECL')
    assert info_patch.model_family == 'PatchTST'
    print(f'  PatchTST path identification: {info_patch.model_family}')

    # Construct DLinear path
    dlinear_fn = test_filename.replace('NormLin', 'DNN').replace('None', 'MA', 1)
    info_dl = ComponentParser.parse_path(dlinear_fn, task_name='LTF', dataset='ECL')
    assert info_dl.model_family == 'DLinear'
    print(f'  DLinear path identification: {info_dl.model_family}')
    print('  [PASS] Model identification correct')

    # ---------- 7. Compatibility Functions ----------
    print()
    print('=' * 60)
    print('Test 7: Compatibility functions is_XXX')
    print('=' * 60)
    d_patch = info_patch.to_dict()
    assert is_PatchTST(d_patch)
    assert not is_DLinear(d_patch)
    d_dl = info_dl.to_dict()
    assert is_DLinear(d_dl)
    assert not is_PatchTST(d_dl)
    print('  [PASS] Compatibility functions correct')

    # ---------- 8. Parsing without dataset ----------
    print()
    print('=' * 60)
    print('Test 8: Without providing dataset parameter')
    print('=' * 60)
    info_no_ds = ComponentParser.parse_path(test_filename, task_name='LTF')
    assert info_no_ds.gym_x_mark == 'True'
    assert info_no_ds.sequence_length == '512'
    print('  [PASS] Parsing correct even without dataset')

    # ---------- 9. Error Path ----------
    print()
    print('=' * 60)
    print('Test 9: Error path handling')
    print('=' * 60)
    try:
        ComponentParser.parse_path('invalid_path_without_tsgym')
        assert False, 'Should raise ValueError'
    except ValueError as e:
        print(f'  Caught expected exception: {e}')
        print('  [PASS] Error path correctly raises exception')

    # ---------- 10. Batch Parsing ----------
    print()
    print('=' * 60)
    print('Test 10: batch parsing')
    print('=' * 60)
    batch = ComponentParser.parse_batch([test_filename, patchtst_fn], task_name='LTF', dataset='ECL')
    assert len(batch) == 2
    assert batch[0].gym_input_embed == 'series-encoding'
    assert batch[1].gym_input_embed == 'series-patching'
    print(f'  batch size: {len(batch)}')
    print('  [PASS] batch parsing correct')

    print()
    print('=' * 60)
    print('All tests passed!')
    print('=' * 60)
