"""Select and verify the one current development baseline; historical runs opt in."""
from __future__ import annotations

import json
from pathlib import Path

from .io import repair_evaluator_code_hashes, sha256_file
from .schema import EvaluationError


REGISTRY_PATH = Path(__file__).parent / 'configs/current_protocol.json'
CURRENT_PROTOCOL = json.loads(REGISTRY_PATH.read_text(encoding='utf-8'))
CURRENT_CONFIG = REGISTRY_PATH.parent / CURRENT_PROTOCOL['config_file']


def require_current_protocol(config: Path = CURRENT_CONFIG, *, historical_protocol: bool = False) -> bool:
    """Return True only for the locked current config and scoring implementation."""
    digest = sha256_file(config)
    if historical_protocol and digest != CURRENT_PROTOCOL['config_sha256']:
        return False
    if digest != CURRENT_PROTOCOL['config_sha256']:
        raise EvaluationError(
            'Current evaluation requires object_observed_repair revision 2: '
            + CURRENT_CONFIG.name + '. Use --historical-protocol only for a labelled historical reproduction.')
    if repair_evaluator_code_hashes() != CURRENT_PROTOCOL['evaluator_code_sha256']:
        raise EvaluationError('Current scoring source must match cd260569; changed scoring needs a separately reviewed baseline')
    return True


def require_current_gt(gt: Path) -> str:
    """A copied canonical file is valid; a rebuilt or different scope is not this baseline."""
    digest = sha256_file(gt)
    for scene, locked in CURRENT_PROTOCOL['canonical_GT'].items():
        if digest == locked['file_sha256']:
            return scene
    raise EvaluationError('Current revision 2 requires one of the locked canonical GT files from the fixed 350-GT scope')


def reject_current_debug_overrides(args) -> None:
    for name in ('debug_max_distance_m', 'debug_diagnostic_max_distance_m',
                 'debug_min_valid_instance_vertices', 'debug_significant_min_vertices',
                 'debug_significant_min_gt_fraction'):
        if getattr(args, name, None) is not None:
            raise EvaluationError('Current baseline parameters are locked; debug changes require a separate historical config, --historical-protocol and separate outputs')
