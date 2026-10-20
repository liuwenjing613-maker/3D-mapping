"""Only change frozen whole-retirement decisions to the existing partial path."""
from copy import deepcopy

def partial_decision(decision):
    revised = deepcopy(decision)
    for key, action in decision.get('old_observation_plans', {}).items():
        assert action in ('whole', 'partial'), action
        if action == 'whole':
            revised['old_observation_plans'][key] = 'partial'
    return revised
