
# Multiview Canonicalizer

This module implements the Conditional Multiview Canonicalizer.

## Usage

### Frozen Energy Canonicalizer (Approach 1)

```python
from canonicalizer.models.frozen_energy import FrozenEnergyCanonicalizer

model = FrozenEnergyCanonicalizer(
    dino_model='dinov2_vitb14',
    num_iterations=5
)

warped_source, g_star = model(source_image, target_image)
```

### Wrappers

To use with an existing matcher (e.g. RoMa):

```python
from canonicalizer.wrappers.roma_wrapper import RoMaWrapper

# Assuming you have a roma_model instance
wrapper = RoMaWrapper(model, roma_model)

matches = wrapper({
    'image0': source, 
    'image1': target
})
```

## Structure

- `core/`: Sim(2) Lie group and warping operations.
- `models/`: Neural network architectures (Frozen Energy, etc.).
- `wrappers/`: Adapters for standard matchers.
