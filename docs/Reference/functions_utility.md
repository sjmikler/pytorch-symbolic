# functions_utility

This module provides a way to add custom functions to the model.

To register a function in your model, instead of doing

- ``function(*args, **kwds)``

do this:

- ``add_to_model(function, *args, **kwds)``.

For this to work, there must be at least one SymbolicTensor among ``*args, **kwds``.

Example for using ``torch.concat``:

```python
from pytorch_symbolic import Input
from pytorch_symbolic.functions_utility import add_to_graph
import torch

v1 = Input((10,))
v2 = Input((20,))
output = add_to_graph(torch.concat, tensors=(v1, v2), dim=1)
output
```

```
<SymbolicTensor at 0x7f31dafd2460; child of 2; parent of 0>
```

This will work for most of the use cases, even if Symbolic Tensors
are hidden in nested tuples, lists or dicts, but you should know that there's
a small call time overhead every time you register a custom function in the model.

Recommended way to use a custom functions is to write yourself an ``nn.Module`` that does
the same as the function. Then you can use it as usually and Pytorch Functional will be overhead free!

::: pytorch_symbolic.functions_utility
	options:
		show_source: false
		heading_level: 2
		show_root_heading: true
		members_order: source
		show_object_full_path: false
		docstring_section_style: table
		show_signature_annotations: true
		separate_signature: true
		annotations_path: brief
		merge_init_into_class: true
		show_root_full_path: true
