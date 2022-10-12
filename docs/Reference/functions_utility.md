# functions_handling

This module provides a way to add custom functions to the model.

To do this, instead of calling ``function(*args, **kwds)``, use ``add_to_model(function, *args, **kwds)``.

Example for using ``torch.concat``:

```python
from pytorch_functional import Input
from pytorch_functional.functions_utility import add_to_model
import torch


v1 = Input((10,))
v2 = Input((20,))
output = add_to_model(torch.concat, tensors=(v1, v2), dim=1)
output
```

```
<SymbolicTensor at 0x7f31dafd2460; child of 2; parent of 0>
```

This will work for most of the use cases, even if Symbolic Tensors 
are hidden in nested tuples, lists or dicts, but you should know that there's
a small call overhead involved.

Recommended way to use a custom function is to write yourself an ``nn.Module`` that does
the same as the function. Then you can use it as usually and Pytorch Functional will be overhead free!

::: pytorch_functional.functions_handling
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
