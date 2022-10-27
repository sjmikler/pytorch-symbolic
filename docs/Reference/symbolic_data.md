# SymbolicTensor

A ``SymbolicTensor`` and ``Input`` are similar to ``torch.Tensor`` object, but it is used only when 
defining the graph, not to perform actual computations. 
You should use it to register new layers in your computation graph and later to create the model.

``SymbolicTensor`` and ``Input`` support slicing and common methods, e.g. `tensor.t()` for transposition.

Examples:

1. ``nn.Linear(10, 10)(symbolic)``
2. ``symbolic(nn.Linear(10, 10))`` - equivalent to 1.
3. ``model = SymbolicModel(inputs=(symbolic_1, symbolic_2), outputs=symbolic_3)``

::: pytorch_symbolic.Input
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

::: pytorch_symbolic.SymbolicData
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

::: pytorch_symbolic.SymbolicTensor
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

::: pytorch_symbolic.SymbolicTuple
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
