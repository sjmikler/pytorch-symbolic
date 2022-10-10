# SymbolicTensor

A ``SymbolicTensor`` is similar to ``torch.Tensor`` object, but it is used only during model creation.

You should use it to register new layers in the computation graph and later to create the model.

Examples:

1. ``nn.Linear(10, 10)(symbolic_tensor)``
2. ``symbolic_tensor(nn.Linear(10, 10))`` - equivalent to 1.
3. ``model = FunctionalModel(inputs=(symbolic_1, symbolic_2), outputs=symbolic_3)``

::: pytorch_functional.SymbolicTensor
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
