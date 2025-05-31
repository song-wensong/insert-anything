from .insert_anything_new_node import MaskOption, ReduxProcess, FillProcess, CropBack

NODE_CLASS_MAPPINGS = {
    "MaskOption": MaskOption,
    "ReduxProcess": ReduxProcess,
    "FillProcess": FillProcess,
    "CropBack": CropBack,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskOption": "InsertAnything - Mask Option",
    "ReduxProcess": "InsertAnything - Redux Process",
    "FillProcess": "InsertAnything - Fill Process",
    "CropBack": "InsertAnything - Crop Back",
}
