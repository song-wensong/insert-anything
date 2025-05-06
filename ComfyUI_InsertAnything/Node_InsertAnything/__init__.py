from .insert_anything_node import InsertanythingLoader, InsertanythingImageProcessor, InsertanythingInferencer, MaskOption

NODE_CLASS_MAPPINGS = {
    "InsertAnythingLoader": InsertanythingLoader,
    "InsertAnythingImageProcessor": InsertanythingImageProcessor,
    "InsertAnythingInferencer": InsertanythingInferencer,
    "MaskOption": MaskOption,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InsertAnythingLoader": "InsertAnything - Load Models",
    "InsertAnythingImageProcessor": "InsertAnything - Preprocess Images",
    "InsertAnythingInferencer": "InsertAnything - Run Inference",
    "MaskOption": "InsertAnything - Mask Option",
}
