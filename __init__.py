from .fitdit_nodes import FitDiTLoader, FitDiTMaskGenerator, FitDiTTryOn

NODE_CLASS_MAPPINGS = {
    "FitDiTLoader": FitDiTLoader,
    "FitDiTMaskGenerator": FitDiTMaskGenerator, 
    "FitDiTTryOn": FitDiTTryOn
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FitDiTLoader": "Load FitDiT Model",
    "FitDiTMaskGenerator": "Generate FitDiT Mask",
    "FitDiTTryOn": "FitDiT Try-On"
} 