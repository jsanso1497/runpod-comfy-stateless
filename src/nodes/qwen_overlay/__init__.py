"""ComfyUI registration for full-resolution Qwen edit overlays."""
from workbench.qwen_overlay import WBOverlayProtection, WBOverlayPrepare, WBOverlayExport
NODE_CLASS_MAPPINGS = {c.__name__: c for c in (WBOverlayProtection, WBOverlayPrepare, WBOverlayExport)}
NODE_DISPLAY_NAME_MAPPINGS = {
    'WBOverlayProtection': 'Protect Face / Element with SAM (Original Image)',
    'WBOverlayPrepare': 'Prepare Full-resolution Edit Overlay',
    'WBOverlayExport': 'Review Masks / Export Transparent Overlay + Composite',
}
