"""FastAPI dependencies and global variables."""

import src as tfcv
import outfit_oracle_fastapi as tfcv_fapi

PRED_MODEL, DEVICE = tfcv.modelling.model_utils.load_model(
    tfcv_fapi.config.SETTINGS.PRED_MODEL_PATH,
    tfcv_fapi.config.SETTINGS.USE_CUDA,
    tfcv_fapi.config.SETTINGS.USE_MPS,
)
