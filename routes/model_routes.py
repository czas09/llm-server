from fastapi import APIRouter
from loguru import logger

from protocol import ModelCard, ModelList
from config import config


model_router = APIRouter()


@model_router.get("/models")
async def show_available_models() -> ModelList: 
    logger.info("当前模型服务：")
    logger.info("    {}".format(config.MODEL_NAME))
    return ModelList(data=[ModelCard(id=config.MODEL_NAME, root=config.MODEL_NAME)])