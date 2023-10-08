import gc
from typing import Iterable, Optional, List, Union

import torch
import torch.nn.functional as F
from loguru import logger