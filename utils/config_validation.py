#!/usr/bin/env python

import re
from typing import Union, List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field, validator

class EarlyStoppingConfig(BaseModel):
    monitor: Union[str, List[str]]
    min_delta: Union[float, List[float], str]
    patience: Union[int, List[int], str]
    verbose: int = 0
    restore_best_weights: Union[bool, List[bool]] = False
    start_from_epoch: Optional[Union[int, List[int], str]] = 0
    mode: str = Field(..., pattern="^(min|max)$")

    @validator('min_delta', 'patience', 'start_from_epoch')
    def validate_range_format(cls, v):
        pattern = r'^\(\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*\)$'
        if isinstance(v, str) and not re.match(pattern, v):
            raise ValueError("thresholds must be a string in the form '(float, float, float)'")
        return v


class ReduceLrConfig(BaseModel):
    factor: Union[float, List[float], str] # = Field(..., gt=0.0, lt=1.0)
    patience: Union[int, List[int], str] # = Field(..., ge=1)
    min_delta: Union[float, List[float], str]

    @validator('factor', 'patience', 'min_delta')
    def validate_range_format(cls, v):
        pattern = r'^\(\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*\)$'
        if isinstance(v, str) and not re.match(pattern, v):
            raise ValueError("thresholds must be a string in the form '(float, float, float)'")
        return v


class TrainingConfig(BaseModel):
    model_name: str
    demo: bool = False
    pretrained: Optional[Union[str, List[str]]]
    test_prop: Union[float, List[float], str] 
    val_prop: Union[float, List[float], str] 
    val_prop_opt: Union[float, List[float], str] 
    epochs: Union[int, List[int], str]
    steps_per_epoch: Union[int, List[int], str]
    learning_rate: Union[float, List[float], str]
    augment: Union[bool, List[bool]]
    random_seed: Union[int, List[int]]
    early_stopping: Optional[EarlyStoppingConfig]
    train_reduce_lr: Optional[ReduceLrConfig]
    base_dir: Optional[str]
    data_dir: Optional[str]

    @validator('test_prop', 'val_prop', 'val_prop_opt', 'epochs', 'steps_per_epoch', 'learning_rate')
    def validate_range_format(cls, v):
        pattern = r'^\(\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*\)$'
        if isinstance(v, str) and not re.match(pattern, v):
            raise ValueError("thresholds must be a string in the form '(float, float, float)'")
        return v
