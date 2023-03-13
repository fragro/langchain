"""Wrapper around HuggingFace Model locally run on either GPU or CPU."""
import importlib.util
import logging
from typing import Any, List, Mapping, Optional

from pydantic import BaseModel, Extra

from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens


logger = logging.getLogger()

class HuggingFaceModel(LLM, BaseModel):
    """Wrapper around HuggingFace Pipeline API.

    To use, you should have the ``transformers`` python package installed.

    Supports direct access to the `generate` method of the model.

    Example passing model and tokenizer in directly:
        .. code-block:: python

            from langchain.llms import HuggingFaceModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_id = "gpt2"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)
            model_kwargs = {"max_new_tokens": 10}
            hf = HuggingFaceModel(model=model, tokenizer=tokenizer, model_kwargs=_model_kwargs)
    """

    model: Any  #: :meta private:
    tokenizer: Any  #: :meta private:
    device: int = -1
    model_kwargs: Optional[dict] = None
    """Key word arguments to pass to the model."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            **{"model": self.model},
            **{"tokenizer": self.tokenizer},
            **{"device": self.device},
            **{"model_kwargs": self.model_kwargs},
        }

    @property
    def _llm_type(self) -> str:
        return "huggingface_model"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, model_kwargs: Optional[dict] = {}) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        if importlib.util.find_spec("torch") is not None:
            import torch

            cuda_device_count = torch.cuda.device_count()
            if self.device < -1 or (self.device >= cuda_device_count):
                raise ValueError(
                    f"Got device=={self.device}, "
                    f"device is required to be within [-1, {cuda_device_count})"
                )
            if self.device < 0 and cuda_device_count > 0:
                logger.warning(
                    "Device has %d GPUs available. "
                    "Provide device={deviceId} to `from_model_id` to use available"
                    "GPUs for execution. deviceId is -1 (default) for CPU and "
                    "can be a positive integer associated with CUDA device id.",
                    cuda_device_count,
                )
            input_ids = input_ids.to('cuda') if self.device > 0 else torch.device("cpu")
        response = self.model.generate(
            input_ids=input_ids,
            pad_token_id=self.tokenizer.eos_token_id,
            **self.model_kwargs
        )
        text = self.tokenizer.decode(response[0])
        if stop is not None:
            # This is a bit hacky, but I can't figure out a better way to enforce
            # stop tokens when making calls to huggingface_hub.
            text = enforce_stop_tokens(text, stop)
        return text
