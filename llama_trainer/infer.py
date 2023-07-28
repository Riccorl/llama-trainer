from typing import Any, Dict, Optional, Union
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, PreTrainedModel, StoppingCriteriaList
from llama_trainer.utils import is_package_available
from llama_trainer.utils.stopping_criteria import SentinelTokenStoppingCriteria


class LlamaInfer:
    def __init__(
        self,
        model_name_or_path: Union[str, PreTrainedModel],
        low_cpu_mem_usage: bool = True,
        torch_dtype: torch.dtype = torch.float16,
        load_in_4bit: bool = True,
    ) -> None:
        # just to be sure
        if is_package_available("flash_attn"):
            from llama_trainer.utils.llama_patch import unplace_flash_attn_with_attn

            unplace_flash_attn_with_attn()

        if isinstance(model_name_or_path, str):
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_name_or_path,
                low_cpu_mem_usage=low_cpu_mem_usage,
                torch_dtype=torch_dtype,
                load_in_4bit=load_in_4bit,
            )
        else:
            model = model_name_or_path

        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)

        self.device = self.model.device

    def __call__(
        self,
        prompt: str,
        do_sample: bool = True,
        max_new_tokens: int = 100,
        top_p: float = 0.9,
        temperature: float = 0.9,
        stop_token: Optional[Union[str, int]] = None,
        generation_kwargs: Dict[str, Any] = None,
    ) -> Any:
        input_ids = self.tokenizer(
            prompt, return_tensors="pt", truncation=True
        ).input_ids

        # check if device of the model is cuda
        input_ids = input_ids.to(self.model.device)

        if generation_kwargs is None:
            generation_kwargs = {}

        stopping_criteria_list: Optional[StoppingCriteriaList] = None

        if stop_token is not None:
            if isinstance(stop_token, str):
                stop_token = self.tokenizer(
                    stop_token, add_special_tokens=False, return_tensors="pt"
                ).input_ids.to(self.model.device)

            stopping_criteria_list = StoppingCriteriaList(
                [
                    SentinelTokenStoppingCriteria(
                        sentinel_token_ids=stop_token, starting_idx=0
                    )
                ]
            )
        generation_kwargs = {
            "input_ids": input_ids,
            "do_sample": do_sample,
            "max_new_tokens": max_new_tokens,
            "top_p": top_p,
            "temperature": temperature,
            "stopping_criteria": stopping_criteria_list,
            **generation_kwargs,
        }
        outputs = self.model.generate(**generation_kwargs)
        outputs = outputs.detach().cpu().numpy()
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][
            len(prompt) :
        ]
        return decoded
