# 🦙 Llama Trainer Utility

[![Upload to PyPi](https://github.com/Riccorl/llama-trainer/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Riccorl/llama-trainer/actions/workflows/python-publish.yml)

A "just few lines of code" utility for fine-tuning (not only) Llama models.

To install:

```bash
pip install llama-trainer
```

### Training and Inference

#### Training

```python
from llama_trainer import LlamaTrainer
from datasets import load_dataset

dataset = load_dataset("timdettmers/openassistant-guanaco")

# define your instruction-based sample
def to_instruction_fn(sample):
    return sample["text"]

formatting_func = to_instruction_fn

output_dir = "llama-2-7b-hf-finetune"
llama_trainer = LlamaTrainer(
    model_name="meta-llama/Llama-2-7b-hf", 
    dataset=dataset, 
    formatting_func=formatting_func,
    output_dir=output_dir
)
llama_trainer.train()
```

#### Inference

```python
from llama_trainer import LlamaInfer
import transformers as tr


llama_infer = LlamaInfer(output_dir)

prompt = "### Human: Give me some output!### Assistant:"
print(llama_infer(prompt))
```
