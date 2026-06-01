from textwrap import dedent

import numpy as np
import torch
from datasets import load_dataset
from peft import peft_model, PeftModel

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

import mt.models.finetune.finetune as finetune
from mt.models.finetune.finetune import get_device,MODEL_NAME

base_model_name = MODEL_NAME
adapter_path = "./outputs/ccs_class_2/"

device = get_device()
tokenizer = AutoTokenizer.from_pretrained(adapter_path)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    dtype=torch.float16 if device.type == "mps" else torch.float32
)

model=PeftModel.from_pretrained(base_model,adapter_path)
model.to(device)
model.eval()

prompt=dedent("""\
    I am easily distracted by sounds or movement <<
    """)

@torch.no_grad()
def eval_model_(prompt,model,tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                    )
    return outputs

outputs = eval_model_(prompt,model,tokenizer)
print(tokenizer.decode(outputs[0],skip_special_tokens=True))
# raw_dataset = load_dataset(
#         "marcelbinz/Psych-101",
#         split=f"train[:{50}]",
#     )
# lengths=[]
# for i in raw_dataset:
#     ids = tokenizer(
#         i['text'],
#         add_special_tokens=True,
#         truncation=False,
#     )['input_ids']
#     lengths.append(len(ids))
#
# print("min: ",np.min(lengths))
# print("max: ",np.max(lengths))
# print("median: ",np.median(lengths))
# print("mean: ",np.mean(lengths))
# print("p90: ",np.percentile(lengths,90))
# print(raw_dataset['text'])



