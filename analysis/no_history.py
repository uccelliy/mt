from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset
import pandas as pd
import argparse
import torch
from mt.models.llm.backends import get_is_quantized, load_causal_lm, set_is_quantized

if __name__ == '__main__':
      parser = argparse.ArgumentParser()
      parser.add_argument("--model", type=str, required=True)
      args = parser.parse_args()

      task_names = [
            "peterson2021using/prompts_zeroshot.jsonl",
            "ruggeri2022globalizability/prompts_zeroshot.jsonl",
      ]
      model, tokenizer = load_causal_lm(args.model)
      l_id = tokenizer(" <<").input_ids[1:]
      r_id = tokenizer(">>").input_ids[1:]
      collator = DataCollatorForCompletionOnlyLM(response_template=l_id, instruction_template=r_id, tokenizer=tokenizer)
      is_quantized = get_is_quantized(model)

      data = []
      with torch.no_grad():
            for i, task_name in enumerate(task_names):
                  dataset = load_dataset(
                        'json',
                        data_files={
                              'test': [task_name],
                        }
                  )

                  set_is_quantized(model, False)
                  training_args = TrainingArguments(
                        output_dir="eval_"+str(i),
                        per_device_eval_batch_size=1,
                        report_to="none"
                  )
                  trainer = SFTTrainer(
                        model=model,
                        tokenizer=tokenizer,
                        args=training_args,
                        train_dataset=dataset['test'],
                        eval_dataset=dataset['test'],
                        dataset_text_field="text",
                        max_seq_length=32768,
                        data_collator=collator,
                  )
                  set_is_quantized(model, is_quantized)
                  result = trainer.evaluate()

                  print(task_name, flush=True)
                  print(result, flush=True)
                  data.append([task_name, result['eval_loss']])
            df = pd.DataFrame(data, columns=['task', str(args.model)])
            print(df, flush=True)
            df.to_csv('results/' + args.model.replace('/', '-') +  '.csv')
