import torch
from datasets import load_dataset
import argparse
from mt.models.llm_backend import load_causal_lm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    layers = [0, 10, 20, 30, 40] # change as needed

    dataset = load_dataset(
        'json',
        data_files="feher2023rethinking/prompts.jsonl"
    )

    model, tokenizer = load_causal_lm(args.model, for_inference=True)

    left_token = " <<"
    l_id = tokenizer(left_token).input_ids[-1]
    print(l_id)

    zero_token = "0"
    zero_id = tokenizer(zero_token).input_ids[-1]
    print(zero_id)

    one_token = "1"
    one_id = tokenizer(one_token).input_ids[-1]
    print(one_id)

    # loop over prompts
    for i, prompt in enumerate(dataset['train']):
        with torch.no_grad():
            tokenized_prompt = tokenizer(prompt['text'], return_tensors='pt')
            tokenized_prompt.input_ids = tokenized_prompt.input_ids[:, :32768]
            relevant_tokens_l = (tokenized_prompt.input_ids == l_id).squeeze()
            relevant_tokens_0 = (tokenized_prompt.input_ids == zero_id).squeeze()
            relevant_tokens_1 = (tokenized_prompt.input_ids == one_id).squeeze()

            relevant_tokens = torch.logical_or(relevant_tokens_l, torch.logical_or(relevant_tokens_0, relevant_tokens_1))
            print(relevant_tokens.float().sum())
            print(tokenized_prompt.input_ids.shape)
            result_model = model(tokenized_prompt.input_ids, output_hidden_states=True, return_dict=True, max_new_tokens=1)

            representations = []
            for j, layer in enumerate(result_model['hidden_states']):
                if j in layers:
                    representations.append(layer[0, relevant_tokens, :].to("cpu").half())
            torch.save(representations, 'results/model=' + args.model.replace('/', '-') + '_participant=' + str(i) + '.pth')
