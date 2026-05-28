from torch.utils.data import DataLoader,Dataset
import torch
from transformers import AutoTokenizer

class TextDataset(Dataset):
    def __init__(self,texts,tokenizer,max_len=512):
        self.tokens = []
        for text in texts:
            token_ids = tokenizer(
                text,
                truncation=True,
                max_length=max_len-1,
            )["input_ids"]
            token_ids = token_ids + [tokenizer.eos_token_id]
            token_ids = torch.tensor(token_ids,dtype=torch.long)
            if len(token_ids)>=3:
                self.tokens.append(token_ids)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        cur_text = self.tokens[idx]
        input_ids = cur_text[:-1]
        target_ids = cur_text[1:]
        return input_ids, target_ids


def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-100)

    return inputs, targets



tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
vocab_size = tokenizer.vocab_size

texts = [
        "Hello world. This is a tiny language model.",
        "The transformer predicts the next token.",
        "Deep learning models learn from data.",
        "PyTorch makes neural networks easier to build."
    ]

dataset = TextDataset(texts, tokenizer)

dataloader = DataLoader(dataset,
                        batch_size=8,
                        shuffle=True,
                        collate_fn=collate_fn
                        )
print("dataset lenght:" ,len(dataset))
sample=dataset[1]
print(f"sample:{sample}")
print(tokenizer.decode([13]))
for batch in dataloader:
    print(f"batch:{batch}")
    break