import torch.optim
import torch
from src.mt.models.LLM import decoder_trainable
from src.mt.data.dataset import TextDataset,collate_fn
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif hasattr(torch.backends,"mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else :
    device = torch.device("cpu")

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
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=8,
                                         shuffle=True,
                                         collate_fn=collate_fn
                                         )

model= decoder_trainable.TrainableDecoder(vocab_size, d_model=64, d_ff=128, num_heads=8, max_seq_len=512).to(device)
optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5)



num_epochs = 500
train_losses = []
val_losses = []

def train_one_epoch(model, optimizer, dataloader, device):
    model.train()
    total_loss = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)

        loss = torch.nn.functional.cross_entropy(logits.reshape(-1,logits.size(-1)),labels.reshape(-1),ignore_index=-100)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        loss= torch.nn.functional.cross_entropy(logits.reshape(-1,logits.size(-1)),labels.reshape(-1),ignore_index=-100)
        total_loss += loss.item()
    return total_loss / len(dataloader)

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, optimizer, dataloader, device)
    val_loss_on_train = evaluate(model, dataloader, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss_on_train)

torch.save(model.state_dict(),"model.pt")

@torch.no_grad()
def generate(model,tokenizer,prompt,device,max_new_tokens=20):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    for _ in range(max_new_tokens):
        logits = model(input_ids)
        next_token_logits = logits[:,-1,:]
        next_token = torch.argmax(next_token_logits,dim=-1,keepdim=True)
        input_ids=torch.cat([input_ids,next_token],dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(input_ids[0])

print(generate(model, tokenizer, "The transformer", device))
print(generate(model, tokenizer, "Deep learning", device))
print(generate(model, tokenizer, "PyTorch", device))

plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss on Train")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()

