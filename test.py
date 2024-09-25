import torch
import tiktoken
from train_gpt import GPT,GPTConfig
import torch.nn.functional as F
model=GPT(GPTConfig(vocab_size=50304))
enc=tiktoken.get_encoding('gpt2')
device='cuda' if torch.cuda.is_available() else 'cpu'
checkpoint=torch.load('gpt_weight_99.pth')
model.load_state_dict(checkpoint['model_state'])
state=model.state_dict()
print(state['transformer.wte.weight'])
model=model.to(device)
model.eval()
num_return_sequences = 4
max_length = 32
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
xgen = tokens.to(device)
sample_rng = torch.Generator(device=device)
sample_rng.manual_seed(42)

while xgen.size(1) < max_length:
    with torch.no_grad():
        
        logits, loss = model(xgen)
        logits = logits[:, -1, :] 
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
        xcol = torch.gather(topk_indices, -1, ix) 
        xgen = torch.cat((xgen, xcol), dim=1)
for i in range(num_return_sequences):
    tokens = xgen[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(f"rank sample {i}: {decoded}")
