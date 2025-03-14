import torch
import torch.nn as nn
from torch.nn import functional as F
from emg2qwerty.charset import charset
from emg2qwerty.modules import (
    MultiBandRotationInvariantMLP,
    SpectrogramNorm,
    TDSConvEncoder,
)
import math

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
#seq_len = # is this variable
num_classes = charset().num_classes
mlp_features = [384, n_embd//2]
# ------------

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        attn_weights = k@q.transpose(-2, -1) * k.shape[-1] ** -0.5
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        v = self.value(x)

        return attn_weights @ v 


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_seq_len=5000):
        """
        d_model: the embedding dimension
        dropout: dropout rate
        max_seq_len: maximum sequence length you expect (e.g. max(seq_lens) from your data)
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Precompute the positional encoding table of shape (max_seq_len, d_model)
        pe = torch.zeros(max_seq_len, d_model, device=device)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)  # (max_seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # apply sin to even indices in the array
        pe[:, 1::2] = torch.cos(position * div_term)  # apply cos to odd indices in the array
        
        # Add a batch dimension: shape (1, max_seq_len, d_model)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    
    def forward(self, x, seq_lens=None):
        """
        x: Tensor of shape (batch_size, seq_len, d_model)
        seq_lens: Optional tensor of shape (batch_size,) with the true lengths (if you want masking)
        """
        # print(x.shape)
        batch_size, seq_len, _ = x.shape
        # x + positional encoding for first seq_len positions
        x = x + self.pe[:, :seq_len].requires_grad_(False)
        
        # Optionally, if you want to mask out positions beyond the true sequence lengths:
        if seq_lens is not None:
            # Create a mask of shape (batch_size, seq_len, 1)
            # For each example, positions >= true length become 0, else 1.
            mask = torch.arange(seq_len, device=x.device).unsqueeze(0) < seq_lens.unsqueeze(1)
            mask = mask.unsqueeze(2).float()
            x = x * mask  # zero out the padded positions
        
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, in_features, num_bands, electrode_channels):
        super().__init__()

        self.tokenizer = nn.Sequential(
            # # (T, N, bands=2, C=16, freq)
            # SpectrogramNorm(channels=num_bands * electrode_channels),
            # # (T, N, bands=2, n_embd)
            # MultiBandRotationInvariantMLP(
            #     in_features=in_features,
            #     mlp_features=mlp_features,
            #     num_bands=num_bands,
            # ),
            nn.Flatten(start_dim=2), 
            nn.Linear(in_features*2, n_embd))
        
        #self.positional_encodings = nn.Parameter(torch.randn(seq_len, n_embd))
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])

        self.ln_f = nn.LayerNorm(n_embd) 
        self.lm_head = nn.Linear(n_embd, num_classes)

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, seq_len, targets=None):
        # print(x.shape)
        max_len = torch.max(seq_len)
        
        pos_enc = PositionalEncoding(n_embd, dropout, max_len)
        tok_emb = self.tokenizer(x) # (B,T,C)
        tok_emb = tok_emb.permute(1, 0, 2)
        x = pos_enc(tok_emb) # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,n_classes)
        #print(logits.shape)
        return logits

class TransformerDecoder(nn.Module):
    def __init__(self, in_features, num_bands, electrode_channels):
        super().__init__()
        
        #self.positional_encodings = nn.Parameter(torch.randn(seq_len, n_embd))
        self.tokenizer = nn.Linear(in_features, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])

        self.ln_f = nn.LayerNorm(n_embd) 
        self.lm_head = nn.Linear(n_embd, num_classes)

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, targets=None):
        # print(x.shape)
        pos_enc = PositionalEncoding(n_embd, dropout) # (B,T,C)
        x = self.tokenizer(x)
        x = x.permute(1, 0, 2)
        x = pos_enc(x) # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,n_classes)
        #print(logits.shape)
        return logits