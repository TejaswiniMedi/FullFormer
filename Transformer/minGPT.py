import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class GPTConfig:
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    def __init__(self,vocab_size,block_size,**kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self,k,v)
class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.key = nn.Linear(config.n_embd,config.n_embd)
        self.query = nn.Linear(config.n_embd,config.n_embd)
        self.value = nn.Linear(config.n_embd,config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd,config.n_embd)
        mask = torch.tril(torch.ones(config.block_size,config.block_size))
        if hasattr(config, "n_unmasked"):
            mask[:config.n_unmasked, :config.n_unmasked] = 1
        self.register_buffer("mask",mask.view(1,1,config.block_size,config.block_size))
        self.n_head = config.n_head
    def forward(self,x,layer_past=None):
        B,T,C = x.size()
        #print(B,T,C)
        k = self.key(x).view(B,T,self.n_head, C//self.n_head).transpose(1,2)
        q = self.query(x).view(B,T,self.n_head, C//self.n_head).transpose(1,2)
        v = self.value(x).view(B,T,self.n_head, C//self.n_head).transpose(1,2)
        present = torch.stack((k,v))
        if layer_past is not None:
            past_key,past_value = layer_past
            k = torch.cat((past_key,k),dim = -2)
            v = torch.cat((past_value, v), dim=-2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if layer_past is None:
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        y = self.resid_drop(self.proj(y))
        return y,present  # TODO: check that this does not break anything , present

class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4*config.n_embd,config.n_embd),
            nn.Dropout(config.resid_pdrop)
        )
    def forward(self,x,layer_past = None,return_present = False):
        if return_present:
            assert not self.training
        attn,present = self.attn(self.ln1(x),layer_past = layer_past)
        x = x + attn
        x = x + self.mlp(self.ln2(x))
        if layer_past is not None or return_present:
            print("yaaaay!")
            return x,present
        return x

class GPT(nn.Module):
    def __init__(self,vocab_size,block_size,n_layer=12,n_head = 8,n_embd=256,embd_pdrop=0.,
                 resid_pdrop = 0.,attn_pdrop= 0.,n_unmasked=0):
        super().__init__()
        config = GPTConfig(vocab_size=vocab_size, block_size=block_size,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                           n_unmasked=n_unmasked)
        self.tok_emb = nn.Embedding(config.vocab_size,config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1,config.block_size,config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.block_size = config.block_size
        self.apply(self.__init__weights)
        self.config = config
        
    def get_block_size(self):
        return self.block_size
    
    def __init__weights(self,module):
        if isinstance(module,(nn.Linear,nn.Embedding)):
            module.weight.data.normal_(mean=0.0,std=0.05)
            if isinstance(module,nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module,nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(self, idx,embeddings = None):
        token_embeddings = self.tok_emb(idx)
        if embeddings is not None:
            token_embeddings = torch.cat((embeddings,token_embeddings),dim=1)
        t = token_embeddings.shape[1]

        assert  t <= self.block_size , "Cannot forward, model block size is exhausted."
        position_embeddings = self.pos_emb[:,:t,:]
        th = token_embeddings + position_embeddings
        x = self.drop(token_embeddings + position_embeddings)  # (8,964,1024)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits,None
