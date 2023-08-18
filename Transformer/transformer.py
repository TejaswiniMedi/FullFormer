import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import numpy as np
from Transformer.minGPT import GPT
from models.Global import NDF    


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer,self).__init__()
        self.sos_token = 0
        self.checkpoint_path = "/home/Tejaswini/PycharmProjects/NDF/models/experiments/chairs_final/checkpoint_chair/"
        self.model = NDF().cuda()
        self.checkpoint = "chairs_final"
        self.model = self.load_model(self.checkpoint)
        transformer_config = {
            "vocab_size" : 2048*3,        
            "block_size" : 1024,                               
            "n_layer" : 12,
            "n_embd": 1024                                           
        }
        self.transformer = GPT(**transformer_config)
        self.pkeep = 0.3     # No. of latents to keep in the sequence

    #@staticmethod
    def load_model(self,checkpoint):
        if checkpoint is None:
            if len(checkpoints) == 0:
                print('No checkpoints found at {}'.format(self.checkpoint_path))
                return 0, 0
            checkpoints = [os.path.splitext(os.path.basename(path))[0].split('_')[-1] for path in checkpoints]
            checkpoints = np.array(checkpoints, dtype=float)
            checkpoints = np.sort(checkpoints)
        else:
            path = self.checkpoint_path + '{}.tar'.format(checkpoint)
        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        return self.model
    
    @torch.no_grad()
    def encode_to_z(self,x):
        encoding_q,indices,loss_q = self.model.encoder(x)
        indices = indices.view(encoding_q.shape[0], -1)
        return encoding_q,indices

    def z_to_udf(self,samples,indices,p1=8,p2=8,p3=8):      
        ix_to_vectors = self.model.embedding_1(indices).reshape(indices.shape[0],-1,p1,p2,p3)
        udf = self.model.decoder(samples,ix_to_vectors)
        return udf
    
    def forward(self,x):
        device = torch.device("cuda")
        _,indices = self.encode_to_z(x)
        sos_tokens = torch.ones(x.shape[0],1)*self.sos_token
        sos_tokens = sos_tokens.long().to("cuda")
        mask = torch.bernoulli(self.pkeep*torch.ones(indices.shape,device=device))
        mask = mask.round().to(dtype=torch.int64)
        random_indices = torch.randint_like(indices,self.transformer.config.vocab_size).long()
        new_indices = mask * indices + (1 - mask) * random_indices
        new_indices = new_indices.long()
        new_indices = torch.cat((sos_tokens,new_indices),dim=1)
        target = indices
        new_indices = new_indices[:, :-1]
        logits, _ = self.transformer(new_indices)
        return logits,target
    
    def top_k_logits(self,logits,k):
        v,ix = torch.topk(logits,k)
        out = logits.clone()
        out[out < v[...,[-1]]] = -float("inf")
        return out
    
    def sample(self,x,c,steps,temperature =1.0,top_k = 100):
        self.transformer.eval()
        x = torch.cat((c,x),dim=1).cuda()
        for k in range(steps):
            logits,_ = self.transformer(x)
            logits = logits[:,-1,:]/temperature
            if top_k is not None:
                logits = self.top_k_logits(logits,top_k)
            probs = F.softmax(logits,dim=1)
            ix = torch.multinomial(probs,num_samples=1)
            x = torch.cat((x,ix),dim=1)
        x = x[:,c.shape[1]:]
        self.transformer.train()
        return x

    def log_udf(self,p,x):
        q, indices = self.encode_to_z(x)
        sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to("cuda")
        start_indices = indices[:, :0].long()
        full_indices = self.sample(start_indices, sos_tokens, steps=indices.shape[1])
        full_sample = self.z_to_udf(p,full_indices)
        return full_sample
