import gol
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from torch.nn.utils.rnn import pad_sequence
from torch.fft import ifft, fft
import time

from layers import GeoConv, SeqConv, Bert, VP_SDE, PointWiseFeedForward, CatConv

class DiffPOI(nn.Module):
    def __init__(self, n_user, n_poi, n_cat, geo_graph: Data):
        super(DiffPOI, self).__init__()
        self.n_user, self.n_poi, self.n_cat = n_user, n_poi, n_cat
        self.hid_dim = gol.conf['hidden']
        self.step_num = 1000
        self.local_pois = 20

        self.poi_emb = nn.Parameter(torch.empty(n_poi, self.hid_dim))
        self.distance_emb = nn.Parameter(torch.empty(gol.conf['interval'], self.hid_dim))
        self.temporal_emb = nn.Parameter(torch.empty(gol.conf['interval'], self.hid_dim))
        
        nn.init.xavier_normal_(self.poi_emb)
        nn.init.xavier_normal_(self.distance_emb)
        nn.init.xavier_normal_(self.temporal_emb)
   
        self.cat_encoder = CatEncoder(self.hid_dim)
        self.geo_encoder = GeoEncoder(n_poi, self.hid_dim, geo_graph)
        self.seq_encoder = SeqEncoder(self.hid_dim)
       
        self.ce_criteria = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(p=1-gol.conf['keepprob'])

        self.cat_layernorm = nn.LayerNorm(self.hid_dim, eps=1e-8)
        self.cat_attn_layernorm = nn.LayerNorm(self.hid_dim, eps=1e-8)
        self.cat_attn = nn.MultiheadAttention(self.hid_dim, num_heads=2, batch_first=True, dropout=0.2)
        self.cat_forward = PointWiseFeedForward(self.hid_dim, 0.2)

        self.seq_layernorm = nn.LayerNorm(self.hid_dim, eps=1e-8)
        self.seq_attn_layernorm = nn.LayerNorm(self.hid_dim, eps=1e-8)
        self.seq_attn = nn.MultiheadAttention(self.hid_dim, num_heads=2, batch_first=True, dropout=0.2)
        self.seq_forward = PointWiseFeedForward(self.hid_dim, 0.2)

        self.geo_layernorm = nn.LayerNorm(self.hid_dim, eps=1e-8)
        self.geo_attn_layernorm = nn.LayerNorm(self.hid_dim, eps=1e-8)
        self.geo_attn = nn.MultiheadAttention(self.hid_dim, num_heads=2, batch_first=True, dropout=0.2)
        self.geo_forward = PointWiseFeedForward(self.hid_dim, 0.2)
        
    
    def catProp(self, poi_embs, cat_graph):  
    
        cat_embs = self.cat_encoder.encode(poi_embs,cat_graph)
        #cat_embs = torch.stack([ i for i in cat_embs], dim=0) 
    
        if gol.conf['dropout']:
            cat_embs = self.dropout(cat_embs) 

        cat_lengths = torch.bincount(cat_graph.batch)
        cat_embs = torch.split(cat_embs, cat_lengths.cpu().numpy().tolist())

        # Self-attention
        cat_embs_pad = pad_sequence(cat_embs, batch_first=True, padding_value=0)
        qry_embs = self.cat_layernorm(cat_embs_pad)
        pad_mask = sequence_mask(cat_lengths) 

        cat_embs_pad, _ = self.cat_attn(qry_embs, cat_embs_pad, cat_embs_pad, key_padding_mask=~pad_mask)
        cat_embs_pad = cat_embs_pad + qry_embs
        cat_embs_pad = self. cat_attn_layernorm(cat_embs_pad)

        cat_encs = self.cat_forward(cat_embs_pad).mean(dim =1)
   
        return cat_encs
    
    def geoProp(self, poi_embs, seqs, seq_encs):
        geo_embs = self.geo_encoder.encode(poi_embs)
        if gol.conf['dropout']:
            geo_embs = self.dropout(geo_embs)

        seq_lengths = torch.LongTensor([seq.size(0) for seq in seqs]).to(gol.device)
        geo_seq_embs = [geo_embs[seq] for seq in seqs]

        # Target-attention
        geo_embs_pad = pad_sequence(geo_seq_embs, batch_first=True, padding_value=0)
        qry_embs = self.geo_layernorm(seq_encs.detach().unsqueeze(1))
        pad_mask = sequence_mask(seq_lengths) 

        geo_embs_pad, _ = self.geo_attn(qry_embs, geo_embs_pad, geo_embs_pad, key_padding_mask=~pad_mask)
        # geo_embs_pad = geo_embs_pad + qry_embs
        geo_embs_pad = geo_embs_pad.squeeze(1)
        geo_embs_pad = self.geo_attn_layernorm(geo_embs_pad)

        geo_encs = self.geo_forward(geo_embs_pad)
   
        return geo_encs, geo_embs

    def seqProp(self, poi_embs, seq_graph):
        seq_embs = self.seq_encoder.encode((poi_embs, self.distance_emb, self.temporal_emb), seq_graph)
        
        if gol.conf['dropout']:
            seq_embs = self.dropout(seq_embs)
        seq_lengths = torch.bincount(seq_graph.batch)
        seq_embs = torch.split(seq_embs, seq_lengths.cpu().numpy().tolist())
        
        # Self-attention
        seq_embs_pad = pad_sequence(seq_embs, batch_first=True, padding_value=0)
        qry_embs = self.seq_layernorm(seq_embs_pad)
        pad_mask = sequence_mask(seq_lengths)

        seq_embs_pad, _ = self.seq_attn(qry_embs, seq_embs_pad, seq_embs_pad, key_padding_mask=~pad_mask)
        seq_embs_pad = seq_embs_pad + qry_embs
        seq_embs_pad = self.seq_attn_layernorm(seq_embs_pad)

        seq_embs_pad = self.seq_forward(seq_embs_pad)
        seq_embs_pad = [seq[:seq_len] for seq, seq_len in zip(seq_embs_pad, seq_lengths)]

        seq_encs = torch.stack([seq.mean(dim=0) for seq in seq_embs_pad], dim=0)

        return seq_encs, seq_embs

    def sdeProp(self, x, condition, target=None):
        local_embs = x
        condition_embs = condition.detach()
        self.freq_mask_x, self.spc_mask_x = freq_mask(local_embs)
        self.sde = VP_SDE(self.hid_dim, self.freq_mask_x, self.spc_mask_x, dt=gol.conf['dt'])
        sde_encs = self.sde.reverse(local_embs, condition_embs, gol.conf['T'])
          
        fisher_loss = None
        if target is not None: # training phase   
            t_sampled = np.random.randint(1, self.step_num) / self.step_num
            mean, std = self.sde.marginal_prob(target, t_sampled)
            z = torch.randn_like(target)
            noise = idft(dft(z * self.spc_mask_x) * self.freq_mask_x).real
            perturbed_data = mean + std.unsqueeze(-1) * noise
            score = - self.sde.calc_score(perturbed_data, condition_embs)
            fisher_loss = torch.square(score + noise).mean()

        return sde_encs, fisher_loss

    def getTrainLoss(self, batch, alpha = 1/2):
        usr, pos_lbl, _, seqs, seq_graph, cur_time, cat_graph, _  = batch
        poi_embs = self.poi_emb
        if gol.conf['dropout']:
            poi_embs = self.dropout(poi_embs)
        
        seq_encs, seq_embs = self.seqProp(poi_embs, seq_graph)
        geo_encs, geo_embs = self.geoProp(poi_embs, seqs, seq_encs)
        cat_encs = self.catProp(poi_embs, cat_graph)
        sde_p, fisher_p = self.sdeProp(geo_encs, seq_encs, target = geo_embs[pos_lbl])
        sde_c, fisher_c = self.sdeProp(geo_encs, cat_encs, target = geo_embs[pos_lbl])
        
        pred_p = alpha * seq_encs @ poi_embs.T+ (1 - alpha) * sde_p  @ geo_embs.T 
        pred_c = alpha * cat_encs @ poi_embs.T+ (1 - alpha) * sde_c  @ geo_embs.T 
       
        return self.ce_criteria(pred_p * pred_c, pos_lbl), (fisher_p + fisher_c) /2
    
    def forward(self, seqs, seq_graph, cat_graph, alpha = 1/2 ):
        poi_embs = self.poi_emb

        seq_encs, seq_embs = self.seqProp(poi_embs, seq_graph)
        geo_encs, geo_embs = self.geoProp(poi_embs, seqs, seq_encs)
        cat_encs = self.catProp(poi_embs, cat_graph)
        sde_p, _= self.sdeProp(geo_encs, seq_encs)
        sde_c, _ = self.sdeProp(geo_encs, cat_encs)

        pred_p = alpha * seq_encs @ poi_embs.T+ (1 - alpha) * sde_p  @ geo_embs.T  /1024
        pred_c = alpha * cat_encs @ poi_embs.T+ (1 - alpha) * sde_c  @ geo_embs.T  /1024
        
        return pred_p * pred_c
 
class CatEncoder(nn.Module):
    def __init__(self, hid_dim):
        super(CatEncoder, self).__init__()
        self.hid_dim = hid_dim  
        self.encoder = Bert(self.hid_dim)
        self.catencoder = CatConv(self.hid_dim) 

    def encode(self, poi_embs, cat): 
        c_list = list(set(c for c_l in cat.x for c in c_l)) 
        c_dict = {c: self.encoder(c).to(gol.device) for c in c_list}  
        cat_embs = [c_dict[c] for c_l in cat.x for c in c_l] 
        cat.x = cat_embs

        return self.catencoder(poi_embs, cat)
        
class SeqEncoder(nn.Module):
    def __init__(self, hid_dim):
        super(SeqEncoder, self).__init__()
        self.hid_dim = hid_dim
        self.encoder = SeqConv(hid_dim)

    def encode(self, embs, seq_graph):
        return self.encoder(embs, seq_graph)

class GeoEncoder(nn.Module):
    def __init__(self, n_poi, hid_dim, geo_graph: Data):
        super(GeoEncoder, self).__init__()
        self.n_poi, self.hid_dim = n_poi, hid_dim
        self.gcn_num = gol.conf['num_layer']

        edge_index, _ = add_self_loops(geo_graph.edge_index)
        dist_vec = torch.cat([geo_graph.edge_attr, torch.zeros((n_poi,)).to(gol.device)])
        dist_vec = torch.exp(-(dist_vec ** 2))
        self.geo_graph = Data(edge_index=edge_index, edge_attr=dist_vec)

        self.act = nn.LeakyReLU()
        self.geo_convs = nn.ModuleList()
        for _ in range(self.gcn_num):
            self.geo_convs.append(GeoConv(self.hid_dim, self.hid_dim))

    def encode(self, poi_embs):
        layer_embs = poi_embs
        geo_embs = [layer_embs]
        for conv in self.geo_convs:
            layer_embs = conv(layer_embs, self.geo_graph)
            layer_embs = self.act(layer_embs)
            geo_embs.append(layer_embs)
        geo_embs = torch.stack(geo_embs, dim=1).mean(1)
        return geo_embs

def sequence_mask(lengths, max_len=None):
    lengths_shape = lengths.shape  # torch.size() is a tuple
    lengths = lengths.reshape(-1)

    batch_size = lengths.numel()
    max_len = max_len or int(lengths.max())
    lengths_shape += (max_len,)

    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .unsqueeze(0).expand(batch_size, max_len)
            .lt(lengths.unsqueeze(1))).reshape(lengths_shape)

def idft(x):
    return ifft(x, dim=1).to(gol.device)
    
def dft(x):
    return fft(x, dim=1).to(gol.device)
    
def freq_mask(x):
    x = x.to(gol.device)
    avg_freq_amp = 0
    avg_freq_amp = torch.abs(dft(x)).mean(dim=0) ** 2
    avg_freq_amp /= x.size(0)
    avg_freq_amp = torch.log(1 + avg_freq_amp)

    avg_spc_amp = x.mean(dim=0) ** 2
    avg_spc_amp /=x.size(0)

    return avg_freq_amp, avg_spc_amp
   
