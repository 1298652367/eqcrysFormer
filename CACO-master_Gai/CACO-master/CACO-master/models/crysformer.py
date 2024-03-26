"""Atomistic LIne Graph Neural Network.

A prototype crystal line graph network dgl implementation.
"""
from typing import Tuple, Union
import torch
import dgl
import dgl.function as fn
from dgl.nn import AvgPooling
from torch import nn
from dgl.nn.functional import edge_softmax
from functools import partial
from .layers import RBFExpansion
from .nets.graph_attention_transformer_md17 import GraphAttentionTransformerMD17

_RESCALE = True
_MAX_ATOM_TYPE = 5
_AVG_DEGREE = 15.57930850982666
_USE_BIAS = True

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act_layer=nn.ReLU):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = act_layer(inplace=True)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class ConditionalAttention(nn.Module):
    def __init__(self, dim, num_heads, use_bias, drop):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv_linear = nn.Linear(dim, dim*3, bias=use_bias)
        self.c_linear = nn.Linear(dim, dim, bias=use_bias)
        # self.gated = nn.ReLU(inplace=True)
        self.gated = nn.Tanh()

        self.h_proj = nn.Linear(dim, dim)
        self.e_proj = nn.Linear(dim, dim)

    def forward(self, g, h, e):
        g = g.local_var()
        qkv = self.qkv_linear(h).reshape(-1, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(1)
        c = self.c_linear(e).reshape(-1, self.num_heads, self.head_dim)
        c = self.gated(c)

        g.ndata['q'] = q
        g.ndata['k'] = k
        g.ndata['v'] = v
        g.apply_edges(fn.u_mul_v('k', 'q', 'score'))

        score = g.edata.pop('score') * c
        attn = score.sum(-1, keepdims=True) * self.scale

        g.edata['attn'] = edge_softmax(g, attn)
        g.update_all(fn.u_mul_e('v', 'attn', 'v'), fn.sum('v', 'h'))

        h_out = self.h_proj(g.ndata.pop('h').reshape(-1, self.dim))
        e_out = self.e_proj(score.reshape(-1, self.dim))

        return h_out, e_out, score.mean(dim=1)


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, use_bias, drop):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.q_linear = nn.Linear(dim, dim, bias=use_bias)
        self.kv_linear = nn.Linear(dim, dim*2, bias=use_bias)
        self.h_proj = nn.Linear(dim, dim)

    def forward(self, g, h, e):
        g = g.local_var()
        q = self.q_linear(h).reshape(-1, self.num_heads, self.head_dim)
        kv = self.kv_linear(e).reshape(-1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(1)

        g.ndata['q'] = q
        g.edata['k'] = k
        g.apply_edges(fn.v_dot_e('q', 'k', 'attn'))

        attn = g.edata.pop('attn') * self.scale
        attn = edge_softmax(g, attn)
        g.edata['v'] = attn * v
        g.update_all(fn.copy_e('v', 'm'), fn.sum('m', 'h'))

        h_out = self.h_proj(g.ndata.pop('h').reshape(-1, self.dim))
        return h_out


class CrysformerLayer(nn.Module):
    def __init__(self, dim, num_heads, use_bias=False, mlp_ratio=2., drop=0.,
                 act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.crossattention = CrossAttention(dim, num_heads, use_bias, drop)
        self.norm1 = norm_layer(dim)

        self.condattention = ConditionalAttention(dim, num_heads, use_bias, drop)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)

        self.mlp1 = MLP(dim, int(dim * mlp_ratio), dim, 2, act_layer)
        self.mlp2 = MLP(dim, int(dim * mlp_ratio), dim, 2, act_layer)
        self.norm4 = norm_layer(dim)
        self.norm5 = norm_layer(dim)

    def forward(self, g, h, e):
        g = g.local_var()

        h = self.norm1(h + self.crossattention(g, h, e))
        h_, e_, score = self.condattention(g, h, e)
        #print('condattention2', score.shape, h_.shape, e_.shape)

        h = self.norm2(h + h_)
        e = self.norm3(e + e_)

        h = self.norm4(h + self.mlp1(h))
        e = self.norm5(e + self.mlp2(e))

        return h, e, score


class DynamicConnection(nn.Module):
    def __init__(self, threshold=1., t=1., act_layer=nn.ReLU):
        super().__init__()
        self.th = threshold
        self.T = t

    def forward(self, g, score, y, lg=None, z=None):
        def edge_norm_lt_th(edges):
            return (edges.data.pop('score').norm(dim=-1) / self.T) < self.th
        # import pdb; pdb.set_trace()
        g.edata['y'] = y
        g.edata['score'] = score
        drop_idx = g.filter_edges(edge_norm_lt_th)
        g.remove_edges(drop_idx)
        y = g.edata.pop('y')
        if lg is not None:
            lg.edata['z'] = z
            lg.remove_nodes(drop_idx)
            z = lg.edata.pop('z')
        return y, z


class CrysformerBlock(nn.Module):
    """Line graph update."""

    def __init__(self, dim, num_heads, use_bias=False, mlp_ratio=2., drop=0.,
                 act_layer=nn.ReLU, norm_layer=nn.LayerNorm, line_graph=True):
        super().__init__()
        self.g_update = CrysformerLayer(dim, num_heads, use_bias, mlp_ratio,
                                        drop, act_layer, norm_layer)
        # self.g_dynamic = DynamicConnection(threshold=0.3, t=1.)
        if line_graph:
            self.lg_update = CrysformerLayer(dim, num_heads, use_bias, mlp_ratio,
                                             drop, act_layer, norm_layer)
            # self.lg_dynamic = DynamicConnection(threshold=0.3, t=1.)

    def forward(self, g, x, y, lg=None, z=None):
        """Node and Edge updates for ALIGNN layer.

        x: node input features
        y: edge input features
        z: edge pair input features
        """

        if lg is not None:
            # lg = lg.local_var()
            y, z, score = self.lg_update(lg, y, z)
            # z, _ = self.lg_dynamic(lg, score, z)

        # g = g.local_var()
        x, y, score = self.g_update(g, x, y)
        # y, z = self.g_dynamic(g, score, y, lg, z)

        return x, y, z


class Crysformer(nn.Module):
    def __init__(self, inputs=['graph', 'line_graph'], targets=[],
                 depth=4, edge_input_dim=80, triplet_input_dim=40,
                 embed_dim=128, num_heads=4, mlp_ratio=2.,
                 use_bias=True, norm_layer=None, act_layer=None):
        """Initialize class with number of input features, conv layers."""
        super().__init__()

        self.inputs = inputs
        self.targets = targets
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.ReLU
        self.dengbian_post_embedding = nn.Linear(embed_dim,3)
        self.atom_embedding = nn.Embedding(95, embed_dim)
        self.dengbian_embedding = nn.Linear(embed_dim,1)
        self.edge_embedding = nn.Sequential(
            RBFExpansion(vmin=0, vmax=8.0, bins=edge_input_dim),
            nn.Linear(edge_input_dim, embed_dim),
            norm_layer(embed_dim),
            act_layer(),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(vmin=-1, vmax=1.0, bins=triplet_input_dim),
            nn.Linear(triplet_input_dim, embed_dim),
            norm_layer(embed_dim),
            act_layer(),
        )

        self.blocks = nn.ModuleList([
                CrysformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    use_bias=use_bias,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    line_graph='line_graph' in inputs,
                )
                for _ in range(depth)
            ])

        self.readout = AvgPooling()
        self.exdd = GraphAttentionTransformerMD17()
        # self.head = MLP(embed_dim, embed_dim, len(targets), 2)
        self.head = nn.Linear(embed_dim, len(targets))

    def forward(
        self, g: Union[Tuple[dgl.DGLGraph, dgl.DGLGraph], dgl.DGLGraph]
    ):
        """
        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)
        """
        #print('ddddd:',g)
        if isinstance(g, list):
            assert len(g) == 2
            g, lg = g
            g = g.local_var()
            lg = lg.local_var()
            z = self.angle_embedding(lg.edata.pop("angle"))
        else:
            g = g.local_var()
            lg, z = None, None


        x = self.atom_embedding(g.ndata.pop("atomic_numbers") - 1)
        y = self.edge_embedding(g.edata.pop('distance'))
        dengbian_x = g.out_degrees().to(torch.int64).cuda()

        dengbian_batch = torch.zeros_like(dengbian_x).long().cuda()
        dengbian_post = self.dengbian_post_embedding(x).view(-1,3)


        for block in self.blocks:
            x, y, z = block(g, x, y, lg, z)

        h = self.readout(g, x)

        if x.shape[0] > 20:
            sdd = self.exdd(dengbian_x, dengbian_post, dengbian_batch)
            out = (self.head(h)+sdd[0])/2
        else:
            out = h

        return out
