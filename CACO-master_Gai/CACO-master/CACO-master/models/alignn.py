"""Atomistic LIne Graph Neural Network.

A prototype crystal line graph network dgl implementation.
"""
from typing import Tuple, Union

import dgl
import dgl.function as fn
import numpy as np
import torch
from dgl.nn import AvgPooling
from torch import nn
from torch.nn import functional as F

from .layers import RBFExpansion


class EdgeGatedGraphConv(nn.Module):
    """Edge gated graph convolution from arxiv:1711.07553.

    see also arxiv:2003.0098.

    This is similar to CGCNN, but edge features only go into
    the soft attention / edge gating function, and the primary
    node update function is W cat(u, v) + b
    """

    def __init__(
        self, input_features: int, output_features: int, residual: bool = True
    ):
        """Initialize parameters for ALIGNN update."""
        super().__init__()
        self.residual = residual
        # CGCNN-Conv operates on augmented edge features
        # z_ij = cat(v_i, v_j, u_ij)
        # m_ij = σ(z_ij W_f + b_f) ⊙ g_s(z_ij W_s + b_s)
        # coalesce parameters for W_f and W_s
        # but -- split them up along feature dimension
        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)
        self.bn_edges = nn.BatchNorm1d(output_features)

        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)
        self.bn_nodes = nn.BatchNorm1d(output_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Edge-gated graph convolution.

        h_i^l+1 = ReLU(U h_i + sum_{j->i} eta_{ij} ⊙ V h_j)
        """
        g = g.local_var()

        # instead of concatenating (u || v || e) and applying one weight matrix
        # split the weight matrix into three, apply, then sum
        # see https://docs.dgl.ai/guide/message-efficient.html
        # but split them on feature dimensions to update u, v, e separately
        # m = BatchNorm(Linear(cat(u, v, e)))

        # compute edge updates, equivalent to:
        # Softplus(Linear(u || v || e))
        g.ndata["e_src"] = self.src_gate(node_feats)
        g.ndata["e_dst"] = self.dst_gate(node_feats)
        g.apply_edges(fn.u_add_v("e_src", "e_dst", "e_nodes"))
        m = g.edata.pop("e_nodes") + self.edge_gate(edge_feats)

        g.edata["sigma"] = torch.sigmoid(m)
        g.ndata["Bh"] = self.dst_update(node_feats)
        g.update_all(
            fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h")
        )
        g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
        g.ndata["h"] = g.ndata["sum_sigma_h"] / (g.ndata["sum_sigma"] + 1e-6)
        x = self.src_update(node_feats) + g.ndata.pop("h")

        # softmax version seems to perform slightly worse
        # that the sigmoid-gated version
        # compute node updates
        # Linear(u) + edge_gates ⊙ Linear(v)
        # g.edata["gate"] = edge_softmax(g, y)
        # g.ndata["h_dst"] = self.dst_update(node_feats)
        # g.update_all(fn.u_mul_e("h_dst", "gate", "m"), fn.sum("m", "h"))
        # x = self.src_update(node_feats) + g.ndata.pop("h")

        # node and edge updates
        x = F.silu(self.bn_nodes(x))
        y = F.silu(self.bn_edges(m))

        if self.residual:
            x = node_feats + x
            y = edge_feats + y

        return x, y


class ALIGNNConv(nn.Module):
    """Line graph update."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        """Set up ALIGNN parameters."""
        super().__init__()
        self.node_update = EdgeGatedGraphConv(in_features, out_features)
        self.edge_update = EdgeGatedGraphConv(out_features, out_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        lg: dgl.DGLGraph,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ):
        """Node and Edge updates for ALIGNN layer.

        x: node input features
        y: edge input features
        z: edge pair input features
        """
        g = g.local_var()
        lg = lg.local_var()
        # Edge-gated graph convolution update on crystal graph
        x, m = self.node_update(g, x, y)

        # Edge-gated graph convolution update on crystal graph
        y, z = self.edge_update(lg, m, z)

        return x, y, z


class MLPLayer(nn.Module):
    """Multilayer perceptron layer helper."""

    def __init__(self, in_features: int, out_features: int):
        """Linear, Batchnorm, SiLU layer."""
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.SiLU(),
        )

    def forward(self, x):
        """Linear, Batchnorm, silu layer."""
        return self.layer(x)


class ALIGNN(nn.Module):
    """Atomistic Line graph network.

    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph.
    """

    def __init__(self, alignn_layers=2, gcn_layers=2, edge_input_features=80,
                 triplet_input_features=40, embedding_features=64, hidden_features=256, output_features=1,
                 link='identity'):
        """Initialize class with number of input features, conv layers."""
        super().__init__()

        self.atom_embedding = nn.Embedding(95, hidden_features)
        # self.atom_embedding = MLPLayer(
        #     atom_input_features, hidden_features
        # )

        self.edge_embedding = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=8.0,
                bins=edge_input_features,
            ),
            MLPLayer(edge_input_features, embedding_features),
            MLPLayer(embedding_features, hidden_features),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(
                vmin=-1,
                vmax=1.0,
                bins=triplet_input_features,
            ),
            MLPLayer(triplet_input_features, embedding_features),
            MLPLayer(embedding_features, hidden_features),
        )

        self.alignn_layers = nn.ModuleList(
            [
                ALIGNNConv(
                    hidden_features,
                    hidden_features,
                )
                for idx in range(alignn_layers)
            ]
        )
        self.gcn_layers = nn.ModuleList(
            [
                EdgeGatedGraphConv(
                    hidden_features, hidden_features
                )
                for idx in range(gcn_layers)
            ]
        )

        self.readout = AvgPooling()

        self.fc = nn.Linear(hidden_features, output_features)
        self.link = None
        self.link_name = link
        if link == "identity":
            self.link = lambda x: x
        elif link == "log":
            self.link = torch.exp
            avg_gap = 0.7  # magic number -- average bandgap in dft_3d
            self.fc.bias.data = torch.tensor(
                np.log(avg_gap), dtype=torch.float
            )
        elif link == "logit":
            self.link = torch.sigmoid

    def forward(
        self, g: Union[Tuple[dgl.DGLGraph, dgl.DGLGraph], dgl.DGLGraph]
    ):
        """ALIGNN : start with `atom_features`.

        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)
        """
        if len(self.alignn_layers) > 0:
            g, lg = g
            lg = lg.local_var()

            # angle features (fixed)
            z = self.angle_embedding(lg.edata.pop("angle"))

        g = g.local_var()

        # initial node features: atom feature network...
        x = self.atom_embedding(g.ndata.pop("atomic_numbers") - 1)

        # initial bond features
        y = self.edge_embedding(g.edata.pop('distance'))

        # ALIGNN updates: update node, edge, triplet features
        for alignn_layer in self.alignn_layers:
            x, y, z = alignn_layer(g, lg, x, y, z)

        # gated GCN updates: update node, edge features
        for gcn_layer in self.gcn_layers:
            x, y = gcn_layer(g, x, y)

        # norm-activation-pool-classify
        h = self.readout(g, x)
        out = self.fc(h)

        if self.link:
            out = self.link(out)

        return out