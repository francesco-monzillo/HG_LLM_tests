import torch
import torch.nn as nn
import torch.nn.functional as F

#Can eventually additionally take LOD-CLOUD Graph-based embeddings into account

EPS = 1e-9

class HypergraphAttentionConv(nn.Module):
    """
    Single HGAT layer.
    Inputs:
      - in_feats: input node feature dim
      - out_feats: output node feature dim
    forward:
      X: [N, F_in]
      H: [N, E] binary incidence (float tensor, 0/1)
    Output:
      X_out: [N, out_feats]
    """
    def __init__(self, in_feats, out_feats, attn_hidden=64, leaky_relu_neg_slope=0.2):
        super().__init__()
        self.W = nn.Linear(in_feats, out_feats, bias=False)  # node transform
        # attention mechanism: score for node and for hyperedge (after pooling)
        self.a_node = nn.Linear(out_feats, attn_hidden, bias=False)
        self.a_edge = nn.Linear(out_feats, attn_hidden, bias=False)
        self.attn_v = nn.Linear(attn_hidden, 1, bias=False)
        self.activation = nn.LeakyReLU(leaky_relu_neg_slope)
        self.out_act = nn.Identity()

    def forward(self, X, H):
        """
        X: [N, F_in]
        H: [N, E] incidence matrix (0/1)
        """
        N, E = H.shape
        # 1) transform node features
        X_t = self.W(X)               # [N, out_feats]

        # 2) compute hyperedge features by averaging transformed node features in each hyperedge
        # denom: size of hyperedge (number of nodes)
        edge_sizes = H.sum(dim=0) + EPS        # [E]
        # To get He: (H^T @ X_t) / edge_sizes[:, None]
        He = (H.T @ X_t) / edge_sizes.unsqueeze(1)   # [E, out_feats]

        # 3) compute attention logits for every (node, hyperedge) pair where H=1
        # prepare repeated tensors for vectorized score computation:
        # compute a_node(X_t) -> [N, attn_hidden]
        s_node = self.a_node(X_t)  # [N, A]
        s_edge = self.a_edge(He)   # [E, A]

        # expand to shape [N, E, A] virtually by broadcasting
        # compute score = attn_v(LeakyReLU(s_node[i] + s_edge[e]))
        # vectorized: s_node.unsqueeze(1) + s_edge.unsqueeze(0) => [N, E, A]
        s_sum = s_node.unsqueeze(1) + s_edge.unsqueeze(0)  # [N, E, A]
        s = self.attn_v(self.activation(s_sum)).squeeze(-1)  # [N, E] logits

        # mask positions not in hyperedge: set to -inf so softmax ignores them
        mask = (H > 0).float()  # [N, E]
        neg_inf = -1e9
        s_masked = s * mask + (1.0 - mask) * neg_inf  # masked logits

        # 4) normalize attention weights per hyperedge (softmax over nodes for each hyperedge)
        # softmax along dim=0 (nodes) for each hyperedge column
        alpha = F.softmax(s_masked, dim=0)  # [N, E]; zeros where H==0

        # zero-out numerical noise where H==0
        alpha = alpha * mask

        # 5) build attention-weighted incidence H_att = alpha (already zero outside incidence)
        H_att = alpha  # [N, E]

        # 6) compute degree matrices based on H_att
        De = H_att.sum(dim=0) + EPS   # [E]
        Dv = H_att.sum(dim=1) + EPS   # [N]

        # Normalize as in HGNN: Dv^-1/2 H De^-1 H^T Dv^-1/2
        Dv_inv_sqrt = (1.0 / torch.sqrt(Dv)).unsqueeze(1)    # [N,1]
        De_inv = (1.0 / De).unsqueeze(1)                     # [E,1] (we'll use broadcasting)

        # Compute intermediate: (Dv^-1/2 * H_att) * De^-1
        # Step A = Dv^-1/2 * H_att  => [N, E]
        A = Dv_inv_sqrt * H_att
        # Step B = A * De_inv^T  => multiply each column e by De_inv[e]
        B = A * De_inv.t()  # broadcasting; [N, E]

        # Final normalized adjacency: H_norm = B @ H_att.T  => [N, N]
        H_norm = B @ H_att.t()   # [N, N]
        # multiply on left again by Dv^-1/2
        H_norm = Dv_inv_sqrt * H_norm  # [N, N]

        # 7) propagate features: use transformed node features X_t as input to propagation
        X_out = H_norm @ X_t    # [N, out_feats]
        return self.out_act(X_out)


class HGAT(nn.Module):
    def __init__(self, in_feats, hidden_dim, out_feats, dropout=0.5):
        super().__init__()
        self.layer1 = HypergraphAttentionConv(in_feats, hidden_dim)
        self.layer2 = HypergraphAttentionConv(hidden_dim, out_feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, H):
        X = self.layer1(X, H)
        X = F.relu(X)
        X = self.dropout(X)
        X = self.layer2(X, H)
        return X
