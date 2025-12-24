import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_sgap(concepts, max_gap=300):
    """
    For each position, calculate gap to NEXT occurrence of same concept.

    Args:
        concepts: Tensor of shape [batch_size, seq_len] containing concept IDs
        max_gap: Maximum gap value to cap at (default 300)

    Returns:
        sgap: Tensor of shape [batch_size, seq_len] with gap values
    """
    batch_size, seq_len = concepts.shape
    sgap = torch.full((batch_size, seq_len), max_gap - 1, dtype=torch.long, device=concepts.device)

    for b in range(batch_size):
        for i in range(seq_len):
            concept = concepts[b, i].item()
            # Find next occurrence of same concept
            for j in range(i + 1, seq_len):
                if concepts[b, j].item() == concept:
                    sgap[b, i] = min(j - i, max_gap - 1)
                    break

    return sgap


def calculate_pcount(concepts):
    """
    For each position, count items since LAST occurrence of same concept.

    Args:
        concepts: Tensor of shape [batch_size, seq_len] containing concept IDs

    Returns:
        pcount: Tensor of shape [batch_size, seq_len] with count values
    """
    batch_size, seq_len = concepts.shape
    pcount = torch.zeros((batch_size, seq_len), dtype=torch.long, device=concepts.device)

    for b in range(batch_size):
        concept_last_pos = {}
        for i in range(seq_len):
            concept = concepts[b, i].item()
            if concept in concept_last_pos:
                pcount[b, i] = i - concept_last_pos[concept]
            else:
                pcount[b, i] = 0  # First occurrence
            concept_last_pos[concept] = i

    return pcount


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class AKTInt(nn.Module):
    """AKT with Interference-based forgetting (AKTInt)

    This model extends AKT with interference decay that captures:
    - sgap: Gap to next occurrence of same concept (recency of future repetition)
    - pcount: Count since last occurrence (interference amount)
    """
    def __init__(self, n_question, n_pid, d_model, n_blocks, dropout, d_ff=256,
            kq_same=1, final_fc_dim=512, num_attn_heads=8, separate_qa=False, l2=1e-5, emb_type="qid", emb_path="", pretrain_dim=768):
        super().__init__()
        """
        Input:
            d_model: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            num_attn_heads: number of heads in multi-headed attention
            d_ff : dimension for fully conntected net inside the basic block
            kq_same: if key query same, kq_same=1, else = 0
        """
        self.model_name = "aktint"
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.l2 = l2
        self.model_type = "akt"  # Use akt architecture type
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        embed_l = d_model
        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid+1, 1)
            self.q_embed_diff = nn.Embedding(self.n_question+1, embed_l)
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l)

        if emb_type.startswith("qid"):
            self.q_embed = nn.Embedding(self.n_question, embed_l)
            if self.separate_qa:
                self.qa_embed = nn.Embedding(2*self.n_question+1, embed_l)
            else:
                self.qa_embed = nn.Embedding(2, embed_l)

        # Architecture Object
        self.model = ArchitectureInt(n_question=n_question, n_blocks=n_blocks, n_heads=num_attn_heads, dropout=dropout,
                                    d_model=d_model, d_feature=d_model / num_attn_heads, d_ff=d_ff, kq_same=self.kq_same, model_type=self.model_type, emb_type=self.emb_type)

        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )
        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid+1 and self.n_pid > 0:
                torch.nn.init.constant_(p, 0.)

    def base_emb(self, q_data, target):
        q_embed_data = self.q_embed(q_data)
        if self.separate_qa:
            qa_data = q_data + self.n_question * target
            qa_embed_data = self.qa_embed(qa_data)
        else:
            qa_embed_data = self.qa_embed(target)+q_embed_data
        return q_embed_data, qa_embed_data

    def forward(self, q_data, target, pid_data=None, qtest=False, dgaps=None):
        """
        Forward pass with interference-based forgetting.

        Args:
            q_data: Question/concept IDs [bs, seq_len]
            target: Response labels [bs, seq_len]
            pid_data: Problem IDs (optional) [bs, seq_len]
            qtest: If True, return additional concat_q output
            dgaps: Dictionary with interference data:
                   {'sgaps': tensor, 'pcounts': tensor}
                   - sgaps: Gap to next occurrence of same concept [bs, seq_len]
                   - pcounts: Count of items since last occurrence [bs, seq_len]

        Returns:
            If qtest=False: (preds, c_reg_loss)
            If qtest=True: (preds, c_reg_loss, concat_q)
        """
        emb_type = self.emb_type
        if emb_type.startswith("qid"):
            q_embed_data, qa_embed_data = self.base_emb(q_data, target)

        pid_embed_data = None
        if self.n_pid > 0:
            q_embed_diff_data = self.q_embed_diff(q_data)
            pid_embed_data = self.difficult_param(pid_data)
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data

            qa_embed_diff_data = self.qa_embed_diff(target)
            if self.separate_qa:
                qa_embed_data = qa_embed_data + pid_embed_data * qa_embed_diff_data
            else:
                qa_embed_data = qa_embed_data + pid_embed_data * (qa_embed_diff_data+q_embed_diff_data)
            c_reg_loss = (pid_embed_data ** 2.).sum() * self.l2
        else:
            c_reg_loss = 0.

        # Extract interference data
        sgap = dgaps.get("sgaps", None) if dgaps is not None else None
        pcount = dgaps.get("pcounts", None) if dgaps is not None else None

        d_output = self.model(q_embed_data, qa_embed_data, pid_embed_data,
                              sgap=sgap, pcount=pcount)

        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q).squeeze(-1)
        m = nn.Sigmoid()
        preds = m(output)
        if not qtest:
            return preds, c_reg_loss
        else:
            return preds, c_reg_loss, concat_q


class ArchitectureInt(nn.Module):
    def __init__(self, n_question, n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type, emb_type):
        super().__init__()
        self.d_model = d_model
        self.model_type = model_type

        if model_type in {'akt'}:
            self.blocks_1 = nn.ModuleList([
                TransformerLayerInt(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same, emb_type=emb_type)
                for _ in range(n_blocks)
            ])
            self.blocks_2 = nn.ModuleList([
                TransformerLayerInt(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same, emb_type=emb_type)
                for _ in range(n_blocks*2)
            ])

    def forward(self, q_embed_data, qa_embed_data, pid_embed_data,
                sgap=None, pcount=None):
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed

        # encoder
        for block in self.blocks_1:
            y = block(mask=1, query=y, key=y, values=y, pdiff=pid_embed_data,
                     sgap=sgap, pcount=pcount)
        flag_first = True
        for block in self.blocks_2:
            if flag_first:
                x = block(mask=1, query=x, key=x,
                          values=x, apply_pos=False, pdiff=pid_embed_data,
                          sgap=sgap, pcount=pcount)
                flag_first = False
            else:
                x = block(mask=0, query=x, key=x, values=y, apply_pos=True, pdiff=pid_embed_data,
                          sgap=sgap, pcount=pcount)
                flag_first = True
        return x

class TransformerLayerInt(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, emb_type):
        super().__init__()
        kq_same = kq_same == 1
        self.masked_attn_head = MultiHeadAttentionInt(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same, emb_type=emb_type)

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True, pdiff=None,
                sgap=None, pcount=None):
        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        if mask == 0:
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True, pdiff=pdiff,
                sgap=sgap, pcount=pcount)
        else:
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False, pdiff=pdiff,
                sgap=sgap, pcount=pcount)

        query = query + self.dropout1((query2))
        query = self.layer_norm1(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2))
            query = self.layer_norm2(query)
        return query


class MultiHeadAttentionInt(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True, emb_type="qid"):
        super().__init__()
        self.d_model = d_model
        self.emb_type = emb_type
        if emb_type.endswith("avgpool"):
            pool_size = 3
            self.pooling = nn.AvgPool1d(pool_size, stride=1, padding=pool_size//2, count_include_pad=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        elif emb_type.endswith("linear"):
            self.linear = nn.Linear(d_model, d_model, bias=bias)
            self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        elif emb_type.startswith("qid"):
            self.d_k = d_feature
            self.h = n_heads
            self.kq_same = kq_same

            self.v_linear = nn.Linear(d_model, d_model, bias=bias)
            self.k_linear = nn.Linear(d_model, d_model, bias=bias)
            if kq_same is False:
                self.q_linear = nn.Linear(d_model, d_model, bias=bias)
            self.dropout = nn.Dropout(dropout)
            self.proj_bias = bias
            self.out_proj = nn.Linear(d_model, d_model, bias=bias)
            self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
            torch.nn.init.xavier_uniform_(self.gammas)

            # Beta parameter for interference decay
            self.beta = nn.Parameter(torch.zeros(n_heads, 1, 1))
            torch.nn.init.constant_(self.beta, 0.1)

            self._reset_parameters()


    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad, pdiff=None, sgap=None, pcount=None):
        bs = q.size(0)

        if self.emb_type.endswith("avgpool"):
            scores = self.pooling(v)
            concat = self.pad_zero(scores, bs, scores.shape[2], zero_pad)
        elif self.emb_type.endswith("linear"):
            scores = self.linear(v)
            concat = self.pad_zero(scores, bs, scores.shape[2], zero_pad)
        elif self.emb_type.startswith("qid"):
            k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
            if self.kq_same is False:
                q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
            else:
                q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
            v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

            k = k.transpose(1, 2)
            q = q.transpose(1, 2)
            v = v.transpose(1, 2)
            gammas = self.gammas
            if self.emb_type.find("pdiff") == -1:
                pdiff = None
            scores = attention_int(q, k, v, self.d_k,
                            mask, self.dropout, zero_pad, gammas, pdiff,
                            sgap=sgap, pcount=pcount, beta=self.beta)

            concat = scores.transpose(1, 2).contiguous()\
                .view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output

    def pad_zero(self, scores, bs, dim, zero_pad):
        if zero_pad:
            pad_zero = torch.zeros(bs, 1, dim).to(device)
            scores = torch.cat([pad_zero, scores[:, 0:-1, :]], dim=1)
        return scores


def attention_int(q, k, v, d_k, mask, dropout, zero_pad, gamma=None, pdiff=None,
              sgap=None, pcount=None, beta=None):
    """
    Attention with interference decay alongside temporal decay:
        attention = base_scores * temporal_decay * interference_decay

    Args:
        q, k, v: Query, Key, Value tensors
        d_k: Dimension per head
        mask: Attention mask
        dropout: Dropout layer
        zero_pad: Whether to zero-pad first position
        gamma: Learnable temporal decay parameter
        pdiff: Problem difficulty (optional)
        sgap: Gap to next occurrence of same concept [bs, seq_len]
        pcount: Count of items since last occurrence [bs, seq_len]
        beta: Learnable interference decay parameter [n_heads, 1, 1]
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
    x2 = x1.transpose(0, 1).contiguous()

    # Temporal decay calculation
    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)
        scores_ = scores_ * mask.float().to(device)
        distcum_scores = torch.cumsum(scores_, dim=-1)
        disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
        position_effect = torch.abs(x1-x2)[None, None, :, :].type(torch.FloatTensor).to(device)
        dist_scores = torch.clamp((disttotal_scores-distcum_scores)*position_effect, min=0.)
        dist_scores = dist_scores.sqrt().detach()

    m = nn.Softplus()
    gamma = -1. * m(gamma).unsqueeze(0)

    if pdiff == None:
        temporal_effect = torch.clamp(torch.clamp(
            (dist_scores*gamma).exp(), min=1e-5), max=1e5)
    else:
        diff = pdiff.unsqueeze(1).expand(pdiff.shape[0], dist_scores.shape[1], pdiff.shape[1], pdiff.shape[2])
        diff = diff.sigmoid().exp()
        temporal_effect = torch.clamp(torch.clamp(
            (dist_scores*gamma*diff).exp(), min=1e-5), max=1e5)

    # Interference decay calculation
    if sgap is not None and pcount is not None and beta is not None:
        sgap_norm = sgap.float() / (sgap.max() + 1e-6)
        pcount_norm = pcount.float() / (pcount.max() + 1e-6)

        # Combined interference score
        interference_score = (1 - sgap_norm) * 0.5 + pcount_norm * 0.5

        # Build pairwise interference matrix
        interf_cumsum = torch.cumsum(interference_score, dim=1)
        interf_matrix = interf_cumsum.unsqueeze(2) - interf_cumsum.unsqueeze(1)
        interf_matrix = torch.abs(interf_matrix)

        # Expand for multi-head attention
        interf_matrix = interf_matrix.unsqueeze(1).expand(bs, head, seqlen, seqlen)

        # Apply learnable interference decay
        beta_transformed = -1. * m(beta).unsqueeze(0)
        interference_effect = torch.clamp(
            (interf_matrix * beta_transformed).exp(), min=1e-5, max=1e5)
    else:
        interference_effect = 1.0

    # Combine: multiplicative forgetting
    total_effect = temporal_effect * interference_effect
    scores = scores * total_effect

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = 0.1 * torch.randn(max_len, d_model)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]
