import math
import torch
from torch import nn
import torch.nn.functional as F


def get_fixed_sin_cos_encodings(d_model, max_len):
    """
    Sin-cos fixed positional encodings.
    """
    assert d_model % 2 == 0
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int, trainable=False):
        super().__init__()
        self.max_len = int(max_len)
        self.trainable = bool(trainable)

        if self.trainable:
            self.pe = nn.Embedding(max_len, d_model)
        else:
            self.register_buffer("pe", get_fixed_sin_cos_encodings(d_model, max_len))

    def forward(self, x):
        batch_size = x.size(0)
        actual_len = x.shape[1]
        assert actual_len <= self.max_len

        pe = self.pe.weight if self.trainable else self.pe
        return pe.unsqueeze(0).repeat(batch_size, 1, 1)[:, :actual_len, :]

    def get_pe(self, position):
        pe = self.pe.weight if self.trainable else self.pe
        return pe[position]


class RelativePositionalEncoding(nn.Module):
    def __init__(self, max_relative_position: int, d_model: int, trainable=False, cross_attn=False):
        super().__init__()
        self.max_relative_position = int(max_relative_position)
        self.trainable = bool(trainable)
        self.cross_attn = bool(cross_attn)
        self.num_embeddings = (
            self.max_relative_position * 2 + 1
            if not self.cross_attn
            else self.max_relative_position + 1
        )

        if self.trainable:
            self.embeddings_table = nn.Embedding(self.num_embeddings, d_model)
        else:
            self.register_buffer(
                "embeddings_table",
                get_fixed_sin_cos_encodings(d_model, self.max_relative_position * 2 + 1),
            )

    def forward(self, length_q, length_k):
        embeddings_table = self.embeddings_table.weight if self.trainable else self.embeddings_table

        if self.cross_attn:
            distance_mat = torch.arange(length_k - 1, -1, -1)[None, :] + torch.arange(length_q)[:, None]
        else:
            distance_mat = torch.arange(length_k)[None, :] - torch.arange(length_q)[:, None]

        distance_mat_clipped = torch.clamp(
            distance_mat,
            -self.max_relative_position,
            self.max_relative_position,
        )

        if not self.cross_attn:
            distance_mat_clipped = distance_mat_clipped + self.max_relative_position

        final_mat = torch.LongTensor(distance_mat_clipped).to(embeddings_table.device)
        embeddings = embeddings_table[final_mat]
        return embeddings


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = float(eps)

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


class Attention(nn.Module):
    def __init__(
        self,
        positional_encoding_k: RelativePositionalEncoding = None,
        positional_encoding_v: RelativePositionalEncoding = None,
    ):
        super().__init__()
        self.positional_encoding_k = positional_encoding_k
        self.positional_encoding_v = positional_encoding_v

    def forward(self, query, key, value, mask=None, dropout=None, one_direction=False):
        scores = torch.matmul(query, key.transpose(-2, -1))

        if self.positional_encoding_k is not None:
            R_k = self.positional_encoding_k(query.size(2), key.size(2))
            scores = scores + torch.einsum("b h q d, q k d -> b h q k", query, R_k)

        scores = scores / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        if one_direction:
            direction_mask = torch.ones_like(scores)
            direction_mask = torch.tril(direction_mask)
            scores = scores.masked_fill(direction_mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        output = torch.matmul(p_attn, value)

        if self.positional_encoding_v is not None:
            R_v = self.positional_encoding_v(query.size(2), value.size(2))
            output = output + torch.einsum("b h q v, q v d -> b h q d", p_attn, R_v)

        return output, p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(
        self,
        num_heads,
        d_model,
        head_size=None,
        dropout=0.0,
        positional_encoding_k=None,
        positional_encoding_v=None,
        final_layer=False,
    ):
        super().__init__()

        self.num_heads = int(num_heads)

        if head_size is not None:
            self.head_size = int(head_size)
        else:
            assert d_model % num_heads == 0
            self.head_size = d_model // num_heads

        self.linear_layers = nn.ModuleList(
            [
                nn.Linear(d_model, self.num_heads * self.head_size)
                for _ in range(3)
            ]
        )

        self.attention = Attention(positional_encoding_k, positional_encoding_v)
        self.dropout = nn.Dropout(p=dropout)

        if final_layer:
            self.final_layer = nn.Linear(self.num_heads * self.head_size, d_model)

        self.layer_norm = LayerNorm(d_model)

    def forward(self, query, key, value, mask=None, one_direction=True):
        batch_size = query.size(0)

        query_, key_, value_ = [
            layer(x)
            .view(batch_size, -1, self.num_heads, self.head_size)
            .transpose(1, 2)
            for layer, x in zip(self.linear_layers, (query, key, value))
        ]

        x, attn = self.attention(
            query_,
            key_,
            value_,
            mask=mask,
            dropout=self.dropout,
            one_direction=one_direction,
        )

        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.head_size)
        )

        if hasattr(self, "final_layer"):
            x = self.final_layer(x)

        return self.layer_norm(x + query)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = LayerNorm(d_model)

    def forward(self, x):
        x_ = self.dropout(self.activation(self.conv1(x.permute(0, 2, 1))))
        return self.layer_norm(self.dropout(self.conv2(x_)).permute(0, 2, 1) + x)


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        hidden,
        attn_heads,
        head_size,
        feed_forward_hidden,
        dropout,
        attn_dropout=0.1,
        self_positional_encoding_k=None,
        self_positional_encoding_v=None,
        final_layer=True,
        **kwargs,
    ):
        super().__init__()
        self.self_attention = MultiHeadedAttention(
            num_heads=attn_heads,
            d_model=hidden,
            head_size=head_size,
            dropout=attn_dropout,
            positional_encoding_k=self_positional_encoding_k,
            positional_encoding_v=self_positional_encoding_v,
            final_layer=final_layer,
        )
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden,
            d_ff=feed_forward_hidden,
            dropout=dropout,
        )

    def forward(self, x, active_entries):
        self_att_mask = active_entries.repeat(1, 1, active_entries.size(1)).unsqueeze(1)
        x = self.self_attention(x, x, x, self_att_mask, True)
        x = self.feed_forward(x)
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        hidden,
        attn_heads,
        head_size,
        feed_forward_hidden,
        dropout,
        attn_dropout,
        self_positional_encoding_k=None,
        self_positional_encoding_v=None,
        cross_positional_encoding_k=None,
        cross_positional_encoding_v=None,
        final_layer=False,
        **kwargs,
    ):
        super().__init__()
        self.layer_norm = LayerNorm(hidden)
        self.self_attention = MultiHeadedAttention(
            num_heads=attn_heads,
            d_model=hidden,
            head_size=head_size,
            dropout=attn_dropout,
            positional_encoding_k=self_positional_encoding_k,
            positional_encoding_v=self_positional_encoding_v,
            final_layer=final_layer,
        )
        self.cross_attention = MultiHeadedAttention(
            num_heads=attn_heads,
            d_model=hidden,
            head_size=head_size,
            dropout=attn_dropout,
            positional_encoding_k=cross_positional_encoding_k,
            positional_encoding_v=cross_positional_encoding_v,
            final_layer=final_layer,
        )
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden,
            d_ff=feed_forward_hidden,
            dropout=dropout,
        )

    def forward(self, x, encoder_x, active_entries, active_encoder_br):
        self_att_mask = active_entries.repeat(1, 1, active_entries.size(1)).unsqueeze(1)
        cross_att_mask = (active_encoder_br.unsqueeze(1) * active_entries).unsqueeze(1)

        x = self.self_attention(x, x, x, self_att_mask, True)
        x = self.cross_attention(x, encoder_x, encoder_x, cross_att_mask, False)
        x = self.feed_forward(x)
        return x


class TransformerMultiInputBlock(nn.Module):
    """
    CT multi-input transformer block.

    Streams:
      - treatments
      - previous outcomes
      - vitals, when n_inputs == 3
    """

    def __init__(
        self,
        hidden,
        attn_heads,
        head_size,
        feed_forward_hidden,
        dropout,
        attn_dropout,
        self_positional_encoding_k=None,
        self_positional_encoding_v=None,
        n_inputs=2,
        final_layer=False,
        disable_cross_attention=False,
        isolate_subnetwork="",
        **kwargs,
    ):
        super().__init__()

        self.n_inputs = int(n_inputs)
        self.disable_cross_attention = bool(disable_cross_attention)
        self.isolate_subnetwork = str(isolate_subnetwork)

        self.self_attention_o = MultiHeadedAttention(
            num_heads=attn_heads,
            d_model=hidden,
            head_size=head_size,
            dropout=attn_dropout,
            positional_encoding_k=self_positional_encoding_k,
            positional_encoding_v=self_positional_encoding_v,
            final_layer=final_layer,
        )

        self.self_attention_t = MultiHeadedAttention(
            num_heads=attn_heads,
            d_model=hidden,
            head_size=head_size,
            dropout=attn_dropout,
            positional_encoding_k=self_positional_encoding_k,
            positional_encoding_v=self_positional_encoding_v,
            final_layer=final_layer,
        )

        if not self.disable_cross_attention:
            self.cross_attention_ot = MultiHeadedAttention(
                num_heads=attn_heads,
                d_model=hidden,
                head_size=head_size,
                dropout=attn_dropout,
                positional_encoding_k=self_positional_encoding_k,
                positional_encoding_v=self_positional_encoding_v,
                final_layer=final_layer,
            )

            self.cross_attention_to = MultiHeadedAttention(
                num_heads=attn_heads,
                d_model=hidden,
                head_size=head_size,
                dropout=attn_dropout,
                positional_encoding_k=self_positional_encoding_k,
                positional_encoding_v=self_positional_encoding_v,
                final_layer=final_layer,
            )

        if self.n_inputs == 3:
            self.self_attention_v = MultiHeadedAttention(
                num_heads=attn_heads,
                d_model=hidden,
                head_size=head_size,
                dropout=attn_dropout,
                positional_encoding_k=self_positional_encoding_k,
                positional_encoding_v=self_positional_encoding_v,
                final_layer=final_layer,
            )

            if not self.disable_cross_attention:
                self.cross_attention_tv = MultiHeadedAttention(
                    num_heads=attn_heads,
                    d_model=hidden,
                    head_size=head_size,
                    dropout=attn_dropout,
                    positional_encoding_k=self_positional_encoding_k,
                    positional_encoding_v=self_positional_encoding_v,
                    final_layer=final_layer,
                )
                self.cross_attention_vt = MultiHeadedAttention(
                    num_heads=attn_heads,
                    d_model=hidden,
                    head_size=head_size,
                    dropout=attn_dropout,
                    positional_encoding_k=self_positional_encoding_k,
                    positional_encoding_v=self_positional_encoding_v,
                    final_layer=final_layer,
                )
                self.cross_attention_ov = MultiHeadedAttention(
                    num_heads=attn_heads,
                    d_model=hidden,
                    head_size=head_size,
                    dropout=attn_dropout,
                    positional_encoding_k=self_positional_encoding_k,
                    positional_encoding_v=self_positional_encoding_v,
                    final_layer=final_layer,
                )
                self.cross_attention_vo = MultiHeadedAttention(
                    num_heads=attn_heads,
                    d_model=hidden,
                    head_size=head_size,
                    dropout=attn_dropout,
                    positional_encoding_k=self_positional_encoding_k,
                    positional_encoding_v=self_positional_encoding_v,
                    final_layer=final_layer,
                )

        self.feed_forwards = nn.ModuleList(
            [
                PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
                for _ in range(self.n_inputs)
            ]
        )

    def forward(self, x_tov, x_s, active_entries_treat_outcomes, active_entries_vitals=None):
        assert len(x_tov) == self.n_inputs

        if self.n_inputs == 2:
            x_t, x_o = x_tov
        else:
            x_t, x_o, x_v = x_tov

        self_att_mask_ot = active_entries_treat_outcomes.repeat(1, 1, x_t.size(1)).unsqueeze(1)
        cross_att_mask_ot = cross_att_mask_to = self_att_mask_ot

        x_t_ = self.self_attention_t(x_t, x_t, x_t, self_att_mask_ot, True)

        if not self.disable_cross_attention and self.isolate_subnetwork not in {"t", "o"}:
            x_to_ = self.cross_attention_to(x_t_, x_o, x_o, cross_att_mask_ot, True)
        else:
            x_to_ = x_t_

        x_o_ = self.self_attention_o(x_o, x_o, x_o, self_att_mask_ot, True)

        if not self.disable_cross_attention and self.isolate_subnetwork not in {"o", "t"}:
            x_ot_ = self.cross_attention_ot(x_o_, x_t, x_t, cross_att_mask_to, True)
        else:
            x_ot_ = x_o_

        if self.n_inputs == 2:
            out_t = self.feed_forwards[0](x_to_ + x_s)
            out_o = self.feed_forwards[1](x_ot_ + x_s)
            return out_t, out_o

        self_att_mask_v = active_entries_vitals.repeat(1, 1, x_v.size(1)).unsqueeze(1)

        cross_att_mask_ot_v = (
            active_entries_vitals.squeeze(-1).unsqueeze(1)
            * active_entries_treat_outcomes
        ).unsqueeze(1)

        cross_att_mask_v_ot = (
            active_entries_treat_outcomes.squeeze(-1).unsqueeze(1)
            * active_entries_vitals
        ).unsqueeze(1)

        if not self.disable_cross_attention and self.isolate_subnetwork not in {"t", "v"}:
            x_tv_ = self.cross_attention_tv(x_t_, x_v, x_v, cross_att_mask_ot_v, True)
        else:
            x_tv_ = 0.0

        if not self.disable_cross_attention and self.isolate_subnetwork not in {"o", "v"}:
            x_ov_ = self.cross_attention_ov(x_o_, x_v, x_v, cross_att_mask_ot_v, True)
        else:
            x_ov_ = 0.0

        x_v_ = self.self_attention_v(x_v, x_v, x_v, self_att_mask_v, True)

        if not self.disable_cross_attention and self.isolate_subnetwork not in {"v", "t"}:
            x_vt_ = self.cross_attention_vt(x_v_, x_t, x_t, cross_att_mask_v_ot, True)
        else:
            x_vt_ = x_v_

        if not self.disable_cross_attention and self.isolate_subnetwork not in {"v", "o"}:
            x_vo_ = self.cross_attention_vo(x_v_, x_o, x_o, cross_att_mask_v_ot, True)
        else:
            x_vo_ = 0.0

        out_t = self.feed_forwards[0](x_to_ + x_tv_ + x_s)
        out_o = self.feed_forwards[1](x_ot_ + x_ov_ + x_s)
        out_v = self.feed_forwards[2](x_vt_ + x_vo_ + x_s)

        return out_t, out_o, out_v
