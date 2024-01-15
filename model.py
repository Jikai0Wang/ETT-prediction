import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

class LSTM(nn.Module):
    def __init__(self,num_features,hidden_size,num_layers,output_time_steps):
        super(LSTM_NAR, self).__init__()

        self.output_time_steps=output_time_steps #输出序列长度
        self.lstm=nn.LSTM(num_features, hidden_size, num_layers,batch_first=True,bidirectional=False)
        self.head=nn.Linear(hidden_size,num_features * output_time_steps)

    def forward(self,input_ids,labels):
        _, (hn, _) = self.lstm(input_ids)
        output = self.head(hn[-1, :, :])
        output = output.view(output.size(0), self.output_time_steps, -1)
        loss=None
        if labels!=None:
            loss_fn = torch.nn.MSELoss()
            loss=loss_fn(output,labels)
        return loss,output


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class Transformer(nn.Module):
    def __init__(self, num_features,d_model, nhead, num_layers, output_time_steps,dropout=0.05):
        super(Transformer, self).__init__()
        self.d_model=d_model
        self.num_features=num_features
        self.output_time_steps=output_time_steps
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.encoder_embedding = nn.Linear(num_features, d_model) #加入嵌入层以匹配特征维度与模型隐状态维度
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,dropout=dropout),
            num_layers=num_layers
        )
        self.head = nn.Linear(d_model, num_features * output_time_steps)
        self.use_position_embedding=True


    def forward(self, input_ids, labels):
        hidden_states = self.encoder_embedding(input_ids).permute(1, 0, 2)
        if self.use_position_embedding:
            hidden_states+=self.position_embedding(input_ids).permute(1, 0, 2)
        encoder_output = self.transformer_encoder(hidden_states).permute(1, 0, 2)
        output = self.head(encoder_output[:,-1,:])
        output = output.view(output.size(0), self.output_time_steps, -1)
        loss = None
        if labels != None:
            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(output, labels)
        return loss,output


@dataclass
class FusedAttentionTransformerConfig:
    hidden_size: int
    num_attention_heads: int
    dropout_prob: float
    intermediate_size: int
    input_len: int
    output_len: int
    feature_size: int
    num_layers: int


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        # Mask heads if we want to
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        outputs = context_layer
        return outputs


class SelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.bn_size = config.input_len if config.is_encoder else config.output_len
        self.LayerNorm = nn.BatchNorm1d(self.bn_size)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class AttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = SelfAttention(config)
        self.output = SelfOutput(config)

    def forward(self, hidden_states: torch.Tensor):
        self_outputs = self.self(hidden_states)
        attention_output = self.output(self_outputs, hidden_states)
        outputs = attention_output
        return outputs


class Intermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class Output(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.bn_size = config.input_len if config.is_encoder else config.output_len
        self.LayerNorm = nn.BatchNorm1d(self.bn_size)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = AttentionLayer(config)
        self.intermediate = Intermediate(config)
        self.output = Output(config)

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    def forward(self, hidden_states: torch.Tensor):
        self_attention_outputs = self.attention(hidden_states)
        attention_output = self_attention_outputs
        layer_output = self.feed_forward_chunk(attention_output)
        outputs = layer_output
        return outputs


class FusedAttentionConnectBlock(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.q_affine = nn.Linear(config.input_len, config.output_len)
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        fused_query = self.q_affine(hidden_states.permute(0, 2, 1)).permute(0, 2, 1)
        mixed_query_layer = self.query(fused_query)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        # Mask heads if we want to
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        outputs = context_layer
        return outputs


class FusedAttentionModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_embedding = nn.Linear(config.feature_size, config.hidden_size)
        config.is_encoder = True
        self.input_layers = nn.ModuleList([Layer(config) for _ in range(config.num_layers)])
        self.fused_attention_connect = FusedAttentionConnectBlock(config)
        config.is_encoder = False
        self.output_layers = nn.ModuleList([Layer(config) for _ in range(config.num_layers)])
        self.head = nn.Linear(config.hidden_size, config.feature_size)
        self.first_pos = PositionalEmbedding(config.hidden_size)
        self.second_pos = PositionalEmbedding(config.hidden_size)

    def forward(self, x, labels=None):
        hidden_states = self.input_embedding(x) + self.first_pos(x)
        for i, layer_module in enumerate(self.input_layers):
            hidden_states = layer_module(hidden_states)
        hidden_states = self.fused_attention_connect(hidden_states)
        hidden_states = hidden_states + self.second_pos(hidden_states)
        for i, layer_module in enumerate(self.output_layers):
            hidden_states = layer_module(hidden_states)
        output = self.head(hidden_states)
        loss = None
        if labels != None:
            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(output, labels)
        return loss, output