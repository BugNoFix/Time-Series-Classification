import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm as wn


# Dataset

class Dataset(torch.utils.data.Dataset):
    def __init__(self, cont, survey, body, labels, time=None):
        self.cont = cont
        self.survey = survey
        self.body = body
        self.labels = labels
        self.time = time
        self.use_time = time is not None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.use_time:
            inputs = (self.cont[idx], self.survey[idx], self.body[idx], self.time[idx])
        else:
            inputs = (self.cont[idx], self.survey[idx], self.body[idx])

        label = self.labels[idx]
        return inputs, label


# 1. RecurrentClassifier (RNN / LSTM / GRU)

class RecurrentClassifier(nn.Module):
    def __init__(
        self, 
        input_size, 
        hidden_size, 
        num_layers, 
        num_classes,
        rnn_type='GRU', 
        bidirectional=False, 
        dropout_rate=0.2
    ):
        super().__init__()

        rnn_map = {
            'RNN': nn.RNN,
            'LSTM': nn.LSTM,
            'GRU': nn.GRU
        }
        rnn_module = rnn_map[rnn_type]

        dropout_val = dropout_rate if num_layers > 1 else 0

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.rnn = rnn_module(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_val,
            bidirectional=bidirectional
        )

        out_dim = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        rnn_out, hidden = self.rnn(x)
        if self.rnn_type == 'LSTM':
            hidden = hidden[0]

        if self.bidirectional:
            hidden = hidden.view(self.num_layers, 2, -1, self.hidden_size)
            hidden = torch.cat([hidden[-1, 0], hidden[-1, 1]], dim=1)
        else:
            hidden = hidden[-1]

        return self.classifier(hidden)


# 2. CNN-GRU

class CNN_GRU(nn.Module):
    def __init__(
        self, 
        input_size, 
        num_classes, 
        hidden_size,
        num_layers, 
        dropout_rate, 
        kernel_size,
        bidirectional=False,
        time_cardinality=None,
        emb_dim_time=None
    ):
        super().__init__()

        self.use_time = time_cardinality is not None and emb_dim_time is not None
        if self.use_time:
            self.time_embedding = nn.Embedding(time_cardinality, emb_dim_time)
            input_size += emb_dim_time

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.conv_block = nn.Sequential(
            nn.Conv1d(input_size, hidden_size // 2, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(hidden_size // 2, hidden_size, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        out_dim = hidden_size * (2 if bidirectional else 1)
        self.norm = nn.LayerNorm(out_dim)

        self.classifier = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(out_dim // 2, num_classes)
        )

    def forward(self, x, time=None):
        if self.use_time:
            if time is None:
                raise ValueError("time=None but time expected")
            if time.dim() == 3:
                time = time.squeeze(-1)
            t = self.time_embedding(time)
            x = torch.cat([x, t], dim=-1)

        x = x.transpose(1, 2)
        x = self.conv_block(x)
        x = x.transpose(1, 2)

        _, h = self.gru(x)
        if isinstance(h, tuple):
            h = h[0]

        if self.bidirectional:
            h = h.view(self.num_layers, 2, h.size(1), self.hidden_size)
            h = torch.cat([h[-1,0], h[-1,1]], dim=1)
        else:
            h = h[-1]

        h = self.norm(h)
        return self.classifier(h)


# 3. EGRU (optional time embeddings)

class EGRU(nn.Module):
    def __init__(
        self,
        n_cont,
        survey_cardinality,
        body_cardinality,
        emb_dim_survey=3,
        emb_dim_body=2,
        hidden_size=64,
        num_classes=3,
        num_layers=2,
        bidirectional=True,
        dropout_rate=0.3,
        time_cardinality=None,
        emb_dim_time=None,
        use_attention=True,
    ):
        super().__init__()

        self.use_time = time_cardinality is not None and emb_dim_time is not None
        self.use_attention = use_attention
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        self.survey_embeddings = nn.ModuleList(
            [nn.Embedding(card, emb_dim_survey) for card in survey_cardinality]
        )
        self.body_embeddings = nn.ModuleList(
            [nn.Embedding(card, emb_dim_body) for card in body_cardinality]
        )

        if self.use_time:
            self.time_embedding = nn.Embedding(time_cardinality, emb_dim_time)
            time_dim = emb_dim_time
        else:
            self.time_embedding = None
            time_dim = 0

        self.cont_proj = nn.Sequential(
            nn.Linear(n_cont, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        input_dim = (
            hidden_size // 2 +
            len(survey_cardinality) * emb_dim_survey +
            len(body_cardinality) * emb_dim_body +
            time_dim
        )

        self.gru = nn.GRU(
            input_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_layers > 1 else 0
        )

        out_dim = hidden_size * (2 if bidirectional else 1)

        if use_attention:
            self.attn = nn.Sequential(
                nn.Linear(out_dim, out_dim // 2),
                nn.Tanh(),
                nn.Linear(out_dim // 2, 1)
            )

        self.fc = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(out_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, cont, surveys, body, time=None):
        B, T, _ = cont.shape

        cont_emb = self.cont_proj(cont)

        survey_embs = torch.cat(
            [emb(surveys[..., i]) for i, emb in enumerate(self.survey_embeddings)],
            dim=-1
        )

        body_embs = torch.cat(
            [emb(body[..., i]) for i, emb in enumerate(self.body_embeddings)],
            dim=-1
        )

        if self.use_time:
            if time is None:
                raise ValueError("time=None but model was initialized with time_cardinality")
            if time.dim() == 3:
                time = time.squeeze(-1)
            time_emb = self.time_embedding(time)
        else:
            time_emb = cont_emb.new_zeros(B, T, 0)

        x = torch.cat([cont_emb, survey_embs, body_embs, time_emb], dim=-1)
        out, _ = self.gru(x)

        if self.use_attention:
            attn_scores = self.attn(out)
            attn_weights = torch.softmax(attn_scores, dim=1)
            context = torch.sum(out * attn_weights, dim=1)
        else:
            context = out[:, -1]

        return self.fc(context)



# 4. CNN_EGRU

class CNN_EGRU(nn.Module):
    def __init__(
        self,
        n_cont, survey_cardinality, 
        body_cardinality, 
        time_cardinality=None,
        emb_dim_survey=3, 
        emb_dim_body=2, 
        emb_dim_time=7,
        hidden_size=64, 
        kernel_size=3,
        num_classes=3, 
        num_layers=2,
        bidirectional=True, 
        dropout_rate=0.45,
        use_attention=True
    ):
        super().__init__()

        self.use_attention = use_attention
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.use_time = time_cardinality is not None

        self.survey_embeddings = nn.ModuleList(
            [nn.Embedding(card, emb_dim_survey) for card in survey_cardinality]
        )
        self.body_embeddings = nn.ModuleList(
            [nn.Embedding(card, emb_dim_body) for card in body_cardinality]
        )

        if self.use_time:
            self.time_embedding = nn.Embedding(time_cardinality, emb_dim_time)
            time_dim = emb_dim_time
        else:
            self.time_embedding = None
            time_dim = 0

        self.cont_proj = nn.Sequential(
            nn.Linear(n_cont, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        input_dim = (
            hidden_size // 2 +
            len(survey_cardinality) * emb_dim_survey +
            len(body_cardinality) * emb_dim_body +
            time_dim
        )

        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_size, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )

        self.gru = nn.GRU(
            hidden_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_layers > 1 else 0
        )

        out_dim = hidden_size * (2 if bidirectional else 1)

        if use_attention:
            self.attn = nn.Sequential(
                nn.Linear(out_dim, out_dim // 2),
                nn.Tanh(),
                nn.Linear(out_dim // 2, 1)
            )

        self.fc = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(out_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, cont, surveys, body, time=None):
        B, T, _ = cont.shape

        cont_emb = self.cont_proj(cont)

        survey_embs = torch.cat(
            [emb(surveys[..., i]) for i, emb in enumerate(self.survey_embeddings)],
            dim=-1
        )

        body_embs = torch.cat(
            [emb(body[..., i]) for i, emb in enumerate(self.body_embeddings)],
            dim=-1
        )

        if self.use_time:
            if time is None:
                raise ValueError("time=None but model was initialized with time_cardinality")
            if time.dim() == 3:
                time = time.squeeze(-1)
            time_emb = self.time_embedding(time)
        else:
            time_emb = cont_emb.new_zeros(B, T, 0)

        x = torch.cat([cont_emb, survey_embs, body_embs, time_emb], dim=-1)

        x = x.transpose(1, 2)
        x = self.temporal_cnn(x)
        x = x.transpose(1, 2)

        out, _ = self.gru(x)

        if self.use_attention:
            attn_scores = self.attn(out)
            attn_weights = torch.softmax(attn_scores, dim=1)
            context = torch.sum(out * attn_weights, dim=1)
        else:
            context = out[:, -1]

        return self.fc(context)


# 5. Transformer

class Transformer(nn.Module):
    def __init__(
        self,
        n_cont,
        survey_cardinality,
        body_cardinality,
        time_cardinality=None,      
        num_classes=3,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=384,
        dropout=0.25,
        max_len=160,
        pooling="mean",
        activation="gelu",
        use_layer_norm=True,
        norm_first=True,
        emb_dim_time=7
    ):
        super().__init__()

        self.pooling = pooling
        self.use_time = time_cardinality is not None  

        self.conv = nn.Conv1d(n_cont, d_model, kernel_size=5, padding=2)
        self.conv_norm = nn.BatchNorm1d(d_model)

        self.survey_embeds = nn.ModuleList(
            [nn.Embedding(card, d_model) for card in survey_cardinality]
        )

        self.body_embeds = nn.ModuleList(
            [nn.Embedding(card, d_model) for card in body_cardinality]
        )

        if self.use_time:
            self.time_emb = nn.Embedding(time_cardinality, emb_dim_time)
            self.time_proj = nn.Linear(emb_dim_time, d_model)
        else:
            self.time_emb = None
            self.time_proj = None

        self.pos_emb = nn.Embedding(max_len, d_model)

        act = F.gelu if activation == "gelu" else F.relu

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=act,
            batch_first=True,
            norm_first=norm_first,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model) if use_layer_norm else None,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, cont, survey, body, time=None):

        B, T, _ = cont.shape

        x = cont.transpose(1, 2)      # (B, features, T)
        x = self.conv_norm(self.conv(x)).transpose(1, 2)

        survey_embs = [emb(survey[..., i]) for i, emb in enumerate(self.survey_embeds)]
        body_embs   = [emb(body[..., i]) for i, emb in enumerate(self.body_embeds)]

        h_survey = torch.stack(survey_embs).mean(0)
        h_body   = torch.stack(body_embs).mean(0)

        if self.use_time:
            if time is None:
                raise ValueError("time=None but model was configured with time_cardinality")
            if time.dim() == 3:
                time = time.squeeze(-1)
            t = self.time_proj(self.time_emb(time))
        else:
            t = 0

        h = x + h_survey + h_body + t

        pos = torch.arange(T, device=cont.device).unsqueeze(0).expand(B, T)
        h = h + self.pos_emb(pos)

        h = self.encoder(h)

        if self.pooling == "mean":
            pooled = h.mean(1)
        else:
            pooled = h.max(1).values

        return self.classifier(pooled)

# 6. TCN

class CausalConv1d(nn.Module):
    """Causal 1D convolution: pads on the left only so we never peek into the future."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1, use_weight_norm=True):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size,
            padding=0,
            dilation=dilation,
            bias=True
        )
        self.conv = wn(conv) if use_weight_norm else conv

    def forward(self, x):  
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation,
                 dropout=0.0, use_weight_norm=True):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch,  out_ch, kernel_size, dilation, use_weight_norm)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation, use_weight_norm)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None
        self._init_weights()

    def _init_weights(self):
        for m in [self.conv1.conv, self.conv2.conv]:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity='relu')
            if self.downsample.bias is not None:
                nn.init.zeros_(self.downsample.bias)

    def forward(self, x):  
        out = self.activation(self.conv1(x))
        out = self.dropout(out)
        out = self.activation(self.conv2(out))
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.activation(out + res)

class TCN(nn.Module):
    def __init__(self, in_channels, num_classes,
                 hidden_channels=64, num_levels=5, kernel_size=5,
                 dropout=0.1, use_weight_norm=True, use_gpool=True):
        super().__init__()
        layers = []
        ch_in = in_channels
        dilation = 1
        for _ in range(num_levels):
            layers.append(
                ResidualBlock(
                    ch_in,
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    use_weight_norm=use_weight_norm
                )
            )
            ch_in = hidden_channels
            dilation *= 2

        self.tcn = nn.Sequential(*layers)
        self.use_gpool = use_gpool
        self.head = nn.Sequential(
            nn.Linear(ch_in, ch_in),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(ch_in, num_classes),
        )

    def forward(self, x):  
        feat = self.tcn(x)              
        if self.use_gpool:
            feat = feat.mean(dim=-1)    
        else:
            feat = feat[..., -1]        
        return self.head(feat)
