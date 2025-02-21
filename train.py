import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import math
from transformers import AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
file_path = "/home/postman/.Alakh/chatbot/emotion-emotion_69k.csv"
df = pd.read_csv(file_path)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df[['Situation', 'emotion', 'empathetic_dialogues', 'labels']].dropna()
df.rename(columns={'Situation': 'situation', 'emotion': 'emotion', 'empathetic_dialogues': 'empathetic_dialogues', 'labels': 'labels'}, inplace=True)
SOS_TOKEN = "[SOS]"
EOS_TOKEN = "[EOS]"
SEP_TOKEN = "[SEP]"
UNK_TOKEN = "[UNK]"
max_sequence_length = 40
sample_percentage = 1
batch_size = 32
epochs = 25
patience = 100
tokenizer = AutoTokenizer.from_pretrained("/home/postman/.Alakh/chatbot/bert-Tokenizer/cacheDir/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594")
special_tokens = {"additional_special_tokens": [SOS_TOKEN, EOS_TOKEN, SEP_TOKEN]}
tokenizer.add_special_tokens(special_tokens)

def preprocess_data_from_file(data, sample_percentage=1.0):
    input_texts = []
    output_texts = []
    data = data.sample(frac=sample_percentage, random_state=42) if sample_percentage < 1.0 else data
    for _, row in data.iterrows():
        context = f"{row['situation']} | {row['emotion']} | {row['empathetic_dialogues']}"
        response = row['labels']
        input_seq = f"{SOS_TOKEN} {context} {SEP_TOKEN}"
        output_seq = f"{SOS_TOKEN} {response} {EOS_TOKEN}"
        input_texts.append(input_seq)
        output_texts.append(output_seq)
    return input_texts, output_texts

train_size = int(0.8 * len(df))
train_df = df[:train_size]
val_df = df[train_size:]
train_inputs, train_outputs = preprocess_data_from_file(train_df, sample_percentage=0.8)
val_inputs, val_outputs = preprocess_data_from_file(val_df)

def texts_to_sequences(texts):
    sequences = []
    for text in texts:
        seq = tokenizer.encode(text, add_special_tokens=False)
        sequences.append(seq)
    return sequences

def pad_sequences(sequences, maxlen):
    padded = np.zeros((len(sequences), maxlen), dtype=int)
    for i, seq in enumerate(sequences):
        trunc = seq[:maxlen]
        padded[i, :len(trunc)] = trunc
    return padded

train_input_sequences = pad_sequences(texts_to_sequences(train_inputs), max_sequence_length)
train_output_sequences = pad_sequences(texts_to_sequences(train_outputs), max_sequence_length)
val_input_sequences = pad_sequences(texts_to_sequences(val_inputs), max_sequence_length)
val_output_sequences = pad_sequences(texts_to_sequences(val_outputs), max_sequence_length)

train_encoder_tensor = torch.tensor(train_input_sequences, dtype=torch.long)
train_decoder_input_tensor = torch.tensor(train_output_sequences[:, :-1], dtype=torch.long)
train_decoder_output_tensor = torch.tensor(train_output_sequences[:, 1:], dtype=torch.long)
val_encoder_tensor = torch.tensor(val_input_sequences, dtype=torch.long)
val_decoder_input_tensor = torch.tensor(val_output_sequences[:, :-1], dtype=torch.long)
val_decoder_output_tensor = torch.tensor(val_output_sequences[:, 1:], dtype=torch.long)

train_dataset = TensorDataset(train_encoder_tensor, train_decoder_input_tensor, train_decoder_output_tensor)
val_dataset = TensorDataset(val_encoder_tensor, val_decoder_input_tensor, val_decoder_output_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
    return mask.to(device)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(0)].transpose(0, 1)
        return self.dropout(x)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt2, _ = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        return tgt

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_encoder_layers=5, num_decoder_layers=5, dim_feedforward=512, dropout=0.2, max_len=max_sequence_length):
        super().__init__()
        self.d_model = d_model
        self.encoder_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        self.pos_decoder = PositionalEncoding(d_model, dropout, max_len)
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_decoder_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
    def encode(self, src):
        m = self.encoder_embedding(src) * math.sqrt(self.d_model)
        m = self.pos_encoder(m.transpose(0, 1))
        for layer in self.encoder_layers:
            m = layer(m, None, src == 0)
        return m
    def decode(self, memory, tgt):
        t = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        t = self.pos_decoder(t.transpose(0, 1))
        tm = generate_square_subsequent_mask(t.size(0))
        for layer in self.decoder_layers:
            t = layer(t, memory, tm, None, tgt == 0, None)
        o = self.fc_out(t)
        return o.transpose(0, 1)
    def forward(self, src, tgt):
        m = self.encode(src)
        o = self.decode(m, tgt)
        return o

vocab_size = len(tokenizer)
model = TransformerModel(vocab_size, d_model=256, nhead=4, num_encoder_layers=5, num_decoder_layers=5, dim_feedforward=512, dropout=0.2, max_len=max_sequence_length).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=0)
best_val_loss = float('inf')
trigger_times = 0
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for src, tgt_inp, tgt_out in train_loader:
        src = src.to(device)
        tgt_inp = tgt_inp.to(device)
        tgt_out = tgt_out.to(device)
        optimizer.zero_grad()
        output = model(src, tgt_inp)
        output = output.reshape(-1, vocab_size)
        tgt_out = tgt_out.reshape(-1)
        loss = criterion(output, tgt_out)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for src, tgt_inp, tgt_out in val_loader:
            src = src.to(device)
            tgt_inp = tgt_inp.to(device)
            tgt_out = tgt_out.to(device)
            output = model(src, tgt_inp)
            output = output.reshape(-1, vocab_size)
            tgt_out = tgt_out.reshape(-1)
            loss = criterion(output, tgt_out)
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(val_loader)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    print(epoch + 1, avg_train_loss, avg_val_loss)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        trigger_times = 0
        torch.save(model.state_dict(), "/home/postman/.Alakh/chatbot/new2/empathetic_transformer2.pt")
    else:
        trigger_times += 1
        if trigger_times >= patience:
            break

model.load_state_dict(torch.load("/home/postman/.Alakh/chatbot/new2/empathetic_transformer2.pt"))
torch.save(model.state_dict(), "/home/postman/.Alakh/chatbot/new2/empathetic_transformer_final2.pt")
torch.save(model, "/home/postman/.Alakh/chatbot/new2/empathetic_transformer_full2.pth")