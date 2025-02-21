import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import heapq
import math
from transformers import AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_TOKEN = "[SOS]"
EOS_TOKEN = "[EOS]"
SEP_TOKEN = "[SEP]"
UNK_TOKEN = "[UNK]"
max_sequence_length = 40
tokenizer = AutoTokenizer.from_pretrained("path_to_BERT_Tokenizer")
special_tokens = {"additional_special_tokens": [SOS_TOKEN, EOS_TOKEN, SEP_TOKEN]}
tokenizer.add_special_tokens(special_tokens)
vocab = tokenizer.get_vocab()
index2word = {idx: token for token, idx in vocab.items()}
vocab_size = len(tokenizer)

def texts_to_sequences(text):
    return tokenizer.encode(text, add_special_tokens=False)

def pad_sequence(seq, maxlen):
    padded = np.zeros((1, maxlen), dtype=int)
    trunc = seq[:maxlen]
    padded[0, :len(trunc)] = trunc
    return padded

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
    def __init__(self, vocab_size, d_model=256, nhead=4, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=512, dropout=0.1, max_len=max_sequence_length):
        super().__init__()
        self.d_model = d_model
        self.encoder_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        self.pos_decoder = PositionalEncoding(d_model, dropout, max_len)
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_decoder_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
    def forward(self, src, tgt):
        src_key_padding_mask = (src == 0)
        tgt_key_padding_mask = (tgt == 0)
        src = self.encoder_embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src.transpose(0, 1))
        tgt = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_decoder(tgt.transpose(0, 1))
        for layer in self.encoder_layers:
            src = layer(src, src_mask=None, src_key_padding_mask=src_key_padding_mask)
        tgt_mask = generate_square_subsequent_mask(tgt.size(0))
        output = tgt
        for layer in self.decoder_layers:
            output = layer(output, src, tgt_mask=tgt_mask, memory_mask=None, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=src_key_padding_mask)
        output = self.fc_out(output)
        return output.transpose(0, 1)

state_dict_path = "path_for_saving_state/empathetic_transformer.pt"
model = TransformerModel(vocab_size, d_model=256, nhead=4, num_encoder_layers=5, num_decoder_layers=5, dim_feedforward=512, dropout=0.1, max_len=max_sequence_length).to(device)
model.load_state_dict(torch.load(state_dict_path, map_location=device))
model.eval()

def generate_response(input_text, max_length=40, temperature=0.7, top_k=20, repetition_penalty=1.2):
    context = f"{SOS_TOKEN} {input_text} {SEP_TOKEN}"
    input_seq = texts_to_sequences(context)
    input_seq = pad_sequence(input_seq, max_length)
    encoder_input = torch.tensor(input_seq, dtype=torch.long, device=device)
    start_token = vocab.get(SOS_TOKEN, 0)
    decoder_input = torch.tensor([[start_token]], dtype=torch.long, device=device)
    token_counts = {}
    for _ in range(max_length - 1):
        with torch.no_grad():
            outputs = model(encoder_input, decoder_input)
        logits = outputs[:, -1, :] / temperature
        for token, count in token_counts.items():
            logits[0, token] /= repetition_penalty ** count
        topk_logits, topk_indices = torch.topk(logits, top_k)
        probs = F.softmax(topk_logits, dim=-1)
        next_token_idx = torch.multinomial(probs, 1)
        next_token = topk_indices.gather(1, next_token_idx)
        token_id = next_token.item()
        token_counts[token_id] = token_counts.get(token_id, 0) + 1
        decoder_input = torch.cat([decoder_input, next_token], dim=1)
        if token_id == vocab.get(EOS_TOKEN, 0):
            break
    generated_ids = decoder_input[0].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text.strip()

def beam_search_decoder(input_text, beam_width=5, max_len=max_sequence_length, length_penalty=1.0):
    context = f"{SOS_TOKEN} {input_text} {SEP_TOKEN}"
    input_seq = texts_to_sequences(context)
    input_seq = pad_sequence(input_seq, max_len)
    encoder_input = torch.tensor(input_seq, dtype=torch.long, device=device)
    start_token = vocab.get(SOS_TOKEN, 0)
    beams = [(0.0, [start_token])]
    for _ in range(max_len - 1):
        candidates = []
        for score, seq in beams:
            if seq[-1] == vocab.get(EOS_TOKEN, 0):
                candidates.append((score, seq))
                continue
            decoder_input = torch.tensor([seq], dtype=torch.long, device=device)
            with torch.no_grad():
                outputs = model(encoder_input, decoder_input)
            logits = outputs[0, -1, :]
            log_probs = torch.log(F.softmax(logits, dim=-1) + 1e-10)
            topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)
            for k in range(beam_width):
                next_token = topk_indices[k].item()
                new_seq = seq + [next_token]
                new_score = score + topk_log_probs[k].item()
                norm_score = new_score / (len(new_seq) ** length_penalty)
                candidates.append((norm_score, new_seq))
        beams = heapq.nlargest(beam_width, candidates, key=lambda x: x[0])
        if all(seq[-1] == vocab.get(EOS_TOKEN, 0) for _, seq in beams):
            break
    best_score, best_seq = beams[0]
    generated_text = tokenizer.decode(best_seq, skip_special_tokens=True)
    return generated_text.strip()

def chat():
    print("SARAH5 is ready! Type 'exit' to stop.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("SARAH5: Goodbye!")
            break
        response = generate_response(user_input)
        print(f"SARAH5: {response}")

if __name__ == "__main__":
    chat()
