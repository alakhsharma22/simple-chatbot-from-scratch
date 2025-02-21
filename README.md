# **Transformer-Based Text Generation Model**

This repository contains an implementation of a **Transformer-based text generation model** built from scratch using **PyTorch**. The model is designed for text generation and utilizes **pretrained tokenizers** along with a **custom Transformer-based architecture**.

While the model performs well in generating responses, it is a **basic version**, and responses may not be accurate. Future improvements can enhance the quality of generated text.

---

## **How the Model Works**
The model follows a sequence-to-sequence (seq2seq) architecture with an **encoder-decoder structure**. Here’s how it works:

1. **Tokenization**:
   - The input text is first **tokenized** using the `bert-base-uncased` tokenizer.
   - Special tokens like `[SOS]` (Start of Sentence) and `[EOS]` (End of Sentence) are added.
   - The text is converted into numerical token IDs.

2. **Embedding and Positional Encoding**:
   - Tokens are converted into **word embeddings**, which are dense vector representations.
   - **Positional encodings** are added to retain sequential information.

3. **Encoder (Transformer Layers)**:
   - The **encoder** consists of multiple **Transformer blocks** (self-attention + feed-forward layers).
   - It processes the input text and generates a **context-aware representation**.

4. **Decoder with Attention**:
   - The decoder takes the encoder output and **previously generated tokens** to predict the next token.
   - It uses **Masked Multi-Head Self-Attention** to prevent looking at future tokens.
   - An additional **Encoder-Decoder Attention Layer** helps focus on relevant parts of the input.

5. **Beam Search Decoding**:
   - Instead of generating a single response greedily, beam search keeps multiple candidates and selects the most **probable sequence**.

---

## **Model Architecture**
The model follows the **Transformer Encoder-Decoder** structure with multiple layers and attention heads. Below is a breakdown of its key components:

| **Component**               | **Description** |
|----------------------------|---------------|
| **Input Tokenization**      | Tokenizes text using `bert-base-uncased`. |
| **Word Embedding Layer**    | Converts tokenized words into dense vectors. |
| **Positional Encoding**     | Adds sequential information to embeddings. |
| **Encoder Layers**          | Stack of Transformer blocks with self-attention. |
| **Multi-Head Self-Attention** | Allows the model to focus on multiple words simultaneously. |
| **Feed-Forward Network (FFN)** | Applies non-linearity for better feature learning. |
| **Dropout Layers**          | Prevents overfitting. |
| **Layer Normalization**      | Stabilizes training and improves gradient flow. |
| **Decoder Layers**          | Uses previous outputs and encoder outputs for prediction. |
| **Masked Multi-Head Attention** | Prevents peeking at future words. |
| **Encoder-Decoder Attention** | Focuses on relevant encoder outputs for response generation. |
| **Beam Search Decoding**     | Generates high-quality responses by considering multiple candidates. |

---

## **Model Summary**

The model follows a **Transformer-based Encoder-Decoder** architecture. Below is a detailed breakdown of its components:

| **Layer (Type)**                | **Output Shape**                      | **Parameters**  | **Description** |
|---------------------------------|---------------------------------|--------------|----------------|
| **Embedding Layer**              | (batch_size, seq_len, d_model)   | N            | Converts input tokens into dense vector representations. |
| **Positional Encoding**          | (batch_size, seq_len, d_model)   | 0            | Adds sequence order information to embeddings. |
| **Transformer Encoder (N=5)**    | (batch_size, seq_len, d_model)   | N            | Stack of 6 self-attention and feed-forward layers. |
| **Multi-Head Attention**         | (batch_size, seq_len, d_model)   | N            | Allows focusing on multiple words at once. |
| **Feed-Forward Network (FFN)**   | (batch_size, seq_len, d_model)   | N            | Applies non-linearity to improve learning. |
| **Dropout**                      | (batch_size, seq_len, d_model)   | 0            | Prevents overfitting by randomly dropping units. |
| **Layer Normalization**          | (batch_size, seq_len, d_model)   | 0            | Normalizes features to stabilize training. |
| **Transformer Decoder (N=5)**    | (batch_size, seq_len, d_model)   | N            | Stack of 6 layers for sequence generation. |
| **Masked Multi-Head Attention**  | (batch_size, seq_len, d_model)   | N            | Prevents peeking at future words during decoding. |
| **Encoder-Decoder Attention**    | (batch_size, seq_len, d_model)   | N            | Ensures the decoder attends to relevant encoder outputs. |
| **Fully Connected Output Layer** | (batch_size, seq_len, vocab_size) | N           | Produces a probability distribution over vocabulary. |

**Total Parameters**: ~50 Million (Depending on Embedding Size and Layer Count)  
**Trainable Parameters**: ~50 Million  
**Non-Trainable Parameters**: 0  

The model consists of **5 encoder layers**, **5 decoder layers**, and **8 attention heads per layer**.

---

## **Tokenizer**
The model uses a **pretrained BERT tokenizer** from Hugging Face’s `transformers` library:

- **Tokenizer Used**: `bert-base-uncased`
- **Special Tokens**:
  - `[SOS]` - Start of sentence
  - `[EOS]` - End of sentence
  - `[SEP]` - Separator token
  - `[UNK]` - Unknown token

The tokenizer is responsible for:
1. **Tokenizing input text** into word/subword tokens.
2. **Converting tokens to numerical indices**.
3. **Ensuring input sequences adhere to the model's length constraints**.

---

## **Data Preparation**
The dataset used is an **emotion-based dataset** that consists of different fields:

- **Situation**: Context of the conversation.
- **Emotion**: Emotion associated with the text.
- **Empathetic Dialogues**: Conversations exhibiting empathy.
- **Labels**: Categorized emotions.

### **Steps for Data Processing**
1. **Load dataset** from CSV using `pandas`.
2. **Preprocess text**: Convert to lowercase, remove unnecessary symbols, and tokenize.
3. **Convert tokens to tensor format** for PyTorch.
4. **Create DataLoader**: Batch and shuffle data for training.

---

## **Generating Responses**
After training, the model can generate text using the **beam search** decoding strategy.

### **Beam Search Decoding**
Beam search is an advanced decoding method that:
- Maintains multiple candidate sequences at each step.
- Selects the most probable sequence instead of the greedy approach.
- Provides more coherent and high-quality generated text.

**Steps for Response Generation:**
1. **Input a prompt** to the trained model.
2. **Tokenize and process the input**.
3. **Generate output** using beam search decoding.
4. **Convert tokenized response back to text**.

---

## **Future Improvements**
- Improve training data quality for better responses.
- Fine-tune the model on larger datasets.
- Implement an attention visualization tool.
- Train on larger Transformer architectures for improved text quality.

---
