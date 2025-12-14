# Optimizing Large Language Model Inference with 8-bit Quantization 
## 1.	Understanding Causal Transformer Architectures
### A.	The Transformer 
The Transformer appeared in 2017 with the paper Attention Is All You Need (Vaswani and al.). It is based on the Attention mechanism. Attention allows the model to evaluate the importance of the words in a sentence compared to the word it is processing. 
The first step is to tokenize the text to get numbers (and not strings) and thus be able to make computations. Every token generates 3 vectors: Query (Q), Key (K), and Value (V). 
To get Q, K and V, we multiply the input matrix $X$ (representing the entire sequence of tokens) by different weight matrices $W_{Q}, W_{K}$ and $W_{V}$
-	The Query ($Q$) is what the token is looking for. It is “asking” the other tokens in the sequence whether they are relevant to the current context.
-	The Key ($K$) is what the token is about. We compare the Query and the Key to determine if a token is relevant. We compute the compatibility scores by calculating the dot product between $Q$ and the transpose of $K$ ($Q \cdot K^T$).
-	The Value ($V$) contains the content of the token, the “useful” information that will be transmitted. If a token is judged relevant (high score via Q and K), its Value will be emphasized in the output.

The raw compatibility scores (logits) resulting from the dot product of $Q$ and $K$ are passed through a Softmax function. This operation normalizes the scores across the sequence, converting them into positive probabilities that sum to 1. This acts as a filter: it highlights the most relevant tokens (high probability) while suppressing irrelevant ones (near-zero probability), creating Attention Weights. It says roughly: take 80% from word 1, 15% from word 2, 3% from word 3, etc.
Then we multiply these scores by the values of these tokens. The Attention Weights are used to compute a weighted sum of the Value vectors ($V$). This aggregates information from all tokens in the sequence based on their relevance.
$$\text{Output} = \sum (\text{AttentionWeight}_i \times V_i)$$
The result is a new context-aware vector representation for the current token, enriched with information retrieved from relevant parts of the sequence.14 We get this formula:
$$\text{Attention}(Q, K, V) = \underbrace{\text{Softmax}\left(\frac{Q \cdot K^T}{\sqrt{d_k}}\right)}_{\text{Weights}} \cdot \underbrace{V}_{\text{Values}}$$

### B.	The Transformer Mechanism (Encoder-Decoder)

When it was first conceived, the transformer was thought for translation (e.g., English to French). It is composed of two entities: the Encoder and the Decoder. The goal of the Encoder is to analyze the sentence in English, while the Decoder is writing/generating the sentence in French, word by word.
The Encoder is responsible for processing the entire input sequence. It uses the Self-Attention mechanism we just described to analyze and understand the sentence. It is important to note that the encoder can look at the entirety of the sentence (preceding and succeeding words).
Following the attention mechanism, the data passes through a Feed-Forward Network (FFN), which processes each token individually to add complexity. To stabilize learning, a Layer Normalization step (often "Add & Norm") is applied after each of these operations. 
The final output of the Encoder is the sequence of context-rich vectors (matrices $K$ and $V$) that represent the semantic meaning of the input.
The Decoder is responsible for generating the output sequence, one token at a time. The decoder cannot see the future; it only has access to the words it has already generated. It thus uses Masked Self-Attention. It also has access to information from the Encoder through Cross-Attention: the queries $Q$ are the already generated words, while $K$ and $V$ come from the encoder.

### C.	Causal Transformer 
Modern LLMs, like GPT or LLaMA, utilize a Decoder-Only architecture instead of Encoder-Decoder.20 They are referred to as Causal Transformers.
What architectural changes are there compared to the Encoder-Decoder?
-	No Encoder: There is no separate “source” text to encode. The input is simply a prompt or the beginning of the sentence being generated.
-	No Cross-Attention: Since there is no encoder, the model entirely relies on Self-Attention.
-	Strict Masking: The attention matrix is modified utilizing a lower-triangular mask. This ensures that for any given token at position $t$, the model can only attend to previous tokens $[1, t]$. The attention scores for tokens $> t$ are set to $-\infty$ (resulting in zero probability after Softmax), effectively "blinding" the model to the future. Here, $Q$ represents the current token $t$ looking for context, while $K, V$ represent the history of tokens $[1, t]$ (including itself).

During the generation phase, the process is auto regressive:
1.	The model processes the input context.
2.	It samples the next token based on the probability distribution.
3.	This predicted token is appended to the sequence.
4.	The new sequence becomes the input for the next step.
This cycle repeats until an "End of Sequence" token is generated.

## 2.	Specificities of the LLaMA Architecture 
While LLaMA is based on the standard Causal Transformer architecture described above, it introduces three key modifications to improve stability and performance.

-	Pre-Normalization using RMSNorm: Unlike the original Transformer which normalizes the output of each sub-layer (Post-Norm), LLaMA normalizes the input of each layer (Pre-Norm). This architectural change, inspired by GPT-3, significantly improves training stability. Furthermore, instead of the standard LayerNorm function, LLaMA utilizes RMSNorm (Zhang and Sennrich, 2019). RMSNorm simplifies the computation by omitting the mean centering and only re-scaling the values, which saves computational resources without sacrificing performance.

-	SwiGLU Activation Function: In the Feed-Forward Network (FFN) layers, LLaMA replaces the standard ReLU activation function with SwiGLU. Empirical results show that SwiGLU significantly improves the performance of the model


-	Rotary Positional Embeddings (RoPE): Instead of adding absolute positional vectors to the input (which fixes a word to position 1, 2, 3...), LLaMA uses RoPE. It encodes the position by rotating the Query and Key vectors in space. The angle of rotation corresponds to the position. Thus, the attention mechanism only depends on the relative distance. So, if we have the same sentence but at another position it will be the same processing. This property also allows the model to generalize better to sequence lengths longer than those seen during training (extrapolation).


## 3.	8-bit Quantization
To optimize the inference of LLaMA, we implement 8-bit quantization. This technique reduces the memory footprint of the model weights and activations, allowing large models to run on consumer hardware with limited VRAM, like the free GPU of Google Colab. 

### A.	The concept of Quantization
Standard Deep Learning models are trained and stored in FP32 (32-bit Floating Point) or FP16.
-	FP32: Requires 4 bytes per parameter. A 7B model requires $\approx 28$ GB of VRAM.
-	INT8: Uses 8-bit Integers (1 byte per parameter). A 7B model requires $\approx 7$ GB of VRAM.

Quantization maps the continuous range of floating-point values to a discrete set of integers $[-127, 127]$.
$$X_{\text{int8}} = \text{round}\left(\frac{X_{\text{fp32}}}{S}\right)$$
Where $S$ is the scaling factor derived from the absolute maximum value of the tensor.

### B.	The Challenge: Emergent Features (Outliers)
Naive quantization often leads to severe performance degradation in Large Language Models.
As detailed in the paper LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale (Dettmers et al., 2022), LLMs larger than 6.7B parameters exhibit "emergent features" characterized by extreme outliers in specific feature dimensions.
These outliers are significantly larger than the rest of the data. Naively scaling the quantization range to accommodate these outliers causes the vast majority of "normal" values to be quantized to zero, destroying the model's information.

### C.	The solution: Vector-wise Quantization & Mixed Precision
To solve this, we utilize the technique implemented in the bitsandbytes library:
-	Vector-wise Quantization: Instead of one scaling factor for the whole matrix, scaling factors are calculated for each row or column independently, preserving precision.
-	Mixed-Precision Decomposition:

            o	The algorithm detects dimensions (columns) containing outliers (values exceeding a threshold).
  
            o	These outlier dimensions are kept in FP16 (16-bit float) to preserve high precision. These outliers represent only 0.1 % of the dimensions, allowing the quantization to INT8 on the remaining 99.9 % to still allow huge memory gains
  
            o	During matrix multiplication, the results are combined. This approach allows the reduction of memory usage by nearly 4x while maintaining inference performance almost identical to the FP16 baseline.

We will implement the 8-bit quantization to see these VRAM improvements. However, we will not be able to compare the quantization with the 32-bit model since the latter is too heavy for the free Colab GPU. 











