
# Steps
1. Get the Llama 2 checkpoints by following the [Meta instructions](https://github.com/facebookresearch/llama). Once we have those checkpoints, we have to convert them into the llama2.c format.
For this we need to install the python dependencies (`pip install -r requirements.txt`) and then use the `export.py` file, e.g. for 7B model

6. python export.py llama2_7b_chat.bin --meta-llama /path/to/7B-chat
7. ./run llama2_7b.bin <strong>or</strong> ./run llama2_7b_chat.bin -m chat  <strong>or</strong> ./runq llama2_7b_q80.bin -m chat

Every weight in the model checkpoint equalts to 32bits(=4 bytes), therefore 7B parameters times 4 bytes = 28GB. That is the size of the model checkpoint

4. 
| Components | Llama2 | GPT-1/2 |
| --- | --- | --- |
| Embeddings | RoPE | Absolute/learned positional  |
| Non-linearity in the MLP| SwiGLU   | GELU  |
| Normalzation layer| RMSNorm | LayerNorm |
| Bias on Linear Layers | True | True |
- Rotary Position Embedding ([RoPE](https://arxiv.org/abs/2104.09864)) Position encoding recently has shown effective in the transformer architecture. It enables valuable supervision for dependency modeling between elements at different positions of the sequence. RoPE is used to to effectively leverage the positional information. Specifically, the proposed RoPE encodes the absolute position with a rotation matrix and meanwhile incorporates the explicit relative position dependency in self-attention formulation. Notably, RoPE enables valuable properties, including the flexibility of sequence length, decaying inter-token dependency with increasing relative distances, and the capability of equipping the linear self-attention with relative position encoding.
- [Absolut Position Encodings](https://paperswithcode.com/method/absolute-position-encodings) where positional encodings are added to the input embeddings at the bottoms of the encoder and decoder sta cks. 
- Multi-layer Perceptron (MLP) 
 is a misnomer for a modern feedforward artificial neural network, consisting of fully connected neurons with a nonlinear kind of activation function, organized in at least three layers, notable for being able to distinguish data that is not linearly separable. Thanks to self-attention transformers are able to learn to focus on relevant parts of the sequence, even far in time, whereas simple MLPs cannot do this. Moreover, positional encoding provides the transformer to use the sequence order information, enabling it to learn to attend even to relative positions.
 - Swish is a non-monotonic activation function that was proposed by Google researchers in 2017. Swish is defined as follows:
Swish(x) = x * sigmoid(beta * x)
- Grated Linear Units (GLU) Gated Linear Units (GLU) are a type of activation function that were proposed by researchers at Microsoft in 2016. GLU is defined as follows:
GLU(x) = x * sigmoid(Wx + b)
where 'W' and 'b' are trainable parameters.
 - SwiGLU is a combination of Swish and GLU activation functions. SwiGLU is defined as follows:
SwiGLU(x) = x * sigmoid(beta * x) + (1 - sigmoid(beta * x)) * (Wx + b)
where 'W', 'b', and 'beta' are trainable 
 - RMSNorm: RMSNorm regularizes the summed inputs to a neuron in one layer ac- cording to root mean square (RMS), giving the model re-scaling invariance property and implicit learning rate adaptation ability. RMSNorm is computationally simpler and thus more efficient than LayerNorm.
 - Layer normalization (LayerNorm) is a technique to normalize the distributions of intermediate layers.

More information for the above can be found [here](https://www.ai-contentlab.com/2023/03/swishglu-activation-function.html)




## Parameters and Training Guide

1. **Transformer Parameters**
   - `dim`: Dimensionalityt o model, for example here is 768
   - `n_layers`: Number of layers in the transforme is 12
   - `n_heads`:  Number of attention heads in the Transformer is 12

   Refer to the Chinchilla paper's table for the pattern of how these parameters grow or shrink together.

2. **Max Context Length**
   - This should be the number of tokens that matter to predict the next token.  Set based on the problem (e.g., Llama 2 uses 2048).

3. **Total Batch Size per Update**
   - Aim for around 100K tokens for medium-sized applications.
   - Max out `batch_size` to system limit. (e.g. mine was 16 in a recent run because after that my GPU runs out of memory)
   - Increase `gradient_accumulation_steps` to reach the target total batch size (~100K tokens).
   

4. **Learning Rate (LR)**
   - Adjust based on network size.
   - Small networks can use a larger LR (e.g., 1e-3 or higher).
   - Larger networks need lower LRs.
   - Suggested: 3e-4 for most medium-sized applications.

5. **Max Iterations (max_iters)**
   -  length of training. Experiment with different settings.
   - Example: 200K (considered a bit too high in the provided example).

6. **Example Training Settings for a 110M Model**
   - `dim`: 768, `n_layers`: 12, `n_heads`: 12.
   - Sequence length: 1024, Batch size: 16.
   - `gradient_accumulation_steps`: 8 (needed to reach ~100K tokens).
   - Learning rate: 4e-4 (considered a bit too low).
   - Max iterations: 200K (considered a bit too high).
   - Dropout: 0.1.
       - N.tokens per update = batch_size x seq_length x gradient_acc_steps = 16 x 1024 x 8 = 131072
        - Therefore, each batch processed during training contains 16 sequences, and each sequence has 1024 tokens. The total number of tokens in each batch would be the product of these two values: 16 × 1024 = 16384 tokens per batch.


7. **Training Setup**
   - Distributed Data Parallel (DDP) on 4 GPUs on a cloud machine.
   - Training duration: ~1 day.
   

8. **Recommendation**
   - Primarily tune mentioned parameters and leave most others unchanged.





# General Information

## 1. Domain

In the context of machine learning and deep learning, a "domain" refers to a specific subject area or problem space in which the AI model is trained and applied. It represents the set of all possible inputs and outputs that the model deals with.

## 2. .bin File

In the context of deep learning, a `.bin` file format typically refers to a binary file that stores serialized data, often representing weights or parameters of a neural network model. These binary files are used to save and load the model parameters, enabling the model to be easily stored and transported. Storing these files in binary format is a common practice for efficiency and simplicity.

## 3. Make Run

When you run `make run`, you are typically instructing the make utility to execute a specific target defined in a Makefile. A Makefile is a text file that contains rules and dependencies for building software. The `run` target in a Makefile often specifies the steps needed to run the compiled program or perform other tasks associated with the execution phase. The Makefile is a configuration file.

## 4. Run Parameters

- `./run`: Execute the program or script named "run."
- `stories42M.bin`: This is likely the input model file.
- `-t 0.8`: Set the temperature for sampling to 0.8. Temperature controls the randomness of the LLM output.
- `-n 256`: Specify the number of steps or iterations for sampling (in this case, 256 steps).
- `-i "One day, sily met a Shoggoth"`: Provide an input prompt or seed for the text generation.

## 5. Gradient Accumulation

Gradient Accumulation is a technique that simulates a larger batch size by accumulating gradients from multiple small batches before performing a weight update. This technique can be helpful in scenarios where the available memory is limited, and the batch size that can fit in memory is small.

## 6. MLP vs. Transformer

In a traditional Multi-Layer Perceptron (MLP), a hidden layer consists of neurons, and each neuron is connected to every neuron in the previous and next layers. The number of neurons in a hidden layer determines the capacity of the network to capture complex patterns.

On the other hand, in the context of transformers, the term "head" refers to the parallel processing units within the self-attention mechanism. Each head operates independently, attending to different parts of the input sequence. The motivation behind having multiple heads is to enable the model to focus on different aspects or patterns in the data simultaneously.

The self-attention mechanism enables the model to weigh the importance of different positions in the input sequence when making predictions for a given position. The outputs from these heads are then typically concatenated or linearly combined to produce the final representation for each position in the sequence.

While both neurons in an MLP hidden layer and heads in a transformer contribute to the expressive power of the model by capturing different aspects of the input data, they operate in different ways. Neurons in an MLP have weighted connections to all neurons in the previous and next layers, allowing for complex learned feature representations. Transformer heads, on the other hand, process information in parallel and capture different patterns or dependencies in the input sequence. The combination of these parallel processing units helps transformers capture long-range dependencies efficiently, allowing for the efficient modeling of complex relationships in sequential data.

## 7. DistributedDataParallel (DDP)
DistributedDataParallel ([DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#:~:text=DistributedDataParallel%20(DDP)%20implements%20data%20parallelism,collective%20communications%20in%20the%20torch.)) implements data parallelism at the module level which can run across multiple machines. Applications using DDP should spawn multiple processes and create a single DDP instance per process. DDP uses collective communications in the torch.