# Extra Material
## Definitions
1. Domain: In the context of machine learning and deep learning, a "domain" refers to a specific subject area or problem space in which the AI model is trained and applied. It represents the set of all possible inputs and outputs that the model deals with
2. .bine file: 
In the context of deep learning, a .bin file format typically refers to a binary file that stores serialized data, often representing weights or parameters of a neural network model. These binary files are used to save and load the model parameters, enabling the model to be easily stored and transported.Storing these files in binary format, is a common practice for efficienct and simplicity. 
3. When you run make run, you are typically instructing the make utility to execute a specific target defined in a Makefile. A Makefile is a text file that contains rules and dependencies for building software. The run target in a Makefile often specifies the steps needed to run the compiled program or perform other tasks associated with the execution phase. The Makefile is a configuration file. 



4. 
    - `./run`: Execute the program or script named "run."
    - `stories42M.bin`: This is likely the input model file.
    - `-t 0.8`: Set the temperature for sampling to 0.8. Temperature controls the randomness of the LLM output.
    - `-n 256`: Specify the number of steps or iterations for sampling (in this case, 256 steps).
    - `-i "One day, sily met a Shoggoth"`: Provide an input prompt or seed for the text generation.

5. 

# Steps
1. Get the Llama 2 checkpoints by following the [Meta instructions](https://github.com/facebookresearch/llama). Once we have those checkpoints, we have to convert them into the llama2.c format.
For this we need to install the python dependencies (`pip install -r requirements.txt`) and then use the `export.py` file, e.g. for 7B model

6. python export.py llama2_7b_chat.bin --meta-llama /path/to/7B-chat
7. ./run llama2_7b.bin <strong>or</strong> ./run llama2_7b_chat.bin -m chat

Every weight in the model checkpoint equalts to 32bits(=4 bytes), therefore 7B parameters times 4 bytes = 28GB. That is the size of the model checkpoint

7. 
| Components | Llama2 | GPT-1/2 |
| --- | --- | --- |
| Embeddings | RoPE | Absolute/learned positional  |
| In the MLP| SwiGLU non-linearity  | GELU non-linearity |
| Normalzation layer| RMSNorm | LayerNorm |
| Bias on Linear Layers | True | True |
- Rotary Position Embedding ([RoPE](https://arxiv.org/abs/2104.09864)) Position encoding recently has shown effective in the transformer architecture. It enables valuable supervision for dependency modeling between elements at different positions of the sequence. RoPE is used to to effectively leverage the positional information. Specifically, the proposed RoPE encodes the absolute position with a rotation matrix and meanwhile incorporates the explicit relative position dependency in self-attention formulation. Notably, RoPE enables valuable properties, including the flexibility of sequence length, decaying inter-token dependency with increasing relative distances, and the capability of equipping the linear self-attention with relative position encoding.
- [Absolut Position Encodings](https://paperswithcode.com/method/absolute-position-encodings) where positional encodings are added to the input embeddings at the bottoms of the encoder and decoder sta cks. 
- Multi-layer Perceptron (MLP) 
 is a misnomer for a modern feedforward artificial neural network, consisting of fully connected neurons with a nonlinear kind of activation function, organized in at least three layers, notable for being able to distinguish data that is not linearly separable. Thanks to self-attention transformers are able to learn to focus on relevant parts of the sequence, even far in time, whereas simple MLPs cannot do this. Moreover, positional encoding provides the transformer to use the sequence order information, enabling it to learn to attend even to relative positions.
l
