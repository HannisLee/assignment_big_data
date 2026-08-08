# L12 Assignment

**Name:** LI HAN
**Student ID:** 33C26029
**GitHub Repository:** https://github.com/HannisLee/assignment_big_data

## Problem Statement

In this assignment, we explain why positional encoding, such as sinusoidal positional encoding, can be **added element-wise** to token embeddings in Transformers.

For example:

```text
Positional embedding = [1, 2, 3]
Token embedding      = [4, 5, 6]
Added representation = [5, 7, 9]
```

The question is not only whether the addition is mathematically possible. The deeper question is why this operation is meaningful in a Transformer model, and why adding position information to word or token information does not destroy the token representation.

## Short Answer

Positional encoding can be added to token embedding because both vectors have the same dimensionality, usually called \(d_{\text{model}}\), and both are continuous vectors in the model's representation space.

For a token at position \(i\), the input to the Transformer is:

$$
x_i = e_i + p_i
$$

where:

| Symbol | Meaning |
|---|---|
| \(e_i\) | token embedding of the token at position \(i\) |
| \(p_i\) | positional encoding for position \(i\) |
| \(x_i\) | final input representation passed into the Transformer |

The token embedding gives the model information about **what the token is**, while the positional encoding gives the model information about **where the token is**. Element-wise addition combines these two kinds of information into one vector of the same size.

## Why Transformers Need Positional Information

The self-attention mechanism in a Transformer compares tokens with each other using query, key, and value vectors. However, self-attention by itself does not naturally know the order of the input sequence.

For example, consider the two sentences:

```text
The dog chased the cat.
The cat chased the dog.
```

They contain many of the same words, but the meanings are different because the word order is different. A Transformer must therefore know not only the token identities, but also the positions of the tokens.

Without positional encoding, the model would process the input more like an unordered set of tokens. If the input tokens were rearranged, the self-attention operation would rearrange its outputs in the same way, but it would not have an independent signal telling it which token came first, second, or third.

Therefore, positional information must be injected into the input representation.

## Same Dimension Makes Element-wise Addition Possible

Token embeddings and positional encodings are designed to have the same length:

$$
e_i \in \mathbb{R}^{d_{\text{model}}}
$$

$$
p_i \in \mathbb{R}^{d_{\text{model}}}
$$

Because they are both \(d_{\text{model}}\)-dimensional vectors, they can be added element by element:

$$
x_i =
\begin{bmatrix}
e_{i,1} \\
e_{i,2} \\
\cdots \\
e_{i,d}
\end{bmatrix}
+
\begin{bmatrix}
p_{i,1} \\
p_{i,2} \\
\cdots \\
p_{i,d}
\end{bmatrix}
=
\begin{bmatrix}
e_{i,1}+p_{i,1} \\
e_{i,2}+p_{i,2} \\
\cdots \\
e_{i,d}+p_{i,d}
\end{bmatrix}
$$

This is exactly the meaning of "added to" in the assignment: each coordinate of the positional encoding is added to the corresponding coordinate of the token embedding.

For the given example:

$$
[1,2,3] + [4,5,6] = [5,7,9]
$$

The result is still a 3-dimensional vector. Similarly, in a real Transformer, if \(d_{\text{model}} = 512\), the token embedding and positional encoding are both 512-dimensional, and their sum is also 512-dimensional.

## Why Addition Does Not Destroy Token Meaning

At first, it may seem strange to add a position vector directly to a token vector. If the token embedding represents word meaning, does adding another vector damage that meaning?

The answer is that neural network embeddings are **distributed representations**. A token embedding is not usually interpreted as a list of independent human-readable features. For example, dimension 1 does not simply mean "plural", dimension 2 does not simply mean "past tense", and so on. Instead, information is spread across many dimensions.

Therefore, adding a positional vector does not simply overwrite the token meaning. It shifts the token vector in the representation space so that the same token has slightly different representations at different positions.

For example:

| Token | Position | Representation idea |
|---|---:|---|
| `cat` | 1 | meaning of `cat` plus position 1 |
| `cat` | 5 | meaning of `cat` plus position 5 |
| `cat` | 10 | meaning of `cat` plus position 10 |

The semantic identity of the token is still present, but the model can now distinguish where that token appears in the sequence.

## Linear Projection View

Another important reason addition works is that the next operations in a Transformer are learned linear projections.

The Transformer computes query, key, and value vectors as:

$$
q_i = x_i W_Q
$$

$$
k_i = x_i W_K
$$

$$
v_i = x_i W_V
$$

Since \(x_i = e_i + p_i\), we have:

$$
q_i = (e_i + p_i)W_Q = e_iW_Q + p_iW_Q
$$

Similarly:

$$
k_i = (e_i + p_i)W_K = e_iW_K + p_iW_K
$$

$$
v_i = (e_i + p_i)W_V = e_iW_V + p_iW_V
$$

This shows that after addition, the model can still learn different effects from token content and position. The token embedding contributes through \(e_iW\), and the positional encoding contributes through \(p_iW\). The learned matrices \(W_Q\), \(W_K\), and \(W_V\) decide how much of each signal should be used.

So, adding the two vectors does not mean the model loses the ability to separate token information from position information. The learned projections can use both.

## Geometric Interpretation

We can also understand positional encoding geometrically.

A token embedding is a point in a high-dimensional space. Adding a positional encoding moves that point by a position-dependent offset.

For example:

$$
x_i = e_{\text{word}} + p_i
$$

If the same word appears in two different positions, the token embedding part is the same, but the positional part is different:

$$
x_1 = e_{\text{cat}} + p_1
$$

$$
x_5 = e_{\text{cat}} + p_5
$$

Thus, the model can tell that both inputs contain the word `cat`, while also knowing that one occurrence is at position 1 and the other is at position 5.

This is the central purpose of positional encoding: it makes identical tokens position-sensitive.

## Why Sinusoidal Positional Encoding Is Useful

In the original Transformer paper, the authors used sinusoidal positional encoding. For position \(pos\) and dimension index \(k\), the encoding is defined as:

$$
PE(pos, 2k) = \sin\left(\frac{pos}{10000^{2k/d_{\text{model}}}}\right)
$$

$$
PE(pos, 2k+1) = \cos\left(\frac{pos}{10000^{2k/d_{\text{model}}}}\right)
$$

This design has several useful properties.

## Multiple Frequencies Capture Different Scales

Different dimensions use sine and cosine waves with different wavelengths. Some dimensions change quickly with position, while others change slowly.

This means the positional encoding can represent both:

| Position property | Captured by |
|---|---|
| nearby local order | high-frequency dimensions |
| long-range position patterns | low-frequency dimensions |

As a result, the model can learn patterns involving both short-distance and long-distance relationships between tokens.

## Relative Position Can Be Inferred

Sinusoidal encoding also helps the model reason about relative distances. This is because sine and cosine obey useful addition identities:

$$
\sin(a+b) = \sin(a)\cos(b) + \cos(a)\sin(b)
$$

$$
\cos(a+b) = \cos(a)\cos(b) - \sin(a)\sin(b)
$$

This means that the encoding of position \(pos + \Delta\) can be expressed using the encoding of position \(pos\) and the offset \(\Delta\). Therefore, the model can more easily learn relationships such as:

```text
the next token
the previous token
a token 3 positions away
```

This is one reason sinusoidal positional encoding is a natural choice for sequence models.

## Deterministic and Length-Generalizable

Sinusoidal positional encoding is not learned from the training data. It is computed directly from the position number. Therefore, it can be generated for positions longer than those seen during training.

This does not guarantee perfect performance on longer sequences, but it gives the model a systematic positional pattern instead of a fixed learned table limited to the training length.

## Why Use Addition Instead of Concatenation

Another possible design would be concatenation:

$$
x_i = [e_i ; p_i]
$$

However, addition has several advantages.

| Method | Resulting dimension | Main issue |
|---|---:|---|
| Addition | \(d_{\text{model}}\) | simple and parameter-efficient |
| Concatenation | \(2d_{\text{model}}\) | doubles input size or requires another projection |

If token embeddings and positional encodings were concatenated, the model would need larger projection matrices or an extra layer to reduce the dimension back to \(d_{\text{model}}\). This would increase the number of parameters and computation.

Addition keeps the architecture simple:

$$
e_i + p_i \in \mathbb{R}^{d_{\text{model}}}
$$

The rest of the Transformer can remain unchanged. This is efficient and works well because the following learned layers can still extract and combine the token and position signals.

## Important Clarification

Adding positional encoding to token embedding does not mean that token meaning and position are the same kind of feature in a human sense.

Instead, it means both are encoded as vectors in the same numerical representation space used by the model. The Transformer does not read the vector coordinates as human-interpretable labels. It learns how to use the combined vector through training.

So the addition is best understood as:

```text
input representation = content signal + position signal
```

not as:

```text
word meaning is literally mixed with position in a human-readable way
```

This distinction is important because embeddings are machine-learned numerical representations, not symbolic feature lists.

## Step-by-step Example

Suppose the model uses a 3-dimensional representation:

```text
Positional embedding = [1, 2, 3]
Token embedding      = [4, 5, 6]
```

Element-wise addition gives:

```text
Input vector         = [1+4, 2+5, 3+6]
                     = [5, 7, 9]
```

The vector `[5, 7, 9]` is then passed into the Transformer. It is no longer only a token embedding and no longer only a positional embedding. It is a combined representation containing both token identity and position information.

In a real Transformer, the same operation happens with a much larger dimension, such as 512, 768, or 1024.

## Summary

The reason positional encoding can be added to token embedding is that both are vectors with the same dimension in the Transformer representation space. Token embedding represents the content of the token, while positional encoding represents the location of the token. Their element-wise sum produces a single input vector that contains both types of information.

This addition is meaningful because embeddings are distributed numerical representations, not independent human-readable feature lists. The model's learned linear projections can still use token and position information separately or jointly. In particular:

$$
(e_i + p_i)W = e_iW + p_iW
$$

so the token signal and positional signal can both influence the attention computation.

Sinusoidal positional encoding is especially useful because it provides smooth, deterministic, multi-frequency position patterns. These patterns help the Transformer recognize token order, distinguish identical tokens in different positions, and learn relative positional relationships.

## Conclusion

Element-wise addition of positional encoding and token embedding is a simple but powerful design choice in Transformers. It works because both vectors have the same dimensionality and are interpreted by the model as continuous representations. The resulting vector combines what the token is with where the token appears.

This gives self-attention the order information it lacks by itself, while keeping the model dimension unchanged and avoiding extra parameters. Therefore, positional encoding can be added directly to token embeddings without breaking the representation; instead, it makes the representation more informative for sequence modeling.
