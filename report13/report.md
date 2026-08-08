# L13 Assignment

**Name:** LI HAN  
**Student ID:** 33C26029  
**GitHub Repository:** https://github.com/HannisLee/assignment_big_data

## Problem Statement

In this assignment, an LLM performs entity matching on (N) candidate product pairs produced by a blocking stage. For every pair, the model must decide whether the two records refer to the same real-world product.

The LLM is served with SGLang, so the prompt should be shortened to reduce repeated inference work while approximately preserving matching quality. All six original prompt components remain useful and must therefore be retained:

1. system message;
2. task description;
3. injected knowledge;
4. instance content;
5. question;
6. output format.

## Main Sources of Unnecessary Cost

The original prompt is clear, but it contains several expressions that an advanced instruction-following model does not need:

- The system message uses multiple sentences to say that the model should follow instructions.
- The task description repeats the decision request later stated in the question.
- Phrases such as "listed below," "based on the information provided," and "before making your decision" add tokens without adding decision criteria.
- The instance appears before the final question and output instruction. Because the instance changes for every pair, the common text after it cannot be part of one continuous shared prefix.
- The output instruction permits only two labels, but its wording can be shorter and can explicitly prohibit explanations.

These costs are repeated (N) times, so even a modest reduction per request can become important for a large candidate set.

## Optimized Prompt Components

The following version preserves the meaning of every component while using fewer words.

| Component | Optimized prompt |
|---|---|
| **system message** | `Follow instructions exactly.` |
| **task description** | `Determine whether Product A and Product B are the same real-world product. Compare all provided attributes.` |
| **injected knowledge** | `Ignore missing values (N/A or nan).` |
| **question** | `Are A and B the same product?` |
| **output format** | `Output one label only: Yes or No.` |
| **instance content** | `A: name="Sequoia American Amber Ale"; factory="Wig And Pen"`<br>`B: name="Aarhus Cains Triple A American Amber Ale"; factory="Aarhus Bryghus"` |

The recommended concatenation order is:

```text
system message
task description
injected knowledge
question
output format
instance content
```

Therefore, the complete optimized prompt for the given pair is:

```text
Follow instructions exactly.
Determine whether Product A and Product B are the same real-world product. Compare all provided attributes.
Ignore missing values (N/A or nan).
Are A and B the same product?
Output one label only: Yes or No.
A: name="Sequoia American Amber Ale"; factory="Wig And Pen"
B: name="Aarhus Cains Triple A American Amber Ale"; factory="Aarhus Bryghus"
```

In an actual chat API, the first line remains the system message and the remaining lines form the user message. The important property is that all fixed instructions occur before the variable pair.

## Why the New Order Helps SGLang

Across the (N) entity-matching requests, the instructions, missing-value rule, question, and output format are identical. Only Product A and Product B change.

Placing all static text first creates the following request structure:

```text
[shared static prefix][variable product pair]
```

SGLang provides prefix caching through RadixAttention. Requests that share the same initial token sequence can reuse cached KV states for that prefix. The official SGLang repository lists RadixAttention for prefix caching as one of its runtime optimizations: [SGLang documentation](https://github.com/sgl-project/sglang/blob/main/README.md).

If the varying instance remains in the middle of the prompt, the requests diverge earlier. Moving the instance to the end maximizes the continuous shared prefix and gives SGLang more reusable work across the (N) requests.

## Prompt Length Comparison

The comparison below counts whitespace-separated words and plain-text characters, with the components joined by newline characters.

| Version | Words | Characters |
|---|---:|---:|
| Original prompt | 109 | 655 |
| Optimized prompt | 57 | 373 |
| Reduction | 47.7% | 43.1% |

These are tokenizer-independent measurements. The exact token count and token reduction depend on the tokenizer of the served model, so a specific token reduction is not claimed here.

The shorter input reduces the amount of prompt processing required during the prefill stage. Prefix reuse can further reduce repeated processing after the first request with the same fixed prefix.

## Decode-Time Reduction

Entity matching is a binary classification task. A long explanation is not required for downstream processing, so the optimized format says:

```text
Output one label only: Yes or No.
```

This has two benefits:

1. The model generates only the classification label instead of reasoning paragraphs.
2. The output can be parsed directly without removing prefixes such as `Answer:` or extracting a label from an explanation.

For the example pair, the names differ substantially and the factories are also different. The correct model response is therefore exactly:

```text
No
```

The explanation above is useful in this report, but it should not be generated during each production inference request.

## Why Matching Quality Should Be Preserved

The optimized prompt retains every piece of information that affects the decision:

| Preserved information | Purpose |
|---|---|
| Same-product decision | Defines the entity-matching task |
| Compare all attributes | Prevents the model from relying only on the product name |
| Real-world product wording | Clarifies that formatting differences do not automatically imply different entities |
| Missing-value rule | Prevents `N/A` and `nan` from being treated as evidence |
| Complete A and B records | Preserves all instance evidence |
| Yes/No labels | Preserves the required output space |

Only generic politeness, repeated wording, and unnecessary prose are removed. No product attribute or decision rule is removed.

## Evaluation Plan

Before replacing the original prompt in a full workload, both versions should be compared on the same labeled validation set.

The validation set should include:

- matching records with minor spelling, abbreviation, punctuation, or capitalization differences;
- clearly different products;
- records with similar names but conflicting factories or other attributes;
- records containing `N/A` or `nan`;
- records in which only one useful attribute is available.

Quality should be compared using accuracy, precision, recall, and F1. Performance should be compared under the same model, hardware, batch size, concurrency, and sampling configuration using:

- input and output lengths;
- request throughput;
- time to first token (TTFT);
- end-to-end latency;
- cold-cache performance;
- warm-cache performance with the common prefix already cached.

The optimized prompt is acceptable if it produces only `Yes` or `No`, ignores missing values, maintains roughly the same entity-matching quality, and improves input/output length or serving performance.

## Additional Runtime Recommendations

Although no SGLang implementation code is required for this assignment, the serving configuration should use deterministic decoding for reproducible classification and a small maximum output length. If supported by the selected SGLang interface, generation can also be constrained to the alternatives `Yes` and `No`.

These settings complement the prompt optimization: the prompt reduces prefill work, while the short constrained response reduces decode work and prevents invalid output formats.

## Conclusion

The optimized prompt keeps all six original components but expresses them more compactly. It reduces the plain-text prompt from 109 to 57 words and places the changing product records after a shared static prefix.

This design speeds up inference in three ways: fewer input tokens need to be processed, SGLang can reuse a longer common prefix through RadixAttention, and the model generates only one classification label. At the same time, the essential task definition, complete attribute comparison, missing-value rule, instance data, decision question, and output choices are all preserved, so entity-matching quality should remain approximately unchanged.
