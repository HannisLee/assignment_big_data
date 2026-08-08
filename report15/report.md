# L15 Assignment

**Name:** LI HAN  
**Student ID:** 33C26029  
**GitHub Repository:** https://github.com/HannisLee/assignment_big_data

## Problem Statement

An LLM agent is given a Markdown file that describes the tools it may use. The file is loaded directly into the model's context when a task is processed. Intuitively, adding more tools should make the agent more capable. In practice, however, performance often drops when the file grows from a small tool list to hundreds of tool definitions.

This assignment explains why this happens and proposes a scalable solution that preserves access to the complete tool library without placing every full tool description in the LLM context at the same time.

## Short Answer

The performance drop is caused by **how the tools are presented**, not by the fact that the tools exist.

When hundreds of tool descriptions are loaded together, they consume a large part of the context window and introduce many irrelevant or similar candidates. The LLM must locate the relevant definition, distinguish it from distractors, remember its parameters, and then solve the original task. Long-context models do not retrieve every item from a long prompt with equal reliability, so a valid context length does not guarantee reliable tool selection. The result is more wrong-tool calls, mixed-up arguments, unnecessary calls, higher latency, and sometimes failure to call any tool.

The recommended solution is **retrieval-augmented tool loading**, also called **Tool RAG** or dynamic tool discovery. Store the complete tool catalog outside the active prompt, retrieve a small task-relevant subset, and load only those full definitions into the LLM context:

```text
User task
    -> search the external tool registry
    -> retrieve and rerank top-k tools
    -> load only their full schemas
    -> LLM selects and calls a tool
    -> use execution feedback to finish or search again
```

The LLM therefore retains logical access to hundreds or thousands of tools while normally seeing only about three to eight relevant candidates at one decision point.

## Why Loading More Tools Can Reduce Performance

### 1. Tool Definitions Consume the Context Budget

A tool definition is not free metadata. Its name, description, parameter documentation, examples, and input schema are serialized into tokens and placed in the prompt. If there are \(N\) tools and definition \(d_i\) has token length \(|d_i|\), the approximate tool-description burden is:

$$
B_{\text{full}} = \sum_{i=1}^{N} |d_i|
$$

Thus, the cost grows approximately linearly with the number of tools. A large tool file leaves fewer useful tokens for:

- the user's task and data;
- system instructions;
- conversation history and memory;
- intermediate observations from tools;
- the model's answer.

Even if the entire file fits inside the advertised context window, it increases input processing, latency, and cost. In a multi-step agent, the overhead may be repeated or remain present while more tool results accumulate.

[Anthropic's Advanced Tool Use article](https://www.anthropic.com/engineering/advanced-tool-use) provides a concrete example: 58 tools across five servers consumed approximately 55,000 tokens before the conversation began. It also reports cases in which tool definitions consumed 134,000 tokens before optimization. This demonstrates that tool metadata can become one of the largest parts of an agent prompt.

### 2. Irrelevant and Similar Tools Act as Distractors

Tool use requires more than remembering a definition. For a task \(q\), the model must choose an action from a candidate set:

$$
\hat{t} = \arg\max_{t \in \mathcal{T}} P(t \mid q, \mathcal{T})
$$

where \(\mathcal{T}\) is the set of visible tools. Increasing \(|\mathcal{T}|\) adds more alternatives to this decision. If most new tools are irrelevant, they do not provide useful capability for the current task; they provide additional opportunities for confusion.

The problem is especially serious when tools have overlapping names or descriptions. For example:

```text
send_user_notification
send_channel_notification
send_workspace_notification
schedule_user_notification
```

All four tools contain similar words and may use similar arguments. The model can select the wrong scope, combine the parameters of two definitions, or invent an argument that exists only in a neighboring schema. Long descriptions can also contain keywords that match the task even when the underlying tool is inappropriate.

Therefore, a larger visible tool set changes the task from selecting among a few well-separated actions into locating one useful action among many semantic distractors. The [RAG-MCP paper](https://arxiv.org/abs/2505.03275) describes this issue as prompt bloat and proposes semantic retrieval so that the LLM receives only relevant tool metadata.

### 3. Long Context Is Not Used Uniformly

Having enough context capacity is not the same as using all context items reliably. The [Lost in the Middle study](https://arxiv.org/abs/2307.03172) found that language models can be less successful when relevant information appears in the middle of a long input than when it appears near the beginning or end.

A Markdown catalog containing hundreds of definitions has the same general risk. The correct tool may be located far from the task instruction or surrounded by similar descriptions. The model's attention is distributed across much more text, and the relevant definition may have a weak influence on the final tool-selection tokens.

This explains why simply moving to a model with a larger context window is incomplete. A larger window prevents truncation, but it does not remove distractors, reduce selection ambiguity, or guarantee equal access to every tool definition.

## Why More Available Tools Are Not the Real Problem

The agent should still be allowed to use the full library. The failure comes from making every complete definition part of every decision.

The distinction is:

| Design | Tools logically available | Full definitions visible now | Expected effect |
|---|---:|---:|---|
| Static full loading | All tools | All tools | High prompt cost and many distractors |
| Hard-coded small list | Only selected tools | Selected tools | Efficient but loses capabilities |
| Dynamic retrieval | All tools | Only relevant top-\(k\) tools | Preserves capability with a small decision set |

Compressing all definitions may reduce tokens, but excessive compression can remove distinctions, parameter requirements, or usage conditions. It also leaves all tools as candidates. Retrieval addresses both prompt size and decision complexity.

## Proposed Solution: Retrieval-Augmented Tool Loading

The solution separates **tool discovery** from **tool invocation**. The full Markdown file becomes an external tool registry rather than a block that is always copied into the prompt.

### 1. Build an External Tool Registry

Each tool should be stored as a separate record containing:

| Field | Purpose |
|---|---|
| Tool ID and name | Stable identification and exact lookup |
| Retrieval description | Explains when and why to use the tool |
| Keywords or domain | Supports lexical and category matching |
| Inputs and outputs | Supports task and dependency matching |
| Full definition or schema | Loaded only after the tool is retrieved |
| Version and availability | Prevents stale or unavailable tools from being selected |

The complete registry can remain source-controlled as Markdown or structured files. An indexing process splits it at tool boundaries and creates a searchable record for every tool. Adding a new tool then requires indexing that record, not modifying or fine-tuning the LLM.

### 2. Use Hybrid Retrieval

For a user task or subgoal \(q\), the system searches tool names and descriptions. A practical retriever combines lexical and semantic evidence:

$$
S(t,q)
=
\alpha S_{\text{BM25}}(t,q)
+
(1-\alpha)S_{\text{vector}}(t,q)
$$

BM25 is useful for exact product names, operations, and parameter terms. Vector similarity is useful when the task and description express the same intent with different words. A reranker can then compare the strongest candidates more precisely and return only the top \(k\), typically three to eight tools.

This retrieval stage needs its own evaluation. The [ToolRet benchmark](https://arxiv.org/abs/2503.01763) shows that models that perform well on conventional information retrieval do not automatically perform well on tool retrieval, and that poor retrieval quality directly reduces the task pass rate of tool-using LLMs. Tool selection should therefore not rely on an arbitrary embedding model without testing its Recall@\(k\) on realistic queries.

### 3. Load Only the Retrieved Full Definitions

Initially, the model sees only:

- a small `search_tools` meta-tool;
- a brief statement of the available tool domains;
- optionally, three to five common and safety-critical tools.

When the agent searches for a capability, the system returns references to the best matching tools and expands only their complete definitions into the active context. The prompt burden becomes:

$$
B_{\text{dynamic}}
=
|d_{\text{search}}|
+
\sum_{t \in \operatorname{TopK}(q)} |d_t|
$$

Since \(k \ll N\), normally:

$$
B_{\text{dynamic}} \ll B_{\text{full}}
$$

The LLM then performs ordinary function calling over a small, relevant candidate set. This architecture is model-independent: it can be implemented by an application-side preprocessor, an agent framework, or a provider's deferred tool-loading feature.

Anthropic reports that its on-demand Tool Search design reduced tool-related token use by 85% in its example while retaining access to the complete library. In its internal large-tool MCP evaluations, the reported accuracy increased from 49% to 74% for Claude Opus 4 and from 79.5% to 88.1% for Opus 4.5. These results are vendor-specific rather than a universal guarantee, but they provide practical evidence for the dynamic-loading principle.

### 4. Retrieve Iteratively for Multi-Step Tasks

A single retrieval pass is not always sufficient. A multi-step task may reveal new subgoals only after a tool returns data. For example:

```text
Find a customer by email
    -> obtain the customer ID
    -> find that customer's unpaid invoices
    -> send a reminder
```

The first query needs a customer-search tool, while later steps need billing and messaging tools. Loading all three domains at the beginning is unnecessary. The agent should search again when the active subgoal changes.

Execution feedback should also influence discovery:

```text
successful result -> continue or finish
no suitable candidate -> rewrite the search query
low retrieval confidence -> increase k or search a domain first
invalid arguments -> reload schema and correct the call
tool unavailable or failed -> exclude it and retrieve an alternative
```

This produces an iterative loop rather than a rigid one-shot filter. However, it still avoids the emergency fallback of injecting the entire catalog.

### 5. Improve Tool Descriptions for Retrieval

Dynamic loading is only as reliable as the catalog. A retrieval description should state:

- the real operation performed by the tool;
- when it should and should not be used;
- important input and output concepts;
- how it differs from similarly named tools;
- domain synonyms that users are likely to mention.

For example:

```text
Weak:
query_orders - Runs an order query.

Better:
search_customer_orders - Searches a customer's orders by customer ID,
date range, status, or total amount. Returns order IDs, items, payment
status, and shipping status. Use this for existing orders, not for
creating a new order.
```

The better description improves both lexical and semantic retrieval while giving the final LLM enough information to distinguish the tool from order-creation or shipment-tracking tools.

## End-to-End Architecture

The proposed system can be summarized as follows:

```text
                         offline / update time
Full Markdown catalog ------------------------------+
    -> split by tool                                 |
    -> validate metadata                             |
    -> BM25 index + vector index                     |
                                                      v
User task -> search_tools -> hybrid retrieval -> reranking
                                              -> top 3-8 tool IDs
                                              -> load full schemas
                                              -> LLM chooses tool
                                              -> validate arguments
                                              -> execute
                                              -> result or error
                                              -> answer / search again
```

The external registry remains comprehensive, while the LLM's current decision boundary remains small.

## Static Loading and Dynamic Retrieval Comparison

| Property | Load every tool | Retrieve top-\(k\) tools |
|---|---|---|
| Access to complete catalog | Yes | Yes, through search |
| Definitions in every prompt | Hundreds | Normally 3-8 plus search tool |
| Input-token growth | Approximately linear in catalog size | Mainly depends on retrieved subset |
| Distractor exposure | High | Low |
| New-tool update | Reload full catalog | Add or update one index record |
| Multi-step adaptation | Same fixed tool list | Search again for each subgoal |
| Main failure mode | Context overload and wrong selection | Retriever may miss the correct tool |

Dynamic retrieval does introduce an extra search step and a new possible failure: the correct tool may not appear in the top \(k\). This is why retrieval quality, fallback behavior, and end-to-end evaluation are essential.

## Evaluation Plan

The solution should be evaluated against static full loading under the same model, system prompt, task set, decoding configuration, and hardware.

### Experimental Conditions

Test both designs with catalogs containing:

- 10 tools;
- 50 tools;
- 100 tools;
- 500 tools.

For each size, include random irrelevant tools and difficult distractors with similar names, descriptions, inputs, or outputs. The relevant tool should also be placed at different positions in the static Markdown file so that position effects can be measured.

### Metrics

| Metric | What it measures |
|---|---|
| Retrieval Recall@\(k\) | Whether the required tool appears in the retrieved set |
| Tool-selection accuracy | Whether the LLM selects the correct candidate |
| Argument accuracy | Whether required parameters and values are correct |
| End-to-end task success | Whether the final task is completed correctly |
| Invalid or unnecessary call rate | Frequency of hallucinated, wrong, or redundant calls |
| Input tokens | Context cost of task and tool definitions |
| Latency and monetary cost | Operational efficiency of the complete workflow |

The evaluation should cover single-tool tasks, multi-tool tasks, tasks for which no valid tool exists, recently added or changed tools, retrieval misses, unavailable tools, and execution failures that require an alternative.

The dynamic solution is successful if it maintains a much more stable task-success rate as the registry grows, substantially reduces the number of tool-definition tokens presented per decision, and recovers safely when the first retrieval or tool call fails.

## Practical Recommendations

The default production design should therefore be:

1. keep the complete catalog outside the active LLM prompt;
2. keep only a search meta-tool and a few essential tools permanently visible;
3. retrieve and rerank a small candidate set for the current subgoal;
4. expand full schemas only after retrieval;
5. validate generated tool names and arguments before execution;
6. re-query after state changes or failures;
7. continuously test retrieval recall and end-to-end task success as tools are added.

Access control must be applied before or during retrieval. A search result should never expose a tool that the current user or agent is not authorized to call. Dynamic retrieval reduces the visible attack surface, but it does not replace authentication, permission checks, schema validation, or execution sandboxing.

## Conclusion

Giving an LLM agent more tools increases its potential capability, but loading hundreds of complete descriptions into every prompt reduces its effective performance. The definitions consume context and computation, similar tools create distractor interference, and long-context models do not access every position with equal reliability. Consequently, the correct tool becomes harder to find and use even though it is technically present in the context.

The scalable solution is to separate availability from visibility. A complete external registry preserves all tools, while hybrid retrieval and reranking select a small task-relevant subset whose full definitions are loaded on demand. Iterative search allows the visible subset to change as a multi-step task evolves.

This Tool RAG architecture changes the agent's problem from "choose one tool among hundreds" to "search the library, then choose one tool among a few relevant candidates." It reduces prompt bloat and distractor confusion without permanently removing capabilities, making tool use more accurate, efficient, and scalable.

## References

1. Nelson F. Liu et al., [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172), 2023.
2. Zhengliang Shi et al., [Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models](https://arxiv.org/abs/2503.01763), 2025.
3. Tao Gan and Qifan Sun, [RAG-MCP: Mitigating Prompt Bloat in LLM Tool Selection via Retrieval-Augmented Generation](https://arxiv.org/abs/2505.03275), 2025.
4. Anthropic, [Introducing Advanced Tool Use on the Claude Developer Platform](https://www.anthropic.com/engineering/advanced-tool-use), 2025.
5. Kevin Lumer et al., [ScaleMCP: Dynamic and Auto-Synchronizing Model Context Protocol Tools for LLM Agents](https://arxiv.org/abs/2505.06416), 2025.
