# L14 Assignment

**Name:** LI HAN  
**Student ID:** 33C26029  
**GitHub Repository:** https://github.com/HannisLee/assignment_big_data

## Problem Statement

Suppose a mixture-of-experts (MoE) large language model is aligned with reinforcement learning from human feedback (RLHF) or reinforcement learning with verifiable rewards (RLVR).

The assignment asks two questions:

1. When PPO or GRPO collects responses, why must it record not only the generated tokens but also the experts selected by the router?
2. Is the same routing record needed when DPO is used for RLHF?

## Short Answer

For **PPO and GRPO, yes**. The rollout should record the expert IDs selected for every generated token at every MoE layer. During log-probability computation and policy training, these IDs are replayed so that the same token follows the same sparse expert path.

Without routing replay, the rollout engine and training engine may select different experts for the same token. The resulting log-probability difference would then contain both a policy change and an unrelated computation-path change. This can corrupt the importance ratio, KL estimate, clipping decision, and gradient.

For **standard offline DPO, no**. DPO trains directly on fixed preferred and rejected responses. It recomputes their log probabilities with the current policy and the reference policy and does not use a PPO-style rollout-to-old-policy importance ratio. Therefore, it does not need the expert choices that were used when the preference responses were originally generated.

## MoE Routing Background

A dense Transformer applies the same feed-forward network to every token. An MoE Transformer instead has a router and several expert feed-forward networks.

For a hidden state \(h_t\), the router computes scores for the available experts and selects a small top-\(k\) subset:

$$
E_t = \operatorname{TopK}(\operatorname{Router}(h_t))
$$

The MoE output can be written conceptually as:

$$
\operatorname{MoE}(h_t)
=
\sum_{e \in E_t} g_{t,e} f_e(h_t)
$$

where:

| Symbol | Meaning |
|---|---|
| \(E_t\) | experts selected for token \(t\) |
| \(g_{t,e}\) | routing weight for selected expert \(e\) |
| \(f_e\) | computation performed by expert \(e\) |
| \(h_t\) | token hidden state entering the MoE layer |

Different experts have different parameters. Consequently, changing \(E_t\) changes the hidden representation and can change the probability of the next generated token.

The visible response tokens therefore describe the external trajectory, but they do not fully describe the sparse internal computation path that produced the trajectory.

## PPO and GRPO Use Rollout Trajectories

[PPO](https://arxiv.org/abs/1707.06347) alternates between sampling trajectories from an old policy and optimizing a clipped surrogate objective. In LLM alignment, a trajectory contains a prompt and a sampled response:

$$
\tau = (x, y_1, y_2, \ldots, y_T)
$$

For each generated token, PPO compares its probability under the updated policy with its probability under the rollout policy:

$$
r_t(\theta)
=
\frac{
\pi_\theta(y_t \mid x,y_{<t})
}{
\pi_{\theta_{\mathrm{old}}}(y_t \mid x,y_{<t})
}
$$

Equivalently:

$$
r_t(\theta)
=
\exp\left(
\log\pi_\theta(y_t \mid x,y_{<t})
-
\log\pi_{\theta_{\mathrm{old}}}(y_t \mid x,y_{<t})
\right)
$$

The ratio is then clipped so that one batch of trajectories cannot move the policy too far in a single update.

[GRPO](https://arxiv.org/abs/2402.03300) changes how the advantage is estimated. It compares rewards among a group of responses and avoids the separate value model used by PPO. However, GRPO still uses a PPO-style policy ratio and clipping operation. Therefore, PPO and GRPO have the same requirement that rollout and training log probabilities be comparable.

## The Routing-Mismatch Problem

In a large RLHF or RLVR system, rollout generation and policy training are often separate stages:

```text
rollout engine
    -> generated tokens and rewards
    -> log-probability computation
    -> PPO/GRPO policy update
```

The stages may use different inference and training engines, kernels, parallel layouts, numeric precision, batching, or router implementations. The policy parameters may also change between rollout collection and later training epochs.

Small numerical or implementation differences near a top-\(k\) boundary can select different experts for the same token:

```text
Rollout:  token t -> experts [2, 7]
Training: token t -> experts [2, 5]
```

Both routes may be valid top-\(k\) results in their respective execution environments, but they execute different subnetworks. The training log probability is then not computed through the same sparse path as the rollout.

This introduces a mismatch that is unrelated to the intended policy update:

- the importance ratio may become artificially large or small;
- PPO/GRPO clipping may activate for the wrong reason;
- KL estimates may include routing-engine differences;
- different experts may receive gradients from those that produced the response;
- repeated minibatch updates may become unstable.

The [NVIDIA NeMo RL router-replay documentation](https://docs.nvidia.com/nemo/rl/nightly/guides/router-replay.html) describes this issue directly: recording and replaying routed expert indices keeps expert assignments consistent across rollout, log-probability, and training stages.

## Routing Replay

Let \(E_t\) denote the recorded routing path for token \(t\), including the selected experts at all MoE layers. With routing replay, a conceptual form of the ratio is:

$$
r_t(\theta)
=
\exp\left(
\log\pi_\theta(y_t \mid x,y_{<t},E_t)
-
\log\pi_{\theta_{\mathrm{old}}}(y_t \mid x,y_{<t},E_t)
\right)
$$

The notation emphasizes that both log probabilities use the same replayed sparse path \(E_t\). It does **not** require treating the router choice as a separate external action in the RL environment. The generated token remains the action being optimized; routing replay controls the internal computation used to evaluate that action consistently.

For a top-\(k\) MoE model, the minimum useful routing record is:

| Dimension | Recorded information |
|---|---|
| Response position | token index \(t\) |
| MoE depth | layer index \(l\) |
| Sparse route | ordered top-\(k\) expert IDs |

Conceptually, the stored data has the form:

$$
\text{route}[t,l,:]
=
[e_1,e_2,\ldots,e_k]
$$

The system does not normally need to store complete hidden states or full expert outputs. The selected expert IDs are sufficient for the training engine to force the same sparse route and recompute the necessary activations and log probabilities.

## Why Tokens Alone Are Insufficient

Consider two trajectories with exactly the same visible token sequence:

```text
Response tokens: [y1, y2, y3, y4]
```

One execution may route the tokens through expert paths \(E^{(a)}\), while another execution uses \(E^{(b)}\). Because the expert parameters differ:

$$
\pi(y_t \mid y_{<t}, E^{(a)}_t)
\neq
\pi(y_t \mid y_{<t}, E^{(b)}_t)
$$

Therefore, replaying only the tokens does not guarantee replaying the computation that assigned their probabilities. Recording the expert IDs resolves this ambiguity.

## Is Routing Replay Needed for DPO?

For standard offline DPO, the answer is **no**.

[Direct Preference Optimization](https://arxiv.org/abs/2305.18290) uses a dataset of fixed preference pairs:

$$
(x,y_w,y_l)
$$

where \(y_w\) is the preferred response and \(y_l\) is the rejected response. Its loss is:

$$
\mathcal{L}_{\mathrm{DPO}}
=
-\log\sigma\left(
\beta
\left[
\log\frac{\pi_\theta(y_w\mid x)}
{\pi_{\mathrm{ref}}(y_w\mid x)}
-
\log\frac{\pi_\theta(y_l\mid x)}
{\pi_{\mathrm{ref}}(y_l\mid x)}
\right]
\right)
$$

DPO performs forward passes over the two fixed responses using:

1. the current trainable policy \(\pi_\theta\);
2. the frozen reference policy \(\pi_{\mathrm{ref}}\).

It does not need to reconstruct the internal computation used by the model that originally generated \(y_w\) or \(y_l\). There is no rollout-old-policy importance ratio whose two sides must reproduce one sampled MoE path.

The current policy and reference policy may route the same preference token to different experts. That is legitimate because they are two different models whose sequence log probabilities are intentionally being compared.

## Important DPO Clarification

Saying that DPO does not need recorded generation-time routes does **not** mean that DPO bypasses the MoE router.

During every DPO forward pass:

- the current MoE policy still routes each token through its selected experts;
- the reference MoE policy also performs its own routing;
- gradients update the trainable model according to its current computation graph.

The difference is that these routes are computed as part of the current DPO forward passes. They do not need to match the routes used when the preference dataset was collected.

Expert IDs might still be stored in a nonstandard DPO system for exact auditing, debugging, a custom router-aware objective, or another special reproducibility requirement. These are optional extensions, not a requirement of the standard DPO loss.

## Comparison

| Method | Training data | Uses rollout policy ratio? | Needs generation-time routing replay? | Reason |
|---|---|---:|---:|---|
| PPO | On-policy sampled trajectories | Yes | Yes for MoE rollout/training consistency | Ratio and clipping require comparable rollout and training log probabilities |
| GRPO | Groups of sampled trajectories | Yes | Yes for MoE rollout/training consistency | Group-relative advantages change the baseline, not the routing-consistency requirement |
| DPO | Fixed preferred/rejected pairs | No | No in standard offline DPO | Log probabilities are recomputed directly under the current and reference policies |

## Final Answers

### 1. Why record the selected experts for PPO/GRPO?

An MoE token probability depends on the experts through which the token is processed. PPO and GRPO reuse sampled trajectories to compute policy ratios, KL terms, clipping decisions, and gradients. If the training pass selects different experts from the rollout pass, these quantities contain an artificial rollout-training mismatch.

Recording the expert IDs per token and per MoE layer allows routing replay, keeps the sparse computation path consistent, and makes the policy update correspond to the trajectory that was actually collected.

### 2. Is it also needed for DPO?

No, not for standard offline DPO. DPO trains on fixed preference pairs and directly evaluates them with the current and reference policies. It does not reconstruct an on-policy rollout or form a PPO/GRPO importance ratio against the rollout policy, so the experts used when the preference responses were generated do not need to be recorded.

## Conclusion

For a dense model, response tokens are usually sufficient to identify a rollout for later policy optimization. For an MoE model, the selected experts are an additional part of the sparse computation that produced the token probabilities.

PPO and GRPO therefore record and replay expert IDs to prevent routing differences from corrupting importance ratios, KL estimates, clipping, and gradients. GRPO's group-relative advantage does not remove this requirement because its policy update remains PPO-like.

Standard offline DPO is different. It compares current-policy and reference-policy log probabilities on fixed preferred and rejected sequences and does not replay the generation policy. It still uses MoE routing during its forward passes, but it does not require the routes from the original response-generation stage.

## References

1. John Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), 2017.
2. Zhihong Shao et al., [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300), 2024.
3. Rafael Rafailov et al., [Direct Preference Optimization: Your Language Model Is Secretly a Reward Model](https://arxiv.org/abs/2305.18290), 2023.
4. NVIDIA NeMo RL, [Router Replay](https://docs.nvidia.com/nemo/rl/nightly/guides/router-replay.html).
