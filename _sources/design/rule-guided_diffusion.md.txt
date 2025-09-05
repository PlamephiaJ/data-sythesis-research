# Rule-guided Diffusion

The diffusion model is one of the earliest generative models used for image generation, and in recent years it has also been applied to text generation tasks. Compared with traditional Transformer-based text generation models, diffusion models perform remarkably well in terms of diversity and quality. However, approaches that rely purely on data-driven generation may overlook domain-specific rules and constraints, leading to results that do not meet expectations. In phishing email generation, following certain rules (such as including specific phishing keywords or mimicking the format of real emails) is crucial for producing high-quality phishing emails. To address this issue, we aim to explore a rule-guided diffusion model approach that integrates rules into the generation process, thereby improving both the quality and diversity of phishing email outputs.

## Methodology

### The process of diffusion models
- Forward process (adding noise):
    $$q(x_t|x_{t-1})=\mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_tI)$$
    In the text scenario, this can be: token embedding → add noise → obtain $x_t$.
- Reverse process (denoising):
    $$p_\theta(x_{t-1}|x_t,cond)=\mathcal{N}(x_{t-1};\mu_\theta(x_t,t,cond),\Sigma_\theta(x_t,t,cond))$$
    $$\mu_\theta(x_t,t,cond)=\frac{1}{\sqrt{1-\beta_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t,cond)\right)$$
    The model learns $\epsilon_\theta(x_t,t,\mathrm{cond})$ and gradually recovers $x_0$.
- Training objective:
    $$L_{simple}=\mathbb{E}_{t,x_0,\epsilon}[\|\epsilon-\epsilon_\theta(x_t,t,\mathrm{cond})\|^2]$$

### Incorporate rules into diffusion
Rules can be integrated into the diffusion process during the noise prediction / sampling constraint stages:
1. Classifier / Guidance style (Inference-time rule guidance, plug-and-play)
    - Treat the rule $R$ as a "constraint classifier":
        - At each denoising step, compute whether the generated sample satisfies the rule.
        - If it deviates from the rule, modify the noise gradient (similar to classifier guidance).
        - Formally:
        $$\nabla_{x_t}\log p(x_t|R)\approx\nabla_{x_t}\log p_\theta(x_t)+w\cdot\nabla_{x_t}\log p(R|x_t)$$

2. Rules encoded as condition vectors (In-training rule guidance)
    - Formalize the rules (regex/logical predicates) → encode them as embeddings.
    - Incorporate them into the noise prediction network (similar to prompts):
    $$\epsilon_\theta(x_t,t,rule\_embedding)$$
    - Effect: The model learns to follow these "rule signals" during denoising.

3. Rule-driven noise masking
    - Rules specify which tokens can be diffused and which must be preserved.
    - For example: If the rule states that verbs must be preserved → set the noise strength of the verb token embeddings to 0 and only add noise to other parts.
    - This is equivalent to the rules controlling the noise schedule.

### Using phishing email detection rules as guidance
#### Transforming rules into a “differentiable/approximately differentiable” scorer
Phishing rules are often discrete (regex/keywords/structure). To use gradients in diffusion, we need to create differentiable approximations or surrogates:
- Structural (subject/greeting/Call To Action/sign-off)
Surrogate: Sequence labeler (BiLSTM/Transformer) scores the completeness of placeholder templates:
r_struct(x) ∈ [0,1]
- Tone/urgency/pressure tactics
Surrogate: Sentiment/tone classifier (e.g., a specially trained "urgency" binary classifier) scores sentence-level embeddings:
r_urgency(x)
- Domain/link inconsistency (only generate placeholders)
Surrogate: Template consistency checker (a differentiable surrogate is a small discriminator that takes the embedding difference vector of [LINK_TEXT] and [URL_HOST] as input and outputs the "inconsistency" probability):
r_mismatch(x)
- Brand impersonation (placeholder level)
Surrogate: Named entity consistency/similarity constraint, aligning [ORG] with the organization mentioned in the text:
r_brand(x)

Rule set:
$$\mathcal{R}=\{r_k(x)\}_{k=1}^K,\text{each}\quad r_k:\mathrm{text}\to[0,1]$$

#### Equations for rule-guided diffusion
In the latent space, diffusion is performed with the current state denoted as $x_t$. The standard model predicts noise $\epsilon_\theta(x_t,t,c)$, where $c$ is the condition (e.g., task/topic).
Incorporating rule-guided gradient adjustment (similar to classifier-free guidance):
$$\tilde{\epsilon}(x_t)=\epsilon_\theta(x_t,t,c)-\lambda\cdot\nabla_{x_t}\mathcal{E}(x_t),\quad\mathcal{E}(x_t)=-\sum_kw_k\log r_k(x_t)$$
- $\mathcal{E}$ is the "energy": the more the rules are satisfied, the lower $\mathcal{E}$ becomes.
- $w_k$ is the rule weight; $\lambda$ is the guidance strength.
- Intuition: Denoise in the direction of increasing rule scores.

#### Approaches to incorporate rules into diffusion
1. Gradient guidance of rule scores
- At each sampling step, use a discriminator/scorer to obtain $r_k(x_t)$, and backpropagate to get $\nabla_{x_t}\log r_k$, which can be injected into the above equation.
- Advantages: Simple to implement and compatible with any rule surrogate.
- PS: Apply temperature/smoothing to each scorer to avoid gradient explosion.
2. Masked diffusion (rules determine "which positions to diffuse")
- First, use rule templates to produce a skeleton (pure placeholders):
`[SUBJ] :: [ORG] [NOTICE]`
`Dear [RECIPIENT], ... [CTA] ... [URL] ... [SIGNOFF]`
- Create a noise mask $M$: slots that must satisfy the rules are frozen or have low noise, while unimportant slots have high noise to increase diversity.
- During sampling:
`x_t = M ⊙ x_t + (1-M) ⊙ (x_t - η * guidance_grad)`
3. Projection/Constraint Sampling
- After each denoising step, perform a projection: project the output back to the "template-valid" set (e.g., complete placeholders, valid order).
- Very stable for strict rules (e.g., length, mandatory fields); for tone-related rules, apply additional guidance.

## References
- [Diffusion Guided Language Modeling](https://arxiv.org/abs/2408.04220v1) : ACL 2024
- [Symbolic Music Generation with Non-Differentiable Rule Guided Diffusion](https://arxiv.org/abs/2402.14285) : ICML 2024