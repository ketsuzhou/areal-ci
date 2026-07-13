# Reinforcement Learning Notes: Value, GAE & GRPO

## 1. The Definition of State Value

The value of a state $S_t$ under policy $\pi$ is defined as the expected return:

$$
V_\theta(S_t) = \mathbb{E}_{\tau \sim \pi} \left[ G_t \mid S_t \right] = \frac{1}{S} \sum_{s=0}^{S} G_t
$$

---

## 2. TD Error and Advantage

The Temporal Difference (TD) error $\delta_t$ and the Advantage $A_t$ for the terminal steps:

$$
\begin{aligned}
A_{T-1} &= -V_\theta(S_{T-1}) + R_{T-1} + \text{sg}[V_\theta(S_T)] \\
        &= \delta_{T-1} \quad (\text{since } V_\theta(S_T) = R_T \text{ or } 0) \\[6pt]
A_{T-2} &= -V_\theta(S_{T-2}) + R_{T-2} + \text{sg}[V_\theta(S_{T-1})] \\
        &= \delta_{T-2}
\end{aligned}
$$

> **Note:** $\text{sg}[\cdot]$ denotes the **stop-gradient** operator.

---

## 3. MSE Loss for Value Function

The Mean Squared Error loss for step $t$:

$$
\mathcal{L}_{\text{MSE}}^{(t)} = \frac{1}{2} \big( \delta_t \big)^2 = \frac{1}{2} \big( R_t + \gamma V_\theta(S_{t+1}) - V_\theta(S_t) \big)^2
$$

The total loss over the trajectory:

$$
\mathcal{L}_{\text{MSE}} = \sum_{t=0}^{T-1} \mathcal{L}_{\text{MSE}}^{(t)}
$$

---

## 4. Policy Gradient Loss

The standard Policy Gradient loss (e.g., PPO/REINFORCE):

$$
\mathcal{L}_{\text{PG}} = - \mathbb{E}_{s,a} \left[ \log \pi_\phi(a|s) \cdot \text{sg}[A_t] \right]
$$

---

## 5. GAE as λ-Return for Advantage Estimation
Generalized Advantage Estimation (GAE) is the λ-return method applied to estimating the advantage function. Following the n-step return idea used in the λ-return formulation, we can list N advantage estimators of increasing horizon:

$$
\begin{aligned}
A_t^{(1)} &= -V_\theta(S_t) + R_t + V_\theta(S_{t+1}) &&= \delta_t \\[4pt]
A_t^{(2)} &= -V_\theta(S_t) + R_t + R_{t+1} + V_\theta(S_{t+2}) &&= \delta_t +  \delta_{t+1} \\[4pt]
A_t^{(n)} &= -V_\theta(S_t) + \sum_{k=0}^{n-1} R_{t+k} + V_\theta(S_{t+n}) &&= \sum_{k=0}^{n-1} \delta_{t+k} \\[4pt]
A_t^{(N)} &= -V_\theta(S_t) + \sum_{k=0}^{N-1} R_{t+k} &&= \sum_{k=0}^{N-1}  \delta_{t+k} \quad (\text{Monte Carlo})
\end{aligned}
$$

### Key Insights:

1.  **$A_t^{(1)}$**: The 1-step TD advantage (pure TD error $\delta_t$). High bias, low variance.
2.  **$A_t^{(n)}$**: The n-step advantage. As $n$ grows, we trade **critic bias** for **return variance**.
3.  **$A_t^{(N)}$**: Pure Monte-Carlo advantage. No bootstrap value, **zero critic bias**, but highest variance and no credit assignment within the episode.

### GAE Interpolation

GAE's $A_t^{\text{GAE}( \lambda)}$ is the exponentially-weighted average of these n-step estimators:

$$
A_t^{\text{GAE}} = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} A_t^{(n)}
$$

This collapses to the backward recursion form:

$$
A_t^{\text{GAE}} = \sum_{k=0}^{\infty} ( \lambda)^k \delta_{t+k}
$$

> **Continuum:** GAE sits on a continuum between **Pure TD** ($\lambda=0 \Rightarrow A_t^{(1)}$) and **Pure MC** ($\lambda=1 \Rightarrow A_t^{(N)}$), interpolated by $\lambda$.

## 5. GRPO as a Special Case of GAE

GRPO (Group Relative Policy Optimization) can be viewed through the lens of n-step returns:

$$
\begin{aligned}
A_t^{(N)} &= -V_\theta(S_t) + \sum_{k=0}^{N-1} R_{t+k}\\[4pt]
A_t^{(N)} &= - \frac{1}{S} \sum_{s=0}^{S} G_t + \sum_{k=0}^{N-1} R_{t+k}\\[4pt] 
A_t^{(N)} &= - \frac{1}{S} \sum_{s=0}^{S} R_T^s + R_{T}\\[4pt] 
\end{aligned}
$$



## VIMPO: Value-Implicit Policy Optimization for LLMs

The central identity of this paper:

$$
\begin{aligned}
\beta \ln \frac{\pi^*(a_t \mid s_t)}{\pi_{\text{ref}}(a_t \mid s_t)} = r(s_t, a_t) + \gamma V^*(s_{t+1}) - V^*(s_t) + \beta \text{KL}^*(s_t)
\end{aligned}
$$
So the TD loss for value become:
$$
\begin{aligned}
\mathcal{L}_V(\pi) = \frac{1}{2} \left[ \sum_{k=0}^{T-1} \left( \beta \ln \frac{\pi(a_k \mid s_k)}{\pi_{\text{ref}}(a_k \mid s_k)} - \beta \operatorname{sg}[\operatorname{KL}_\pi(s_k)] \right) - (R_{\text{final}} - \overline{R}_{\text{final}}) \right]^2
\end{aligned}
$$
the policy gradient for acotor become:
$$
\mathcal{L}_{\text{PG}} = - \mathbb{E}_{s,a} \left[ \log \pi_\phi(a|s) \cdot \text{sg}[A_t] \right]
$$

## The Critic model still has its usege.

Information Asymmetry: The Critic with "Privileged Information": 
In traditional symmetric architectures, the Actor and Critic only see the same observations (e.g., a robot relying solely on camera images). In an asymmetric architecture, however, the Critic can access privileged information during training, while the Actor is restricted to the limited sensor data it would actually receive during deployment. 

- Actor model of an agent can view only its message information. Critic model can view message information of all agent.
- Critic model can view the future or memory information: Post-hoc privileged information.

Computational and Representation Asymmetry:

- Beyond information volume, the Critic can indeed trade more computation for overall performance gains during training. 