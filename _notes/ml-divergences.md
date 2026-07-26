---
layout: post
title: "ML Divergences"
date: 2025-01-21
tags: machine-learning, probability
citation: true
toc:
  sidebar: left
---
In machine learning, divergence is a measurement of the difference between two distributions. It can be asymmetric so it is not required to be a distance. Kullback-Leibler (KL) divergence is the most popular form of divergence. We will explore the properties of KL divergence that make it so useful. For two probability distributions $$A$$ and $$B$$, it is defined as follows:

$$
D_{KL}(A||B) = \sum_xA(x)\log\frac{A(x)}{B(x)}
$$

We are only considering discrete distributions. For continuous distributions we can just swap out the summation for an integral. 

In ML, we typically use $$P(x)$$ to represent the true distribution from the real world, which remains fixed. $$Q(x)$$ represents our model's distribution that we aim to fit. KL is asymmetric, so the order in which we put these two functions is important.

## Forward KL

$$
D_{KL}(P||Q) = \sum_xP(x)\log\frac{P(x)}{Q(x)}
$$

- Zero-avoiding: $$Q$$ avoids taking zero value when $$P$$ is non-zero. $$Q$$ covers
- Mode-covering / mean seeking: If there are multiple modes in $$P$$, $$Q$$ will assign probability to all. $$Q$$ assigns some probability mass over the whole distribution of $$P$$.

![[source](https://blog.evjang.com/2016/08/variational-bayes.html) (Eric Jang blog)](/assets/img/notes/ml-divergences/forward-kl-mode-covering.png)

## Reverse KL

$$
D_{KL}(Q||P) = \sum_xQ(x)\log\frac{Q(x)}{P(x)}
$$

- Zero-seeking: $$Q$$ avoids taking non-zero values when $$P$$ is zero.
- Mode-seeking: $$Q$$ assigns most of the probability to a single mode.

![[source](https://blog.evjang.com/2016/08/variational-bayes.html) (Eric Jang blog)](/assets/img/notes/ml-divergences/reverse-kl-mode-seeking.png)

## Information Theory Interpretation

KL has an information theory interpretation as relative entropy. When comparing distributions $$A$$ and $$B$$, $$D_{KL}(A\Vert B)$$ represents the expected number of extra bits needed to encode samples from distribution $$A$$ using a code that was optimized for distribution $$B$$.

The value $$\log\frac{A(x)}{B(x)}$$ measures the amount of surprise, measured in nats. If the base is 2, we get bits. This is the surprise of observing an event $$x$$ from $$A$$ given the $$B$$. We weight by $$A(x)$$ to get an expectation of this surprise. 

When we use forward KL , $$D_{KL}(P\Vert Q)$$, we're measuring the expected number of extra bits needed to encode samples from the true distribution $$P$$ using a code optimized for our model distribution $$Q$$. We are weighing surprise by what it is likely in the true distribution. 

In the reverse case, $$D_{KL}(Q\Vert P)$$, we're measuring the extra bits needed to encode samples from our model $$Q$$ using a code optimized for the true distribution $$P$$. Now that we're weighting the surprise by $$Q(x)$$, we care most about events that our model thinks are likely.

This information-theoretic perspective illuminates why KL divergence is such a natural choice for training machine learning models. When we train a model, we're essentially trying to minimize the extra information (measured in bits or nats) needed to encode samples from the true data distribution using our model's distribution. The closer our model distribution gets to the true distribution, the fewer extra bits we need for encoding, and thus the lower the KL divergence.

KL can also be expressed in terms of entropy and cross entropy:

$$
D_{KL}(P || Q) = H(P, Q) - H(P)
$$

where $$H(P,Q)$$ is the cross-entropy between $$P$$ and $$Q$$, and $$H(P)$$ is the entropy of $$P$$. This formulation reveals that minimizing KL divergence is equivalent to minimizing cross-entropy when the true distribution $$P$$ is fixed. Many ML models are trained to minimize cross entropy.

# Properties

Here we will discuss the favorable mathematical properties of KL divergence. We will aim to convince ourselves that KL is unique in satisfying our desired properties.

We are getting these properties from [https://blog.alexalemi.com/kl.html](https://blog.alexalemi.com/kl.html)

## Continuous

Discontinuities don’t make sense in this context and also make it difficult to optimize as the divergence would become non differentiable. We require a divergence measure that is continuous for non negative inputs.

## Non-Negative

Negative divergence does not make sense in the same way a negative distance does not. We also want the property that the divergence is only 0 when both probabilities are equal.

We can prove that KL is always non negative using Jensens inequality: $$f(E[X]) \leq E[f(X)]$$ for convex functions $$f$$. Let $$f(x) = -\log(x)$$. This gives us:

$$
\begin{aligned}
    f\left(\mathbb{E}_{P(x)}\left[\frac{Q(x)}{P(x)}\right]\right) &\leq \mathbb{E}_{P(x)}\left[f\left(\frac{Q(x)}{P(x)}\right)\right] \\
    -\log\left(\mathbb{E}_{P(x)}\left[\frac{Q(x)}{P(x)}\right]\right) &\leq \mathbb{E}_{P(x)}\left[-\log\left(\frac{Q(x)}{P(x)}\right)\right] \\
    -\log\left(\sum_x P(x) \cdot \frac{Q(x)}{P(x)}\right) &\leq \sum_x P(x) \cdot \left(-\log\left(\frac{Q(x)}{P(x)}\right)\right) \\
    -\log\left(\sum_x Q(x)\right) &\leq -\sum_x P(x) \log \frac{Q(x)}{P(x)} \\
    -\log(1) &\leq -\sum_x P(x) \log \frac{Q(x)}{P(x)} \\
    0 &\leq \sum_x P(x) \log \frac{P(x)}{Q(x)} \\
    0 &\leq D_{KL}(P || Q)
    \end{aligned}
$$

The intuition behind this is that log ratio can take positive and negative values, but the weight ensures the more positive values dominate. The weighing term is necessary for this measure to be nonnegative.

## Monotonic

If we update $$Q(X)$$ in one direction, increasing or decreasing the probability, the divergence with respect to $$P(X)$$ should only move in one direction. Intuitively, the divergence should increase as the difference from $$P(X)$$ increases, and decrease as $$Q(X)$$ gets closer to $$P(X)$$.

Let’s set $$P(X)$$ to be uniform with 5 possible events: $$[0.2, 0.2, 0.2, 0.2, 0.2]$$,

$$Q_1(X)$$ puts all the weight on 3 events: $$[0.33,0.33,0.33,0,0]$$

$$Q_2(X)$$ puts all the weight on 2 events: $$[0.5,0.5,0,0,0]$$

The divergence with $$Q_2$$ should be greater than with $$Q_1$$ 

## Reparameterization Invariant

This means that the units of the input $$X$$ should not affect the value of the divergence. This is important since we want to measure the difference in probability. Probability has no units. Most divergence measures meet this property since they work with probability distributions $$P(X)$$ and $$Q(X)$$ and do not access $$X$$ itself.

## Decomposable

Let’s say we have two joint distributions $$P(X,Y)$$ and $$Q(X,Y)$$. Recall the chain rule of probability: $$P(X,Y) = P(X\vert Y)P(Y)$$. We want our divergence measure to decompose in this way:

$$
\begin{aligned}D(P(X,Y)||Q(X,Y)) &= D(P(X)||Q(X)) + E_{P(X)}[D(P(Y|X)||Q(Y|X))] \\  \end{aligned}
$$

We have the expectation in this equivalence because we need to weigh the conditional probabilities by the probability of the condition variable.

This equivalence has important implications. When we learn about variables sequentially, the total information gained must equal the information gained from learning about both variables simultaneously. The left side is the information gain of learning about both variables at once, whereas in the right side you learn about $$Y$$ first.

We can prove that KL divergence satisfies this property:

$$
\begin{aligned} D_{KL}(P(X, Y) || Q(X, Y)) &= ∑_x ∑_y P(x, y) \log( \frac{P(y|x) P(x)}{ Q(y|x) Q(x)} )\\ 
&= ∑_x ∑_y P(x, y) [\log(\frac{P(x)}{Q(x)}) + \log(\frac{P(y|x)}{Q(y|x)})] \\ 
&= ∑_x ∑_y P(x, y) \log(\frac{P(x)}{Q(x)}) + ∑_x ∑_y P(x, y) \log(\frac{P(y|x)}{Q(y|x)}) \\ 
&= ∑_y P(y|x) ∑_x  P(x) \log(\frac{P(x)}{Q(x)}) + ∑_x P(x) ∑_y P(y|x) \log(\frac{P(y|x)}{Q(y|x)}) \\ 
&= ∑_x  P(x) \log(\frac{P(x)}{Q(x)}) +  ∑_x  P(x) ∑_y P(y|x) \log(\frac{P(y|x)}{Q(y|x)}) \\ 
&= D_{KL}(P(X) || Q(X)) + E_{P(X)}[D_{KL}(P(Y|X=x)||Q(Y|X=x))] 
\end{aligned}
$$

Many divergence measures share properties like monotonicity, continuity, and reparameterization invariance. However, KL divergence stands out because it is linearly decomposable. This unique property requires both the logarithmic form for decomposability and the specific KL form for non-negativity.

# Alternatives to KL Divergence

We will go through some alternative divergence measures and why they are insufficient. We want to show that they all break one of the desirable properties of the KL divergence.

## $$L_p$$ Norm

The simplest divergence measure is to just take the difference between the probability values.

$$
D_{norm}(P,Q) = \sum_x||P(x)-Q(x)||_p
$$

We can set $$p$$ to be different values to represent different norms such as L1 and L2.

This is not decomposable.

## Total Variation Distance

TV is formally defined as the max difference between the probability distributions over subsets $$F$$ of possible events ($$F$$ is the set of all subsets, $$A$$ is the optimal subset to maximize the probability difference):

$$
TV(P,Q)=\sup_{A\in F}|P(A)-Q(A)|
$$

For discrete distributions it can be computed as:

$$
TV(P,Q)=\frac{1}{2}|P(X) - Q(X)| =\frac{1}{2}\sum_x|P(x)-Q(x)|
$$

We can prove this equality by considering the optimal subset of events $$A$$. Let’s consider two events:

- $$B$$: $$\{x \in X \text{  s.t. }  P(x) > Q(x)\}$$
- $$C$$: $$\{x \in X \text{  s.t. }  Q(x) > P(x)\}$$

We can set the optimal set $$A$$ to the larger set between $$B$$ and $$C$$. This is due to the absolute value and we need to consider both cases. Because we are splitting the full set $$X$$ into two, we know that $$P(A) \geq 1/2$$ or $$Q(A)\geq 1/2$$, depending on whether we chose $$B$$ or $$C$$.

Let $$A'$$ be the complement of $$A$$, containing all events in $$X$$ not present in $$A$$. From conservation of probability mass we get:

$$
|P(A) - Q(A)| = |P(A') - Q(A')|
$$

To change $$P$$ to $$Q$$, we move probability mass from $$A$$ to $$A’$$. The sum of both sides of this equation equals the L1 norm:

$$
|P(X) - Q(X)|  = |P(A) - Q(A)| + |P(A') - Q(A')|
$$

To isolate $$\vert P(A) - Q(A)\vert $$, we can just divide the L1 norm by 2.

The intuition of the 1/2 is that you want to move 1/2 of the difference in probabilities to make it equal. Let’s say there are two events and $$P$$ assigns probabilities [0.7, 0.3] and $$Q$$ assigns [0.5, 0.5]. The L1 difference is 0.4, but you want to move 0.2 to get equal probabilities. We are measuring the amount of probability mass that has to be moved. This is equivalent to the Wasserstein distance with the distance function $$d(x_i, x_j)$$ always set to 1. 

$$
TV(P,Q) = |P(A) - Q(A)| = \frac{1}{2}\sum_x|P(x)-Q(x)|
$$

This has the property in that it is bounded by the KL divergence through [Pinsker’s inequality](https://en.wikipedia.org/wiki/Pinsker%27s_inequality):

$$
TV(P,Q) \leq\sqrt{\frac{1}{2}D_{KL}(P||Q)}
$$

This is not decomposable.

## Wasserstein

Also known as earth mover’s distance (EM), this represents the amount of probability mass that needs to be moved to change one distribution into another. 

This requires determining an optimal mapping between points in the input $$X$$ to move probability mass.  This is called the optimal transport plan.

[https://alexhwilliams.info/itsneuronalblog/2020/10/09/optimal-transport/](https://alexhwilliams.info/itsneuronalblog/2020/10/09/optimal-transport/)

$$
W_p(P, Q) = \left( \min_{\gamma} \sum_{i=1}^n \sum_{j=1}^m \gamma_{ij} \, d(x_i, x_j)^p \right)^{1/p} \\ \sum_{j=1}^m \gamma_{ij} = p_i, \quad \sum_{i=1}^n \gamma_{ij} = q_j, \quad \gamma_{ij} \geq 0,
$$

$$\gamma_{ij}$$ represents the probability mass transported from $$x_i$$ to $$x_j$$. $$\gamma_{ii}$$ represents the probability that stays on $$x_i$$. We require the gammas to add up to $$p_i$$ so all of the original probability mass is allocated, even if it on the existing point. 

$$d$$ is a distance measure between points. Wasserstein distance also considers how far you have to move the probability mass. This can be the norm or euclidean distance. This means it would be more expensive to move mass from 1 to 5 than from 1 to 2.

$$p$$ is a power term which is equivalent to the parameter in norms.

To optimize this, we first compute a cost matrix which is NxN matrix containing all the pairwise distances under $$d$$. Finding the optimal transport plan $$\gamma$$ is a linear programming problem that can be solved with different methods, which are out of scope here. 

This is not decomposable. It is also is not reparameterization invariant if the distance measure utilizes the input values. If the distance function outputs a constant value, then it is invariant, but then it just reduces to a $$L_p$$ norm.

However, EM better handles cases where one of the probability distributions is zero. Recall that $$D_{KL}(A\Vert B)$$ is infinite where $$B$$ is zero and $$A$$ is non zero. With high dimensional probability distributions, there may be less overlap due to the curse of dimensionality. In these cases EM may be more robust. 

This is especially relevant for generative modeling. The [WGAN](https://arxiv.org/abs/1701.07875) paper argues that EM is superior for GANs. This is because WGAN is fully continuous and does not have the discontinuity that KL has when the denominator is 0. This occurs when the supports of the generator and discriminator are mismatched. This happens because each distribution lies in a low dimensional manifold of a high dimensional latent space.

## Jensen-Shannon

Jensen-Shannon divergence is a symmetric version of KL divergence. It is the sum of the forward KL of each distribution with respect to $$M$$, which is the average of the two distributions. 

$$
D_{JS}(P||Q) = D_{KL}(P || M) + D_{KL}(Q || M) \\ M(x) = (P(x) + Q(x)) / 2 
$$

The usage of a combined probability distribution makes it impossible to decompose this. 

## $$f$$-Divergence

$$f$$-divergence is class of divergences which can be formulated as follows:

$$
D_f(P||Q)=\sum_xQ(x)f(\frac{P(x)}{Q(x)})
$$

$$f$$ can be any function that meets certain conditions.

- $$f(0)=f(0^+)$$, it is continuous at zero approaching from the right, the function should also be defined at 0
- $$f(1)=0$$, this ensures identical distributions have 0 divergence
- $$f$$ is convex on the range $$(0, \infty)$$
    - Applying $$f$$ to the ratio would get a value less than or equal to applying $$f$$ separately to the numerator and denominator.
    - $$f$$ measures the surprise when observing a data point from $$P$$, convexity ensures the surprise strictly increases as the distributions become more different
- $$af(\infty) = \lim_{x \to \infty} af(x) = af'(\infty)$$ for $$a > 0$$

We get the KL divergence when $$f(x) =x\log(x)$$:

$$
\begin{aligned} D_f(P||Q)&=\sum_xQ(x)f(\frac{P(x)}{Q(x)}) \\ &=\sum_xQ(x)\frac{P(x)}{Q(x)}\log(\frac{P(x)}{Q(x)}) \\ &=\sum_xP(x)\log(\frac{P(x)}{Q(x)})\end{aligned} 
$$

We get total variation when $$f(x) = \frac{1}{2}\vert x-1\vert $$

$$
\begin{aligned} D_f(P||Q)&=\sum_xQ(x)\frac{1}{2}|\frac{P(x)}{Q(x)}-1| \\ &=\frac{1}{2}\sum_x|P(x)-Q(x)|\end{aligned}
$$

This divergence is not always decomposable. It would require the function $$f$$ to be decomposable.

$$
f(\frac{P(X,Y)}{Q(X,Y)}) = f(\frac{P(X)}{Q(X)})+E_{P(X)}f(\frac{P(Y|X)}{Q(Y|X)})
$$

Generally, only the logarithm has this decomposition property. KL divergences are special cases of $$f$$ divergences that are decomposable.

While $$f = \log(x)$$ is linearly decomposable by replacing the weighting distribution with Q, this results in the negative reverse KL divergence. This is not a valid divergence because it is non-positive. Note that $$-\log(x)$$ gives us the reverse KL divergence.

---

Other divergences: $$\chi^2$$-divergence, Squared Hellinger distance, Le Cam distance, [Renyi Divergence](https://en.m.wikipedia.org/wiki/R%C3%A9nyi_entropy#R%C3%A9nyi_divergence)

# Sources

https://dibyaghosh.com/blog/probability/kldivergence/

https://blog.alexalemi.com/kl.html

https://blog.alexalemi.com/kl-is-all-you-need.html

https://people.lids.mit.edu/yp/homepage/data/LN_fdiv.pdf

[https://people.ece.cornell.edu/zivg/ECE_5630_Lectures6.pdf](https://people.ece.cornell.edu/zivg/ECE_5630_Lectures6.pdf)

[https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures](https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures)

https://arxiv.org/abs/1701.07875

https://lilianweng.github.io/posts/2017-08-20-gan/