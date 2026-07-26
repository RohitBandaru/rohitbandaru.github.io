---
layout: post
title: "Foundation Models for Robotics: Vision-Language-Action (VLA)"
description: "How vision-language-action models bring foundation models to robotics, covering RT-1, RT-2, OpenVLA, pi0, and Hi Robot."
tags: robotics, computer-vision, deep-learning, reinforcement-learning, transformers
thumbnail: assets/img/blog/vla/pi0.png
citation: true
toc:
  sidebar: left
keywords: vision language action, VLA, robotics foundation models, embodied AI, multimodal robotics, robot learning, foundation models, transformer architectures, visual reasoning, language grounding, action prediction, robotic manipulation, autonomous systems
---

AI research has converged around foundation models across various domains. Rather than developing specialized models for specific tasks like visual question answering, researchers now employ versatile multimodal language models that tackle a wide variety of problems. This shift has extended to models operating in the physical world, particularly robotics.

A long-standing goal of robotics is to develop versatile systems that perform effectively across diverse settings, applications, and hardware configurations. This makes deployment easier, improves model robustness, and enables transfer learning between domains. We've had tremendous success achieving this with foundation models for language. LLMs can be applied to many different problems and easily adapted with limited task-specific data, benefiting from learning across domains. We seek to achieve this same type of foundation model for embodied AI.

This blog post focuses on how foundational LLMs are being adapted for robotics applications. We will make frequent analogies to show how robotics research aligns with LLM research, cover the unique challenges of robotics and the adaptations needed to address them, and explore how the fields have become tightly coupled so that progress in one field benefits the other.

Looking toward the future, we should consider what form AGI will take. Many researchers believe intelligence can be built in the digital world through LLMs. This means agents for coding, research, and knowledge work. Many believe real-world tasks like robotics will be easily solved once AGI is achieved in text. There's an alternate view that true intelligence requires understanding the physical world, and that AI shouldn't be constrained to operating through digital chat interfaces. In building embodied AI, vision-language-action models represent another avenue that complements rather than competes with LLM research.

## Data

The prevailing narrative is that data scarcity is the primary bottleneck holding robotics back from achieving the success we've seen in language applications. While hardware and algorithmic advances remain important, limited data availability represents the current constraint on robotics progress. To understand the data constraints of robotics, we need to examine the type and scale of data needed to train effective robotics models in comparison to LLMs.

## LLM Training Scale

Let's first establish the data scale used for LLM/VLM training so we can then understand the data gap of robotics.

LLM pretraining operates on trillions of tokens. [Qwen2.5](https://arxiv.org/abs/2412.15115) trains its base model on 18 trillion tokens. The training follows a clear hierarchy: pretraining uses internet-scale data, while post-training stages use progressively less but higher-quality data. Qwen2.5 uses approximately 1 million examples (32 billion tokens with 32k sequence length) across all post-training phases, demonstrating how each successive stage requires orders of magnitude less data but with better task alignment.

VLMs follow similar patterns but with less transparency about exact data volumes. [Kimi-VL](https://arxiv.org/abs/2504.07491) reports using trillions of vision tokens, fewer than text tokens but still at massive scale. While they don't specify image counts or video durations, the scale likely reaches billions of images.

{% include figure.liquid loading="eager" path="assets/img/blog/vla/kimi.png" alt="Kimi-VL data scale" class="img-fluid mx-auto d-block" width=600 source="https://arxiv.org/abs/2504.07491" %}

Token counts provide only rough data comparisons since they depend heavily on tokenizer vocabulary size and encoding schemes. A tokenizer with 150k vocabulary packs more information per token than one with 30k vocabulary. For vision data, token counts depend on ViT patch sizes, input resolution, and video frame sampling rates, making direct comparisons even more problematic.

Fine-tuning data varies widely in scale. Pretraining makes these models sample efficient, so most use cases require only billions of tokens or less. Some applications, however, use substantially more data. For example, [Qwen2.5-Coder](https://arxiv.org/abs/2409.12186) constructs a code-heavy dataset of 5.5 trillion tokens for continued pretraining on top of the Qwen2.5 base model. The general principle is that models are trained on as much data as possible. Despite this variation, billions of tokens is typically sufficient to enable strong LLM performance in text domains.

### Robot Data Scale

The data for training robots mainly consists of videos accompanied by robot actions. Actions are typically floating-point values to map to low-level commands to control the hardware. These robot “episodes” can also contain additional sensor data or multiple camera views. This data is diverse due to different robot embodiments, each with varying sensors, action spaces, and capabilities. The goal of a foundation model is to train a single model that can be deployed across many robots, requiring a large dataset that contains data from a diverse set of robots.

{% include figure.liquid loading="eager" path="assets/img/blog/vla/trajectory.png" alt="Sample action trajectory from DROID" class="img-fluid mx-auto d-block" width=600 source="https://droid-dataset.github.io/" %}

One of the most significant efforts in building this kind of dataset is [Open X-Embodiment](https://robotics-transformer-x.github.io/). This collective effort from 21 academic and industrial institutions amalgamates 72 different datasets from 27 different robots, covering 527 skills across 160,266 tasks. The dataset standardizes data from diverse robot types (single arm, double arm, wheeled robots) with varying sensors and action spaces into a unified format.

If we want to compare the training scale directly with VLMs, we have to rely on token count. It’s difficult to quantify the scale of videos that VLMs train on since they only report total token counts. These total token counts are not useful without knowing the frame sampling frequency and tokens per image.

VLMs are trained on trillion token internet-scale data. It is possible to get competitive in token count with robotics data. Open X-Embodiment contains 2.4 million episodes. If we assume 30 seconds per episode, 30 Hz frame sampling, and ~512 vision tokens per frame, we can get over a trillion tokens. This token count also doesn’t account for additional sensor and camera views, and the action tokens. However, this token count shouldn’t be compared with the trillions of tokens that LLMs train on. The parameters used in this calculation are estimates only. Also, there is a lot of redundancy in these video tokens. We should expect a trillion video tokens to have less signal than a trillion text tokens.

{% include figure.liquid loading="eager" path="assets/img/blog/vla/embodiment_distribution.png" alt="Embodiment Diversity in Open X-Embodiment" class="img-fluid mx-auto d-block" width=600 source="https://arxiv.org/abs/2310.08864" %}

Rather than token counts, we should focus more on the quantity and diversity of video. It is hard to say how many videos the largest VLMs are trained on, but we do know that billions of general videos are available, far more than robot videos. This is because there are far more videos available through internet-scale pretraining than there are robot videos. The way to address this is to utilize non-robot videos to take advantage of transfer learning. The diversity of robot data is limited. While the 27 different embodiments represent unprecedented variety in robotics research, this is still minuscule compared to the diversity found in internet-scale datasets. One interesting avenue to address limited embodiment diversity is training in simulation. Skild AI [reports](https://www.skild.ai/blogs/omni-bodied) training on 100,000 different robots generated in simulation.

{% include figure.liquid loading="eager" path="assets/img/blog/vla/task_distribution.png" alt="Task distribution in Open X-Embodiment" class="img-fluid mx-auto d-block" width=600 source="https://arxiv.org/abs/2310.08864" %}

These datasets are also limited in the diversity and coverage of tasks. For example, simple tasks are overrepresented while highly dexterous tasks are less covered.

Robotics datasets are large and are growing. However, they are limited in the diversity of embodiments, environments, and tasks. This is a fundamental challenge in building a general-purpose foundation model for robotics.

### Transfer Learning

LLM applications utilize three types of data in a hierarchical structure, with each level decreasing in availability but increasing in quality and task alignment: internet-scale text data for pretraining → instruction tuning data → task-specific data. Robotics follows a similar data hierarchy: internet-scale video data → ego-centric video data → robot data with actions → specific robot data with actions. There are some efforts to collect large amounts of ego-centric data ([Ego4D](https://arxiv.org/abs/2110.07058)), as this kind of data is easier to collect than robot data (can use smart glasses). LLM researchers developed these training recipes due to the limited amount of high-quality instruction-following data that is available. We can follow this pattern for robotics.

These larger datasets used to train LLMs/VLMs are meant for models to learn an entire data modality (text, images). Robotics data uses the same modalities of data. The only new modality is actions, which is why we need to fine-tune to create VLAs. This is also why we want to leverage language and vision models that are trained on massive amounts of image/text data. Robotics data can be viewed as a type of fine-tuning data.

However, maybe we should consider interacting with the physical world to be a completely new domain. This means it requires a lot of data to learn, and transfer learning from digital tasks is limited. Then it makes sense to compare the scale of action data with the scale of image or text data. However, this is largely a philosophical question. The empirical reality is that robotics is far more data hungry than a typical LLM application.

### Scaling Laws

If we assume that scaling laws remain consistent across domains, model performance should improve with increased data volume. This highlights the need to train on the largest possible datasets. [Qwen2.5-Coder](https://arxiv.org/abs/2409.12186) demonstrates this principle by training on 5.5 trillion tokens, significantly more than typical fine-tuning applications, to achieve strong coding performance. Language model scaling laws are logarithmic, meaning performance scales linearly with the logarithm of training tokens rather than the raw token count.

This work by [Lin et al. 2025](https://arxiv.org/abs/2410.18647) argues that scaling laws in robotics depend more on the number of environments and objects rather than the number of demonstrations. A large amount of robotics data is collected in constrained lab environments. This suggests that large-scale deployment and data collection would unlock significant gains. It would also be interesting to study the scaling laws for general-purpose models on the number of embodiments.

Robotics is considered data-constrained because the amount of robotics data readily available is orders of magnitude less than what's needed to reliably deploy general-purpose robots. This constraint exists because robot data collection is expensive and time-consuming, there is limited robotics data currently available compared to other domains, and robot intelligence appears to require substantial amounts of data to achieve reliable performance.

### Data Collection

The fundamental challenge with robotics data collection is that it requires large-scale deployment of robots to generate sufficient training data. However, robots need to be effective and affordable enough to enable broad deployment in the first place. Many companies are betting that leveraging pretrained VLMs will make robots capable enough to reach this large-scale deployment threshold.

Large-scale deployment alone may not be sufficient to solve robotics. Tesla has had large-scale deployments of cars collecting massive amounts of data for years, yet has only recently started deploying autonomous vehicles successfully.

Despite these challenges, many companies are investing heavily in data collection to bridge the gap toward large-scale deployment. [Figure](https://www.figure.ai/news/figure-announces-strategic-partnership-with-brookfield) is partnering with a real estate company to conduct large-scale humanoid robot data collection in thousands of different homes. This approach is expensive but may be necessary to achieve functional general-purpose robots.

## LLMs for Robotics

We first consider how robotics can be treated as an application of LLMs. We will focus on [Gemini Robotics](https://arxiv.org/abs/2503.20020), which applies a SOTA LLM for robotics without significant changes in architecture or training.

First, let’s consider what properties we need from an LLM in order for it to be useful for robotics. If we want to power robots with AI models, we need two capabilities. **Embodied reasoning** is the ability of models to understand the physical world and how to interact with it. We also need models capable of **embodied action.** This means that the model can use its embodied reasoning to interface with hardware and perform complex actions in the real-world.

The benefits of applying LLMs are that these models come pretrained with immense world knowledge. They also display common high-level intelligent capabilities such as reasoning and long-term planning. The goal can then be simplified to just adapting LLMs for robotics, which we expect to be simpler and more data efficient than training from scratch.

### Embodied Reasoning

In the Gemini Robotics paper, the authors define embodied reasoning as follows.

> “The set of world knowledge that encompasses the fundamental concepts which are critical for operating and acting in an inherently physically embodied world.” - [Gemini Robotics](https://arxiv.org/abs/2503.20020)

This is an ability of vision language models (VLMs) and is not necessarily tied to robotics. VLMs refer to multimodal language models that can process visual and text inputs. Testing embodied reasoning simply involves prompting VLMs about images. Classical computer vision tasks like object detection and multi-view correspondence fall under embodied reasoning. These tasks are all expressed as language prompts.

{% include figure.liquid loading="eager" path="assets/img/blog/vla/er_tasks.png" alt="Gemini Robotics embodied reasoning examples" class="img-fluid mx-auto d-block" width=600 source="https://arxiv.org/abs/2503.20020" %}

Embodied reasoning can also be tested through visual question answering. These questions test the understanding required to interact with the environment.

{% include figure.liquid loading="eager" path="assets/img/blog/vla/embodied_reasoning.png" alt="Gemini Robotics visual question answering examples" class="img-fluid mx-auto d-block" width=600 source="https://arxiv.org/abs/2503.20020" %}

In addition to general physical reasoning, we can take advantage of world knowledge to make decisions. For example, we could ask a robot to pick up a healthy snack in the kitchen. The world knowledge in the VLM would be used to figure out how to execute this ambiguous command.

Applying VLMs means that robotics can take advantage of advances in LLM capabilities. For example, we can take advantage of the impressive recent progress made in LLM reasoning and test-time compute. We can prompt the VLM to output a reasoning trace which allows it to better solve an embodied reasoning problem.

{% include figure.liquid loading="eager" path="assets/img/blog/vla/reasoning_trace.png" alt="Gemini Robotics reasoning trace example" class="img-fluid mx-auto d-block" width=600 source="https://arxiv.org/abs/2503.20020" %}

Gemini Robotics introduces an [Embodied Reasoning QA (](https://github.com/embodiedreasoning/ERQA)ERQA) benchmark that measures the embodied reasoning capabilities of LLMs. This benchmark requires no fine-tuning and works with any VLM. For models intended for embodied AI applications, performance results on this or other embodied reasoning benchmarks ([EmbodiedBench](https://embodiedbench.github.io/), [EAR-Bench](https://github.com/ZJU-REAL/OmniEmbodied)) should be reported.

### Embodied Action

Embodied reasoning enables models to understand the physical world and reason about objects, spatial relationships, and physical interactions. For robotics applications, we want to leverage this understanding to enable meaningful actions in the real-world. This means translating high-level understanding into precise control commands through the robot's hardware APIs. Each robot has a different interface, and the knowledge of how the robot is controlled is not present in the VLMs.

When applying LLMs to new domains, we typically follow a progression of increasingly complex methods: zero-shot prompting, few-shot prompting/prompt engineering, fine-tuning, and reinforcement learning. The application of LLMs to robotics follows this same pattern.

#### Zero-Shot Control

One way to express robot control is by specifying an API with code. The context to the VLM will include the documented API. The robot can reason based on this API and produce code that would execute a command.

{% include figure.liquid loading="eager" path="assets/img/blog/vla/gemini_zero_shot.png" alt="Gemini Robotics zero-shot control example" class="img-fluid mx-auto d-block" width=600 source="https://arxiv.org/abs/2503.20020" %}

[Levine et al. 2025](https://arxiv.org/abs/2407.08693) explore this in more depth. Reasoning improves performance on robotics tasks as well as the interpretability of the actions.

#### Few-Shot Control

Zero-shot control works well when the API is high-level. For high-level APIs, we may have another model generating the low-level actions. However, if we want the VLM to output low-level actions, we need to give some examples. A few-shot prompt that includes examples of how to use low-level commands to complete actions would give the model the necessary information. It would then extrapolate based on these examples on how to execute novel commands.

{% include figure.liquid loading="eager" path="assets/img/blog/vla/gemini_few_shot.png" alt="Gemini Robotics few-shot control example" class="img-fluid mx-auto d-block" width=600 source="https://arxiv.org/abs/2503.20020" %}

### Fine-tuning

Similar to traditional LLM applications, we can enhance VLMs for robotics by fine-tuning them on action data, creating what are known as vision-language-action (VLA) models. This approach bridges the gap between understanding the physical world and executing precise actions.

The fine-tuning process follows a supervised learning paradigm where human operators manually demonstrate tasks with robots. These demonstrations generate datasets of visual observations paired with corresponding low-level action trajectories. This approach is known as behavior cloning or imitation learning in robotics, but can also be considered a form of supervised fine-tuning (SFT) process as used in LLM training.

{% include figure.liquid loading="eager" path="assets/img/blog/vla/gemini_vla.png" alt="Gemini Robotics fine-tuning approach" class="img-fluid mx-auto d-block" width=600 source="https://arxiv.org/abs/2503.20020" %}

Modern implementations like Gemini Robotics employ a distributed architecture that balances capability with operational efficiency. A large VLA model runs in the cloud for high-level reasoning and planning, while a compact action decoder resides on the robot itself, translating model outputs into low-latency control signals.

Given the current research focus on fine-tuning foundation models for robotic applications, the remainder of this blog will explore VLAs in greater detail.

### Reinforcement Learning

A natural extension of this approach is training VLAs with reinforcement learning. While RL for LLMs has become a very active research area recently, we see limited adoption of these methods for VLAs. This limitation exists because robots must physically perform actions to generate training data. In contrast, methods like RLHF/RLVR allow LLMs to generate responses in various contexts that are automatically verified by another model or algorithm. This online learning process scales easily by running multiple LLMs in parallel.

However, scaling online learning becomes significantly more challenging when it requires physical actions in the real-world. Researchers have attempted to address this by creating environments where multiple robots can interact simultaneously. For example, Google built a [robot arm farm](https://arxiv.org/abs/1603.02199) in 2016. Yet in the context of modern deep learning, a scale of 14 robots is minuscule. We would need millions. We may be able to achieve this once robots are able to be mass-deployed. Then there will be a new set of challenges if we want to do distributed online reinforcement learning.

{% include figure.liquid loading="eager" path="assets/img/blog/vla/robot_farm.png" alt="Google robot arm farm" class="img-fluid mx-auto d-block" width=600 source="https://arxiv.org/abs/1603.02199" %}

There is a rich body of literature to work around the challenge of scaling online learning. The robot policy can be learned in simulation or in the real-world (which is more expensive and slower).

We have seen a lot of progress in applying RL on LLMs to improve performance on verifiable tasks such as math and coding. The advancements here may translate into robotics. The main gap is that we can't quickly execute a trajectory in the real-world.

With the SFT approach, the robot policy is limited by the data it is fine-tuned on, since it is optimized to imitate it. With RL, the policy is optimized directly on task performance and can learn trajectories that are better than what is in the datasets. We don’t want robot performance to be limited by how well humans perform with teleoperation. In order for RL to be applicable for real-world actions, this likely requires a strong base policy trained through SFT along with sample efficient RL methods.

## Synthetic Data

LLM training is increasingly relying on synthetic data generation. World models can be thought of as the source of synthetic data for the real-world. World models are another big area of research relevant for robotics. These models can be thought as the source of synthetic data for robotics that can be used both for training/eval data generation, and RL environment creation.

World models generate synthetic training data by predicting future video frames given robot actions. Sim-to-real transfer is a huge area of research in robotics, where the goal is to train policies in simulation and deploy them in the real-world. World models enable generating realistic simulations of different real-world environments, which are becoming indistinguishable from real data.

[1X](https://www.1x.tech/1x-world-model.pdf) uses world models for evaluation in simulation. However, current world models suffer from high computational latency and accumulate errors over longer prediction horizons. There are some efforts like [V-JEPA 2](https://arxiv.org/abs/2506.09985) that aim to use world models in the policy itself, but this is limited by the latency of the world model.

## Language in Robotics

Do we need robots to be language promptable? A general-purpose robot needs to understand language. Without language the robot would be restricted to a subset of possible actions. For example, you can have a button for washing the dishes and one for doing the laundry. But eventually you'll want the robot to do a task for which there is no button.

Language also serves as a source of world knowledge. When you ask a robot to "prepare a healthy snack," the robot can leverage knowledge about nutrition and food categories from its pretrained language model without explicit programming for every food item.

I would argue that language understanding is a necessary component for a general-purpose robotics foundation model. However, it is an open question on whether the LLM has to be the main component in the model. For example, we may one day build powerful vision models for robotics, and add a small language component. Whereas with VLAs, we do the opposite.

The necessity of language for general-purpose robotics is one reason why many are betting on VLAs.

## VLA Architectures

It is possible to fine-tune an LLM to predict robot actions directly. This approach treats actions as discrete tokens and formats input data to match the model's requirements. However, specific architectural changes can significantly improve these models' effectiveness.

As we explored in my [VLM blog post](https://rohitbandaru.github.io/blog/Vision-Language-Models/), enabling LLMs to understand images requires architectural adaptations. Similarly, getting these models to understand and perform actions demands additional modifications. We will now examine the common architectural changes used to build vision-language-action models.

## Discrete Actions

LLMs are built to predict discrete tokens. The simplest implementations of VLAs discretize actions so the model can predict them as tokens in its vocabulary. The actions are trained and decoded autoregressively, with discretized tokens added to the model's vocabulary. During inference, the output is detokenized, meaning discrete tokens are mapped back to continuous actions for robot execution.

{% include figure.liquid loading="eager" path="assets/img/blog/vla/openvla.png" alt="OpenVLA discrete action tokenization" class="img-fluid mx-auto d-block" width=600 source="https://arxiv.org/abs/2406.09246" %}

[RT-2](https://arxiv.org/abs/2307.15818) and [OpenVLA](https://arxiv.org/abs/2406.09246) are two works that implement discretized autoregressive action training. Although it's possible to represent actions as text using the normal text tokenizer (treating this as standard SFT), it's more effective to treat actions as distinct tokens.

The autoregressive loss is typically applied only to the actions, with visual and text tokens treated as a prefix. This approach constrains the model to predict only actions rather than text, enabling a smaller output layer and more efficient cross-entropy calculation. However, it would be possible to simultaneously fine-tune on the text prompt by running SFT next-token prediction on both the prompt and actions. When fine-tuning only on actions, the output computation is smaller since cross-entropy operates over the action space rather than the full text vocabulary. In practice, it seems like it is more common to run SFT on vision-language data separately from action fine-tuning.

## Continuous Actions

Although discretization is the most natural way to get VLMs to predict actions, this comes at a cost. Discrete tokens can’t effectively model smooth continuous trajectories so the resulting robot behavior may seem unnatural. It also limits the granularity in which we can control the robot, imposing an artificial limitation on the robot’s abilities. If we want higher frequency control for dexterous tasks, discrete tokens will become highly repetitive and lacking in the ability to represent subtle movements.

Discretization also introduces a tradeoff between output vocabulary size and action granularity. We could increase the number of discrete buckets, but this can also make it more difficult to train with limited data. Since actions are inherently continuous, it would be better to have model architectures that can predict continuous actions directly. Continuous action prediction enables finer grained actions that allow for high frequency dexterous tasks.

A naive solution is to fine-tune a VLM to predict continuous actions using a linear layer on top of the output embeddings with MSE loss. However, this approach is not effective when outputs have multiple modes. An MSE loss would train the model to average these modes, and this average is not a good action.

Some works, such as [Liang et al. 2025](https://arxiv.org/abs/2502.19645v1), train VLAs with regression loss to predict continuous actions. However, this approach is not as effective on its own compared to discretization-based methods. They need to co-train with discrete action prediction for this to be effective.

## Flow Matching / Diffusion

Flow matching and diffusion are the two most effective ways to train transformers to generate continuous outputs. These methods were first researched for image generation, but are well applied to action generation as well. We see some VLA architectures use flow matching while others use diffusion. Flow matching and diffusion are [equivalent](https://diffusionflow.github.io/) methods so we will not concern ourselves with the distinction.

The $$\pi_0$$ [model](https://arxiv.org/abs/2410.24164v1) uses flow matching. Flow matching training involves sampling noise and time, and training a vector field to predict the direction to update the noise to get to the target. Inference involves taking multiple steps to integrate on the time to go from noise to a predicted final target. Diffusion is similar but is framed as removing a step of noise from an image.

```python
def flow_matching_training(
        observation, # contains text prompt and latest images
        true_actions # next chunk of continuous actions
    ):
    # Sample random noise and timestep
    noise = random_normal(shape=true_actions.shape)
    t = random_uniform(0, 1)

    # Create noisy interpolation between true_actions and noise
    # t=0: pure true_actions, t=1: pure noise
    x_t = t * noise + (1 - t) * true_actions

    target_velocity = noise - true_actions  # Points FROM actions TO noise

    # Forward pass through vision-language-action model
    vision_lang_tokens = encode_vision_language(observation)
    action_tokens = encode_actions_with_time(x_t, t)
    predicted_velocity = transformer([vision_lang_tokens, action_tokens])

    # Train to predict the velocity field
    loss = mse(predicted_velocity, target_velocity)

def flow_matching_inference(observation, num_steps=10):
    x = random_normal(action_shape)  # Start from noise
    dt = 1.0 / num_steps

    # Iteratively follow the flow: t goes from 1 → 0
    for step in range(num_steps):
        t = 1.0 - step * dt

        # Get velocity prediction at current state and time
        vision_lang_tokens = encode_vision_language(observation)
        action_tokens = encode_actions_with_time(x, t)
        velocity = transformer([vision_lang_tokens, action_tokens])

        x = x - dt * velocity  # Follow flow BACKWARD (noise → actions)

    return x  # Continuous actions
```

Flow matching is more effective at generating continuous actions. The main drawback is that it is slower to train than autoregressive architectures. Flow matching can result in faster inference than autoregressive prediction since we can predict a whole chunk at once. To generate a chunk of 50 actions, we may need to only run 10 inferences for flow matching. This is faster than decoding 50 steps autoregressively. We can parallelize inference with autoregressive models. This is only in the case that the number of flow steps is less than the number of steps in the chunk.

## Action Expert

Although it's possible to fine-tune a VLM directly to predict actions, VLAs empirically perform better when using separate weights for action learning. This approach gives the model additional capacity to learn action outputs without interfering with the VLM's existing capabilities.

These specialized weights can be added on top of the transformer, such as a multilayer MLP instead of a single output projection layer. However, incorporating these extra parameters into the transformer layers themselves proves more effective.

$$\pi_0$$ adds a 300 million parameter action expert to a PaliGemma VLM. The model's inputs are divided into a prefix and suffix. The prefix contains vision tokens for recent image frames and text prompts, which can be processed directly by the pretrained VLM. The suffix adds robotics inputs: $$q_t$$ representing the proprioceptive state (current position of robot), and noise (interpolation between noise and actions for flow matching).

{% include figure.liquid loading="eager" path="assets/img/blog/vla/pi0.png" alt="π₀ action expert architecture" class="img-fluid mx-auto d-block" width=600 source="https://arxiv.org/abs/2410.24164v1" %}

The action expert is a transformer model with the same number of layers but smaller embedding dimensions and MLP widths. The attention heads and per-head embedding dimension must match the main model to allow prefix tokens in the attention mechanism. When processing, suffix tokens go through the action expert transformer while incorporating the KV embeddings from the prefix (these are computed once then cached). The prefix uses full self-attention, while the suffix uses causal attention. This makes sense since we don't need to predict tokens in the prefix. This smaller action expert architecture enables faster inference while still leveraging the capabilities of the larger VLM.

Various modifications to this architecture are possible. For instance, [SmolVLA](https://arxiv.org/abs/2506.01844) alternates between self-attention among action tokens and cross-attention with prefix tokens. This approach further reduces computational requirements compared to the semi-causal full attention mechanism used in $$\pi_0$$.

## Action Chunking

We see in the $$\pi_0$$ and many other VLA architectures that we predict multiple tokens at the same time. This is called action chunking and was introduced by [Zhao et al. 2023](https://arxiv.org/abs/2304.13705). $$\pi_0$$ implements this by running flow matching on multiple action tokens in parallel. Since flow matching is an iterative process at inference time, we get an efficiency gain from this parallelization. This is similar to multi-token prediction used in [DeepSeek-V3](https://arxiv.org/abs/2412.19437v1).

At inference time, the robot is continuously executing actions at 50 Hz. The model operates at 50 Hz with action chunks of size 50 to predict a full second of action. There are two inference strategies. In the synchronous strategy, we cycle through a process of generating a chunk, executing the chunk, and collecting observations for the next chunk. This means that during inference we introduce a delay where the robot is not performing any action.

We can also implement a real-time strategy. To avoid inference delays, we have to introduce lag in the observations. For example, every 0.5 seconds, while the robot continues executing the current action chunk, the model predicts the next 25 actions based on a new observation. The inference time is only 73ms for on-board inference (or 86ms for off-board inference), which is well within the 0.5 second window. However, this means the observations used for planning are always 0.5 seconds stale. By the time the robot starts executing actions 26-50, those actions were planned based on what the robot observed 0.5 seconds ago, before it executed actions 1-25.

Chunking is effective since actions have a lot of redundancy. This means that these consecutive tokens are highly correlated so they can usually be predicted in parallel. If the actions varied drastically between time-steps, action chunking likely wouldn’t be stable.

The original work ([Zhao et al. 2023](https://arxiv.org/abs/2304.13705)) also evaluates temporal ensembling, which is later found not to be effective in $$\pi_0$$. This involves introducing an overlap between chunks and taking the average of the predicted actions. However, incorporating temporal ensembling could be useful if we want to increase the frequency of observations.

{% include figure.liquid loading="eager" path="assets/img/blog/vla/act.png" alt="Action chunking methodology" class="img-fluid mx-auto d-block" width=600 source="https://arxiv.org/abs/2304.13705" %}

### Real-Time Action Chunking

The main problem with action chunking is that we either have to have inference delays or introduce lag in the observations. If we use real-time with observation lag, the model doesn't see the result of the second half of the latest chunk while planning the next chunk, since those actions haven't been executed yet. This can cause discontinuities in the action that lead to jerky, unnatural movement at chunk boundaries. This delay can also be exacerbated by latency in retrieving the observations. It is possible to address this issue by adding the unexecuted actions to the input of the model, but the $$\pi_0$$ model doesn't seem to do this.

{% include figure.liquid loading="eager" path="assets/img/blog/vla/chunk_discontinuity.png" alt="How delays introduce discontinuities in action chunking" class="img-fluid mx-auto d-block" width=600 source="https://www.pi.website/download/real_time_chunking.pdf" caption="This illustration shows how delays introduce discontinuities. This also shows how temporal ensembling results in erroneous trajectories." %}

[Black et al. 2025](https://www.pi.website/research/real_time_chunking) proposes an inference time solution that does not require training changes. The main goal is to improve consistency between chunks as simply smoothing over the discontinuities wasn’t working well enough. They treat generating new action chunks like an [image inpainting](https://huggingface.co/docs/diffusers/en/using-diffusers/inpaint) problem. This is another area where VLA research has taken advantage of image generation.

We can first introduce some terminology to define the inference process. $$H$$ (prediction horizon) represents how many actions are predicted in each chunk. $$s$$ (execution horizon) represents how many actions we execute from each chunk before starting inference for the next chunk. $$d$$ (inference delay) represents how many controller timesteps it takes to generate a chunk. If we have 73ms delay at 50Hz, $$d$$ is 3 timesteps.

With $$H=50$$ and $$s=25$$, if we start inference after executing $$s$$ actions, then by the time Chunk 1 is ready ($$d$$=3 timesteps later), we'll have executed $$s+d=28$$ actions from Chunk 0. This means when we eventually switch to Chunk 1, the first d=3 actions need to be frozen to match actions 25-27 from Chunk 0. However, actions 28-49 can be edited while we predict Chunk 1. We can consider 3 regions:

{% include figure.liquid loading="eager" path="assets/img/blog/vla/rtc.png" alt="Action chunking regions: frozen, intermediate, and fresh" class="img-fluid mx-auto d-block" width=600 source="https://www.pi.website/download/real_time_chunking.pdf" %}

1. **Frozen region (first** $$d$$ **actions)**: Weight = 1. These actions from the previous chunk will have already executed by the time inference finishes, so they must be frozen to match what actually happened.
2. **Intermediate region (actions** $$d$$ **to** $$H-s$$**)**: Exponentially decaying weights from 1 to 0. These actions belong to the current chunk. The exponential decay encourages smoothness between chunks.
3. **Fresh region (last** $$s$$ **actions)**: Weight = 0. These actions are beyond the end of the previous chunk and need to be freshly generated with no constraints.

These weights are used in [pseudoinverse guidance (ΠGDM)](https://iclr.cc/virtual/2023/poster/11030) adapted for flow matching. At each denoising step during action generation, a gradient-based guidance term encourages the final generation to match the frozen and intermediate values from the previous chunk. We use the weights on this term to determine how close we want the action to be to the previously generated action.

## Hierarchical

We want these robot systems to be able to handle complex and even ambiguous commands. For example, we may want to prompt a robot with “organize my kitchen”. Such commands require longer term planning and breaking it down into lower level tasks. The goal of hierarchical systems is to have one system propose low-level instructions that another model executes.

The high-level policy takes a command and outputs both a response to the user and instructions for the low-level policy. For example, if you prompt the model to make a sandwich, the high-level policy will generate a sequence of actions like "1. Pick up bread 2. Place bread on plate..." The low-level policy then executes these commands through precise physical actions. This allows the high-level policy to focus on planning complex, long-horizon tasks.

$$\pi_0$$ does an evaluation where they break apart complex high-level commands to lower level simpler text commands, using both expert humans and another VLM. They find that $$\pi_0$$-HL and $$\pi_0$$-Human both improve the performance on complex tasks, compared with giving $$\pi_0$$ the complex prompt directly. This motivates further research into hierarchical systems where one model processes complex text prompts

In the [Hi Robot](https://arxiv.org/abs/2502.19417) work, the authors build this kind of hierarchical system with a high-level policy generating commands to a low-level policy. The high-level policy is just a VLM that outputs text, while the low-level policy generates actions.

{% include figure.liquid loading="eager" path="assets/img/blog/vla/hi_robot.png" alt="Hi Robot hierarchical architecture" class="img-fluid mx-auto d-block" width=600 source="https://arxiv.org/abs/2502.19417" %}

This hierarchical approach enhances the system's ability to handle longer-term and more complex tasks, while also improving interpretability by making the high-level policy outputs readable. The approach more than doubles performance on certain open-ended tasks such as table bussing and sandwich making, compared to $$\pi_0$$.

One gap with breaking part a high-level task into low-level commands is that the low-level policy loses global context. Sometimes different subtasks can be partially parallelized. For example if a robot is going to retrieve a tool for a subtask, it may save time by also picking up a tool for the next subtask. An alternative architecture could be to have a single model that is trained to first output text (similar to a reasoning trace in LLMs) and then generate actions. This can be considered a combined VLM/VLA architecture. However, this means that the high-level policy VLM has to be the same as the low-level policy’s, whereas we would prefer to use a larger model for the high-level policy.

$$\pi_{0.5}$$ implements this kind of combined [model](https://arxiv.org/abs/2504.16054). It is trained to predict a single subtask at a time based on the observation. It runs an inference loop where it generates a subtask based on the observation, executes the subtask to completion, and then generates a new subtask based on the latest observation.

{% include figure.liquid loading="eager" path="assets/img/blog/vla/pi05.png" alt="π₀.₅ combined VLM/VLA architecture" class="img-fluid mx-auto d-block" width=600 source="https://arxiv.org/abs/2504.16054" %}

Although this approach is effective in executing complex multi-step tasks, it is limited in the fact that it greedily only generates one subtask at a time. Some tasks may benefit from generating a complete plan of subtasks before executing. However, we may also want to update the plan as we complete subtasks. In a follow-up [work](https://www.physicalintelligence.company/research/knowledge_insulation), they introduce a stop grad between the action expert and the VLM backbone. The idea is that the VLM parameters should only be trained to output text. Gradients from predicting continuous actions can interfere with the VLM’s abilities. This is considered knowledge insulation. However, we still want the VLM to learn from the actions, so it is co-trained to predict discretized actions.

{% include figure.liquid loading="eager" path="assets/img/blog/vla/knowledge_insulation.png" alt="Knowledge insulation in VLA training" class="img-fluid mx-auto d-block" width=600 source="https://www.physicalintelligence.company/research/knowledge_insulation" %}

Training effective VLAs requires balancing different capabilities: continuous action prediction, discrete action understanding, and the semantic comprehension of the VLM.

## Model Size

[GR00T N1](https://arxiv.org/abs/2503.14734) and $$\pi_0$$ both use 2B parameter LLMs in their VLA architectures. Small models are required to enable on-device inference and real time latency. Some hierarchical approaches ([Figure Helix](https://www.figure.ai/news/helix) uses a 7B VLM) can use larger models for the lower frequency component. Gemini Robotics uses a lower latency LLM hosted in the cloud, which likely has an even larger parameter count. However, we are generally limited to using the smallest LLMs for robotics.

## Training Recipes

The $$\pi_0$$ [model](https://arxiv.org/abs/2410.24164v1) also emphasizes developing effective training recipes. These methods are inspired from LLM training. They take a VLM that is pretrained on internet-scale data. They train the action expert using the Open-X Embodiment dataset. They then fine-tune the full system with a dataset that they manually collect. They argue that post training on high-quality data is important as the lower quality data contains mistakes from teleoperation that we don’t want robots to imitate.

## Conclusion

Currently, reinforcement learning is one of the most active areas in AI research. Many researchers are trying to scale up RL training for LLMs in number of examples, environments, and tasks.

While hierarchical setups are useful to break apart complex tasks into subtasks, there are many alternative architectures. For example we could use [Coconut (Chain of Continuous Thought)](https://arxiv.org/abs/2412.06769) to simulate planning without forcing the steps to be human readable. We may see techniques from LLM reasoning research applied here.

Typically, VLAs operate on very short amounts of video. $$\pi_0$$ only sees 2 seconds of video. This is mainly due to the latency constraint. Although long-context LLMs can process hours of video, this comes at a latency cost. Videos are very expensive in terms of token counts. Currently it is infeasible to train an effective VLA to use long visual context. However further research into token compression methods in LLMs/VLMs could benefit robotics. It is an open question as to how much long visual context is needed for effective robotics.

Compared to other tasks, robotics is especially latency-sensitive. Digital tasks like coding don’t have a time limit. We would like digital tasks to run fast, but in robotics because the world is changing, being slow can affect your ability to accomplish the task. Advances in the inference latency of LLMs could enable us to deploy larger and more performant models to robots.

It will be exciting to see how LLM research and VLA robotics research influence each other going forward.

## Resources

### Papers

**Google DeepMind**

Google’s line of work set up VLAs to be the very active area of research it is today. We will now go through architectural improvements to make VLAs more effective.

- [SayCan](https://arxiv.org/abs/2204.01691)
- [RT-1](https://arxiv.org/abs/2212.06817)
- [RT-2](https://arxiv.org/abs/2307.15818)
- [RT-X](https://arxiv.org/abs/2310.08864)
- [Gemini Robotics](https://arxiv.org/abs/2503.20020)

**Physical Intelligence**

- [$$\pi_0$$: A Vision-Language-Action Flow Model for General Robot Control](https://www.physicalintelligence.company/download/pi0.pdf)
  - action expert trained with flow matching
- [FAST: Efficient Action Tokenization for Vision-Language-Action Models](https://arxiv.org/abs/2501.09747)
  - Autoregressive training is bad with repetitive tokens (which is common for robot actions). $$\pi_0$$-FAST does action compression to address this. Frequency sparse action sequence tokenization. 10% compression
  - Discrete Cosine Transform (DCT) used to compress action tokens.
- [**Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models**](https://arxiv.org/abs/2502.19417)
- $$\pi_{0.5}$$: [A Vision-Language-Action Model with Open-World Generalization](https://www.physicalintelligence.company/download/pi05.pdf)
- [Knowledge Insulating Vision-Language-Action Models: Train Fast, Run Fast, Generalize Better](https://www.physicalintelligence.company/download/pi05_KI.pdf)
- [Real-Time Execution of Action Chunking Flow Policies](https://www.physicalintelligence.company/download/real_time_chunking.pdf)

**Other VLA**

- [Octo](https://arxiv.org/abs/2405.12213) (May 2024)
- [OpenVLA](https://arxiv.org/abs/2406.09246) (June 2024)
  - open source
- [TinyVLA](https://arxiv.org/abs/2409.12514) (September 2024)
- [CogAct](https://arxiv.org/abs/2411.19650) (November 2024)
  - VLM + diffusion
- [Magma](https://arxiv.org/abs/2502.13130) (February 2025)
- [HybridVLA](https://arxiv.org/abs/2503.10631) (March 2025)
  - Cotraining with diffusion and autoregressive training
- [GR00T N1](https://arxiv.org/abs/2503.14734) (March 2025)
  - Nvidia
  - VLM for reasoning + Diffusion transformer for actions
- [SmolVLA](https://arxiv.org/abs/2506.01844) (June 2025)
  - Hugging Face
  - flow matching action expert (pi0 architecture)
- [MolmoAct](https://arxiv.org/abs/2508.07917) (August 2025)
  - open source (Allen Institute for AI)
  - action reasoning model (ARM)
  - Completely autoregressive
  - Pre/Mid/Post training phases
- Many many more

### Startups

Although their approaches aren’t always very public (except PI), it’s worth noting their approaches.

- DYNA
  - DYNA-1 (June 2025) [https://www.dyna.co/dyna-1/research](https://www.dyna.co/dyna-1/research)
    - First scalable foundation reward model for robotics
    - Folding laundry can be improved with a reward model. Humans are imperfect at folding. Behavior cloning is limited from teleoperated folding is imperfect. It is more powerful to use a strong reward model to judge the robot’s folding and use that signal to iterate and improve.
- Figure
  - Helix (February 2025) [https://www.figure.ai/news/helix](https://www.figure.ai/news/helix)
  - S2: VLM built off of an open source 7B VLM
    - Passes a latent vector to the S1 model
  - S1: Higher frequency 80M model
- Skild
  - (July 2025) https://www.skild.ai/blogs/building-the-general-purpose-robotic-brain
  - Contains a low frequency and high frequency model

### Talks

- [Stanford CS25: V2 I Robotics and Imitation Learning](https://www.youtube.com/watch?v=ct4tdyyNDY4)
- [Stanford Seminar - Connecting Robotics and Foundation Models, Brian Ichter of Google DeepMind](https://www.youtube.com/watch?v=wnPNRh-jtZM)
- [Stanford CS25: V3 I Low-level Embodied Intelligence w/ Foundation Models](https://www.youtube.com/watch?v=fz8wf9hN20c)
- [Stanford CS25: V3 I Low-level Embodied Intelligence w/ Foundation Models](https://www.youtube.com/watch?v=fz8wf9hN20c)
- [U of T Robotics Institute Seminar: Sergey Levine (UC Berkeley)](https://www.youtube.com/watch?v=EYLdC3a0NHw)
- [Princeton Robotics Seminar - Brian Ichter](https://www.youtube.com/watch?v=whn4eXlV0hk)
