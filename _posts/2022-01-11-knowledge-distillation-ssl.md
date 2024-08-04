---
layout: post
title: Knowledge Distillation as Self-Supervised Learning
tags: paper-review self-supervised-learning knowledge-distillation computer-vision
thumbnail: assets/img/blog/distillation_ssl/seed2.png
toc:
  sidebar: left
---

Self-supervised learning (SSL) methods have been shown to effectively train large neural networks with unlabeled data. These networks can produce useful image representations that can exceed the performance of supervised pretraining on downstream tasks. However, SSL is not effective with smaller models. This limits applications where computational power is limited, such as edge devices. Knowledge distillation (KD) is a popular method to train a smaller student network from a larger and more powerful teacher network. The [SEED](https://arxiv.org/abs/2101.04731) paper by Fang et al., published in ICLR 2021, applies knowledge distillation to self-supervised learning to pretrain smaller neural networks without supervision. In this post, we will discuss self-supervised learning and knowledge distillation and how they are unified in SEED.

# Self-supervised Learning

Self-supervised learning is a form of unsupervised learning. Self-supervision refers to labels that are generated from the data itself rather than manual annotations (ex: images vs class labels). Different SSL methods have different tasks that are used for the self-supervision.

In computer vision, it is very common to pretrain a neural network on ImageNet classification. This is an example of supervised pretraining. This network can then be fine-tuned for various downstream tasks, such as semantic segmentation, object detection, or even medical image classification. Supervised pretraining has been a standard practice in computer vision.

{% include figure.liquid loading="eager" path="assets/img/blog/distillation_ssl/sl_vs_ssl.png" description="Self-supervised vs Supervised Pretraining" width=400 %}

Self-supervised learning provides an alternative to supervised pretraining with two main benefits:

1. Generalizability: A supervised objective like classification can limit what a model learns about data. This is because not all the information in an image is needed for classification. For example, you can train a network to classify cats and dogs. The color of the animal's fur is not relevant to this objective. Therefore, representations from this network may not be useful for a downstream task of fur color classification.

2. Unlabeled data: The amount of available unlabeled data dwarfs labeled datasets. SSL is a form of unsupervised learning. It can leverage datasets of billions of images rather than be limited to supervised datasets, such as ImageNet which has about one million images.

There are many methods of SSL. Most of the recent state of art methods implement a form of contrastive learning. This includes [SimCLR](https://arxiv.org/abs/2002.05709), [SwAV](https://arxiv.org/abs/2006.09882), and [MoCo](https://arxiv.org/abs/1911.05722). In contrastive learning, representations are pushed towards positive examples and away from negative examples. In SSL, the positive examples are variations of the original image and the negative examples are from other images in the dataset. Contrastive SSL methods share some common steps:

1. Image augmentation: In supervised learning, augmentations, such as random cropping, flipping, and color distortions are used to generate more training data. In SSL, augmentation is used to produce positive examples. It is needed to avoid the trivial solution of encoding raw pixel values without learning anything about the content of the image.

2. Contrastive loss: The goal of contrastive learning is to push positive examples closer together and negatives apart. It is most common to see a version of the [InfoNCE](https://arxiv.org/abs/1807.03748) loss. This loss (defined below) is meant to maximize the similarity of a data point with one positive example, and minimize the similarity with many negative examples. The similarity function $$s$$ is usually just the dot product.

3. Negative samples: SSL needs a large amount of negative examples for the best performance. We want to push an image representation away from all other possible image representations from the dataset. This can be accomplished by having a large batch size (SimCLR). All the other images in the batch will be negative examples. An alternative is to keep negative examples in memory through multiple training batches. MoCo does this by keeping a queue of the most recent image representations. It is preferred to keep recent image representations, since the network changes gradually over time. Recent representations are more similar to representations that would be generated from the current network. The queue essentially approximates a large training batch.

$$
\begin{equation}
\mathcal{L}_{\mathrm{InfoNCE}} = -\mathbb{E}  \left[
\mathrm{log} \frac{ \exp(s(x, y)) } { \sum_{y_j} \exp(s(x,y_j)) }
\right]
\end{equation}
$$

## MoCo

[MoCo](https://arxiv.org/abs/1911.05722) (momentum contrast) by He et al. implements contrastive SSL by keeping a queue of examples. The queue allows for a large number of negative examples to be used in the contrastive loss The momentum encoder is trained at the same time as the encoder in a bootstrapped fashion. They must have identical architectures for the momentum update to occur.

{% include figure.liquid loading="eager" path="assets/img/blog/distillation_ssl/moco.png" description="MoCo training" width=700 %}

With a queue of representations encoded by the momentum encoder, the main encoder is trained to contrast the representations. $$q$$ is the query or the representation from the encoder. $$k_+$$ is the corresponding representation from the momentum encoder. The loss aims to push $$q$$ towards $$k_+$$ and away from all other representations $$k$$ in the queue which serve as negative examples.

$$
\begin{equation}
\mathcal{L}_i = -\log\frac{\exp(q*k_+/\tau)}{\sum_{i=0}^K\exp(q*k_i/\tau)}\\
\end{equation}
$$

MoCo is very effective in pretraining large neural networks for many downstream tasks. SEED aims to extend this for smaller networks.

# Knowledge Distillation

In knowledge distillation ([Hinton et al.](https://arxiv.org/abs/1503.02531), [Buciluǎ et al.](https://www.cs.cornell.edu/~caruana/compression.kdd06.pdf)), a large teacher model is used to train a smaller and more efficient student model. It is useful in the case that a large neural network can perform well on a task, but a small network cannot be directly trained to high accuracy. This makes it relevant to SSL, where only large neural networks have strong performance.

In supervised learning for classification, the labels are hard targets or one-hot encoded vectors. All of the probability is assigned to one class, and all other classes have a value of zero. The teacher model will have a softmax layer which will return a soft target. The soft target will assign some probability to other classes. Knowledge distillation uses the teacher network to produce these soft targets and uses them to train the student model.

{% include figure.liquid loading="eager" path="assets/img/blog/distillation_ssl/soft_vs_hard.png" description="Shiba Inu dogs are known to have cat-like characteristics, soft labels can encode this by assigning some probability to the cat class." width=600 %}

The soft targets encode more information than hard targets. Hinton describes this as "dark knowledge". For example, from a soft target you can tell which class is the second most likely or the relative probabilities between two classes. This information is not available in a hard target.

$$
\begin{equation}
p_i = \frac{\exp(\frac{z_i}{T})}{ \sum_{j} \exp(\frac{z_j}{T})}\\
\end{equation}
$$

The soft targets can be made softer by increasing the temperature of the softmax. The temperature $$T$$ is typically set to 1. However, in knowledge distillation higher temperatures can yield better results as it increases the magnitude of the non-max values.

{% include figure.liquid loading="eager" path="assets/img/blog/distillation_ssl/kd.png" description="Knowledge Distillation" width=700 %}

1. A teacher model is trained for high accuracy. This can be a large neural network or an ensemble.

2. The teacher model generates soft labels for a dataset. This dataset can be the same or different from the hard labeled dataset.

3. The student network is trained to predict the soft labels. It can also be simultaneously trained with hard labels in a separate loss term.

Distillation can use unlabeled data. Once a model is trained, it can be used to produce soft labels for a large unsupervised dataset. This can be larger than the initial labeled dataset and effectively train the student network on a much larger dataset.

# Knowledge Distillation for SSL

Knowledge distillation aims to transfer dark knowledge between models. Self-supervised learning aims to increase the dark knowledge learned by a model. When training a model on a supervised classification objective, it will not need to learn information that does not help with classification. The objective limits what the model learns. Self-supervised learning methods are designed to be general and not task specific.

SSL does not perform well with smaller models which limits its applicability. Also, the downstream task is likely less complex than the SSL task and can be achieved more efficiently with a smaller model. Knowledge distillation offers a way to reduce the size of the model while maintaining accuracy and relevant knowledge.

One way to apply KD to SSL is to train the teacher on a SSL objective and then apply KD to train the student on a downstream task. This would require first fine-tuning the teacher network on the downstream test, with a new output layer. It would be more efficient to distill knowledge to a smaller network before training on the downstream task.

Although it is simple to apply knowledge distillation on a supervised downstream task, you cannot directly apply it to the self-supervised training objective. This is because SSL models do not output classification predictions. SSL models output feature representations of the input. Training a student network to match these feature representations would not be effective. Self-supervised training involves optimizing with an objective on top of the representations.

# SEED

In the [SEED](https://arxiv.org/abs/2101.04731) paper, the authors propose a self-supervised approach to knowledge distillation. It uses a contrastive objective on the representations.

This will allow knowledge distillation to occur before the downstream task. The method produces an SSL trained student network that can be efficiently fine-tuned on downstream tasks. SEED extends self-supervision to smaller models allowing us to compress SSL models to use in more applications.

## Method

{% include figure.liquid loading="eager" path="assets/img/blog/distillation_ssl/seed.png" description="SEED" source="https://arxiv.org/abs/2101.04731" %}

1. Train the teacher, independent of the student network. Any of the recent state-of-the-art SSL methods or even supervised models (ResNet trained on ImageNet classification) can be used here. The only requirement is that the model must produce image representations. The teacher networks weights are then frozen.

2. Apply an augmentation to the input image. The same augmentation of the image is used for both the student and the teacher networks. In most other SSL methods, different augmentations would be used. SEED reports better performance when using the same augmentation. This may be because trivial solutions are avoided by the pretraining of the teacher network.

3. Input the image to both the student and teacher networks to get two vector representations: $$Z^S$$ and $$Z^T$$.

4. Add teacher vector $$Z^T$$ to the instance queue $$D$$ which is a fixed size FIFO queue that persists between training batches. Self-supervised learning in general benefits from a large number of negative examples.

5. Apply the self-supervised SEED loss, using the student and teacher vectors, and the instance queue. The student and teacher vectors are each compared to every embedding in the queue, to produce two similarity vectors. A cross-entropy loss is applied between the similarity vectors The student network is trained to produce vectors that have the same similarities as the teacher. We will further explain the loss used in SEED.

## Loss

In self-supervised learning, a supervised objective is formed from the input rather than human annotations. In this case, the supervised objective is predicting the current image representation from a queue containing the current representation and negative examples. Knowledge distillation is applied with respect to this objective. The scores from applying the softmax to the teacher similarity vector form the soft label.

The cross-entropy loss is used like the contrastive InfoNCE loss in SSL. The student vector is pushed towards the teacher vector and away from the vectors in the queue. However, some of the negative vectors are closer than others. The student network is also trained to match this information. This is where the dark knowledge of KD is applied.

Unlike the InfoNCE loss, there are no hard positive and negative examples in this objective. The teacher network creates a soft probability distribution. Each example is assigned a continuous score between 0 and 1 that indicates how positive the example is. SEED can be viewed as a _soft contrastive learning_ method.

$$
\begin{align}
\mathcal{L}_{SEED} &= - \sum_i^N \textbf{p}^T(\textbf{x}_i; \theta_T, \textbf{D}^+) * \log \textbf{p}^S(\textbf{x}_i; \theta_S, \textbf{D}^+) \\
&= - \sum_i^N \sum_j^{K + 1} \frac{\exp(\textbf{z}_i^T * \textbf{d}_j / \tau^T)}{\sum_{d\sim\textbf{D}^+}\exp(\textbf{z}_i^T * \textbf{d} / \tau^T)} *
\log \frac{\exp(\textbf{z}_i^S * \textbf{d}_j / \tau^S)}{\sum_{d\sim\textbf{D}^+}\exp(\textbf{z}_i^S * \textbf{d} / \tau^S)}
\end{align}
$$

For each example in the batch (size $$N$$), two similarity functions are applied: one using the teacher network $$p^T$$ and one using the student network $$p^S$$. The similarity function is applying an inner product and softmax with the vectors to the instance queues. This produces a probability distribution with more probability on examples in the queue that are close to the input. Since we want these probability distributions to match, a cross entropy loss is applied between the two probability distributions.

$$
\mathcal{L}_{cross-entropy} = \sum_{i} y_i * \log(\hat{y}_i)
$$

Referring to the formula for cross-entropy. $$\textbf{p}^T(..)$$ corresponds to the label $$y_i$$. In classification, $$y_i$$ would be binary or a one-hot encoded vector. In this case, $$\textbf{p}^T(..)$$ is a score between 0 and 1. As in KD, this is a soft label. With hard labels and standard contrastive learning, the scores would be binary with a 1 for the current datapoint. $$\textbf{p}^S(..)$$ corresponds to the prediction $$\hat{y}_i$$. Here the prediction is the student similarity score. We want the student to produce similarity scores matching the teacher.

## SEED vs MoCo

SEED is trained very similarly to MoCo. The differences are the lack of momentum weight updates and the soft contrastive loss.

{% include figure.liquid loading="eager" path="assets/img/blog/distillation_ssl/seed2.png" description="SEED training" width=700%}

# Self-supervised vs Supervised Knowledge Distillation

SEED or self-supervised distillation in general does not aim to replace supervised knowledge distillation. The authors report their best results when training models with both self-supervised and supervised knowledge distillation.

{% include figure.liquid loading="eager" path="assets/img/blog/distillation_ssl/s_vs_sl_kd.png" description="Self-supervised KD with Supervised KD" width=900%}

SEED allows for more effective self-supervised training of smaller models. It is better to train a large model with SSL and distill it to a small model than to train the small model directly. After SEED pretraining, the model can be fine-tuned with supervised knowledge distillation with the downstream task. In this step, the student is initialized from the self-supervised KD trained model, instead of initializing from scratch.

# Conclusion

Self-supervised knowledge distillation allows the impressive gains of large SSL models to be transferred to smaller neural networks. This allows for more applications of these models. We can even view knowledge distillation as a form of self-supervised learning. Hard labels are not used in SSL. The soft labels provide self-supervision since they are produced from the data.

SEED essentially adapts momentum contrast to be used as knowledge distillation. An interesting future direction would be adapting other SSL methods such as SimCLR to be used as knowledge distillation. Nearly every contrastive SSL method can be adapted in this way.
