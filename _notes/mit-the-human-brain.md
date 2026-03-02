---
layout: post
title: "MIT The Human Brain"
date: 2025-01-04
tags: neuroscience, cognitive-science
citation: false
toc:
  sidebar: left
---

[https://ocw.mit.edu/courses/9-13-the-human-brain-spring-2019/](https://ocw.mit.edu/courses/9-13-the-human-brain-spring-2019/)

# Lecture 1: Introduction to the Human Brain

The important question is not where certain mental processes are located in the brain, but which processes are and how they’re related. “Architecture of the mind”

Different ways to study the brain.

1. Behavioral
2. Anatomical
3. Functional brain activation

$$
E_k = \frac{RT}{zF}\ln{\frac{[K^+]_o}{[K^+]_i}}
$$

Back third of the brain does vision. Animals don’t do language.

# Lecture 2: Neuroanatomy

Four main components

1. Brain stem and cerebellum
2. Limbic system
3. White matter
4. Cortex

Can measure the receptive field of a neuron by measuring the neuron while moving a light source to see when it fires. Retinotopic maps show how neurons in the visual cortex respond to an image. Need attention to be fixated at a point. Similar maps can be constructed for touch (which part of the body is touched), audio (frequencies)

Cortical area: region of the cortex with a distinct function, connectivity to other areas, and sometimes cytoarchitecture (physical differences)

There are areas that respond to motion in a specific direction (ex: top right to bottom left). This is similar to a convolutional filter but with a temporal dimension. Is this optical flow? Different areas for moving objects and stationary objects.

There is a difference in the response of a neuron and its causal role. A neuron can be artificially stimulated to isolate its causality.

# Lecture 4: **Cognitive Neuroscience Methods I**

Ill-posed problems:

If you hear a new word, there are many possible meanings depending on the context. If you see an object, there are many different possible shapes that could have generated the image. There are many ill posed problems in perception and cognition.

Psycho-physics: showing people an input like an image or video and asking them what they perceive.

L (luminance, light hitting eyes) = I (Illuminence, light hitting object) \* R (Reflectance, color of object)

Perceived color depends on L but also I. We don’t actually sense I, but we can estimate it to solve the illposed problem of getting R from L. We don’t perceive L, we perceive an estimate of R.

People are good at recognizing familiar faces, but are worse at identifying unknown people. We build representations of a person with multiple views.

fMRI measures oxygen, which is a lagging indicator of neural activity. Good spacial resolution but poor temporal resolution. Can only used to compare neural activity, no absolute measure.

# Lecture 5: **Cognitive Neuroscience Methods II**

Face recognition is a separate pathway than object recognition. Prosapagnosia means a person is normal at object recognition but bad at face recognition. Opposite syndrome is the opposite.

ERG (Event related potential) is an EEG measure in response to a stimulis.

Invasive cranial measurements are the best. Can also stimulate certain electrodes and ask the subject about changes in their perception. This is powerful to establish causality.

Magnetic detection depends on the direction of the electric currents. Due to right hand rule, you can only detect the magnetism of currents that are perpendicular.

# Lecture 6: Experimental Design

Causality from visual input is easy to measure, but it is harder to measure causality from a neuron’s activity since you need to be able to control the activity.

Heisenberg principle of cognitive neuroscience, can’t temporal resolution and spatial resolution. fMRI is good spatial but bad temporal. Intracranial recording is the only method with both but is invasive and rare.

Transcranial magnetic stimulation (TMS) disrupts neural activity in a 1 cm region. The causal effects can be behaviorally measured. Can also stimulate at different time intervals to show when the region is causally engaged.

Confounds: variables that affect the output and can complicate inference

Minimal pairs: an experiment and a control that only differ in one isolated mental process

# **Lecture 7: Category Selectivity, Controversies, and Multiple Voxel Pattern Analysis (MVPA)**

In an fMRI experiment if you show stimuli in fast intervals, the responses are linearly mixed together. However, with enough trials you can extract the category specific responses.

Brains don’t perfectly align from person to person.

2x2 experiments allow testing two variables effect on activity

epiphenominal: information a region of a brain can contains information about something but not be used. This information is just a byproduct of the neural network. An analogy is a NN trained on faces can generate representations that discriminate between cats and dogs.

How can we tell if the information is used and not epiphenominal? We check whether the information is more correlated within the class than outside the class.

# **Lecture 8: Navigation I**

Monarch butterfly takes 4 generations to complete a migration. Precise navigation to a very specific place is encoded in the dna.

A species of ants can return to their home in a straight vector, no matter where they are located. Why can’t humans do this. It would not be a huge cost to encode this capability in the human brain if ants can do this so accurately. Maybe human navigation is designed more to use landmarks, as evidenced by the RSC region. Ants don’t have the same object recognition capability, so they just keep track of vectors.

Two fundamental questions

1. Where am i?
   1. recognize familiar location
   2. what kind of place, if unfamiliar
   3. layout of place
2. How to get to point b from point a
   1. beaconing: can see or hear destination
   2. where is b
   3. current heading with respect to mental map
   4. what routes are possible

Event related fMRI adaptation: Useful when the responses are the same between two classes of stimuli. In this case two stimuli are shown very quickly with a small gap. If the response is higher when the two stimuli are the same, it means that the neurons are processing information about the difference. Response is lower for stimulus the brain region thinks is the same because it “gets bored” or there is no new information to process in the internal representation space.

# **Lecture 9: Navigation II**

PPA and TOS involved in scene and spatial layout perception

RSC: getting current location and direction with respect to mental map

Place cells: in hippocampus, activate in particular locations. is it relative to some anchor point, if you travel a long distance will this anchor point be redefined and will the place cell fire again

dead reckoning: keeping track of vectors based on speed and direction and being able to determine the vector to get to a place.

head direction cells: brain’s compass

Grid cells: fire at the centers of a hexagonal grid, can be used to track distance. number of times fired is a measure of distance

border cells: activate at the border of an environment, such as walls, cliffs, etc.

Reorienting: when you arrive at a new location and have to place yourself in your cognitive map

Information encapsulation: brain regions only have access to certain types of information

# **Lecture 10: Development, Nature & Nurture I**

Empiricists: all knowledge comes from experience

Kant: There exists “a priori conditions” of cognition, innate knowledge or structure to the brain such as space and time

At birth: most neurons are present, most long range connections are present

During first 1-2 years of life: brain doubles in volume, cortex thickens, number of synapses increases, myelination increases

Two theories on learning face perception

1. Humans are innately wired to pay attention to certain things in order to learn. They may have a basic template for a face at birth which guides them to look at faces and finetune their understanding.
2. Humans are born with a nearly complete perception system, which is only lightly fine tuned

# **Lecture 11: Development, Nature & Nurture II**

Diffusion tractrography

Diffusion MRI finds direction of water diffusion at each point of the brain

Tractography processes these vectors to understand long range connectivity of different regions

Structural connectivity could explain why a functional region exists where it does

Rewired ferret [study](https://pubmed.ncbi.nlm.nih.gov/10786793/) suggests that the function of the region of the brain can be determined by its structural connectivity. Rewiring visual input to auditory cortex preserves the ferrets ability to see. Other regions of the brain learn to read visual information fro the auditory cortex.

Innate means determined at birth and does not require relevant experience, not present at birth. Puberty is innate even though it occurs later in life.

Controlled rearing: experiments where an animal is raised in different environments to control what kind experience they can get. This can test whether a behavior is learned or innate.

Kennard Principle: It is easier to recover from brain damage at earlier age, more plasticity when you are younger

Hebb Principle: Having brain damage early can sometimes be worse because it damages the foundation.

Many brain regions cannot be reimplemented in other parts of the brain following a lesion, even at birth.

Brain reorganization might be difficult because most regions of the brain are already tasked with something. There isn’t much generic computational matter that can be used to recover from a lesion. There is an exception with blind people, the visual cortex can be used for other tasks. It is used for language. FFA slightly activates when blind people hear sounds associated with faces. There some inherent multimodality in the brain. However, this is not observed in non-blind people.

# **Lecture 13: Number**

Number sense: higher level representation of number without counting

- Can represent high magnitude without verbal counting
- Approximate, discrimination depends on ratio not absolute difference
  - easier to discriminate between 1 and 2 than between 100 and 110
  - Weber’s law
- abstract, generalize across modalities
- can be used in mathematical operations

Weber fraction: minimum distinguishable ratio of quantities. This can be predictive of arithmetic ability.

Acalculia: inability to calculate

Are there ML datasets for counting? Prompt tuning for multitask ViTs

# **Lecture 15: Hearing and Speech**

Hearing is challenging because

1. invariance, same sound is different in different environments
2. multiple sound sources, diarization
   1. ill posed problem
3. Reverberation from environment
   1. sounds arrive at multiple times
   2. gives info on environment, we can hear echos
4. Speaker variability
   1. We are able to get acclimated to a person after hearing them talk for a short time
   2. Need to confound speaker, language, and speech

Human audition is different from animals. We have brain regions that are much more precise than general audition. For example, speech processing.

# **Lecture 16: Music**

Music is unique and universal to humans.

Why is music so important to humans?

Data driven experiments: get a lot of data and try to find arbitrary patterns. This is different than doing a study to just prove a pre selected hypothesis.

Direct contrast doesn’t discriminate on music. However, if you do ICA (similar to PCA but without orthogonality constraint), one component is strongly music selective. This can then be mapped to a region in the brain. This suggests that the part of the brain that recognizes music overlaps with other functions.

# **Lecture 18: Language I**

Some kinds of thinking doesn’t require language, for example approximate number system

How connected is language to thought?

Global aphasics: people who have lost language ability

Earlier lecture mentions an experiment where humans can’t reorient based on non spatial clues when you tie up the language system. However, aphasics have this ability. One theory is that language is used to learn the ability to reorient when the aphasics were younger and had language ability. However, that doesn’t explain why people with their language tied up can’t do this. Maybe aphasics learned a different mental mechanism to reorient.

Language influences thought. There is an aboriginal language where they use compass directions instead of “left” or “right”. This makes them keep track of compass direction at all times.

# **Lecture 20: Theory of Mind & Mentalizing**

Mentalizing: inferring the mental state of other agents

rTBJ selectively activates when thinking about another persons thoughts

# **Lecture 21: Brain Networks**

White matter makes up a larger proportion of human brains compared to animals.

Connectivity fingerprint can identify cortical regions across animals

Connectivity plays a causal role in development

Diffusion imaging helps find big fiber bundles by measuring water diffusion

Fractional anistropy can be a measure of the strength of a connection

FA can be affected by connections going in a perpendicular direction, and it is prone to artifacts

Tractography: Following vectors from diffusion imaging to reconstruct structural connections

Can’t distinguish directions at intersections

Resting functional MRI correlations: two areas that are physically distant but highly correlated are likely connected

Done at rest, why? Some regions are more active at rest.

# **Lecture 24: Attention and Awareness**

Covert attention: peripheral vision

Overt attention

Automatic attention: stimulus driven

Voluntary attention

[Assignments](https://www.notion.so/Assignments-e07c45cfd87948e2ae34f99d320edf63?pvs=21)

[Reading](https://www.notion.so/Reading-cc7552ea37494478bd1d240a8839b493?pvs=21)

Perceptual narrowing → contrastive learning

We learn that two phonemes mean the same thing. Eventually our mental representations became invariant to their differences.
