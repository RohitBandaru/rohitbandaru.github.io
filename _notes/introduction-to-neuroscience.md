---
layout: post
title: "Introduction to Neuroscience"
description: "Notes from Bing Wen Brunton's introductory neuroscience course, covering the neuron doctrine, resting potential, synapses, and synaptic transmission."
date: 2025-01-04
tags: neuroscience
citation: false
toc:
  sidebar: left
---

Course by Bing Wen Brunton (University of Washington) [https://www.youtube.com/playlist?list=PLqgZEQsU_8E0l1P9bKR6yKOKPMpoJ_tLR](https://www.youtube.com/playlist?list=PLqgZEQsU_8E0l1P9bKR6yKOKPMpoJ_tLR)

## Neuron Doctrine (2)

The neuron doctrine: Neurons are distinct cells that communicate through synapses and don’t fuse together.

## Resting Potential (3)

Cells have a negative resting potential. Which means the inside of a cell is negatively charged with respect to the outside. It is -60 to -80 mV. The cell has less Na+ and Cl- than the outside but more K+.

Nernst potential is the potential when K+ is at equilibrium between the inside of a cell and the outside. The equilibrium is the balance between the electric gradient and the concentration gradient.

Nernst Equation for K

$$
E_k = \frac{RT}{zF}\ln{\frac{[K^+]_o}{[K^+]_i}}
$$

z is the valence of the ion (+1 for K+)

R is universal gas constant 8.3 J/mol

F is Faradays constant 9.6e4 C/mol

Potential is measured in joules per coulomb. 1 V to 1 J / 1 C

RT/F is about 25 mV

The ratio of ions is the main variable

The sodium-potassium pump maintains the concentration gradients of more K and less Na in the cell. This pump is always running and consuming energy to maintain the neuron’s resting potential.

1 ATP → 3 Na+ out, 2 K+ in

25mV \* ln(1/5) = -40mV from -75 mV

The Goldman Hodgkin-Katz (GHK) Equation

![Screenshot 2023-05-07 at 9.07.08 PM.png](/assets/img/notes/introduction-to-neuroscience/ghk-equation.png)

Nernst equation with concentration ratios of different ions weighted by the permeability to the ion. Formula can be changed to account for higher valency ions.

Conductance is 1/R and depends on the number of ion channels.

$$
V=IR\\I = V/R = Vg\\I_K = (V_M-E_K)g_K
$$

I_K is the flow of ions, VM is membrane voltage, E_K is equilibrium

This is equivalent to an RC circuit

The membrane can be considered as a capacitor, with the inside of the cell having positive charge.

![Screenshot 2023-05-07 at 9.14.39 PM.png](/assets/img/notes/introduction-to-neuroscience/neuron-rc-circuit.png)

Separate channels for each of the three ions, so we can have independent equations for the conductances of each ion.

The channels don’t have directionality, in contrast to pumps.

## Active Properties of Neurons (4,5)

![Screenshot 2023-05-08 at 8.07.45 AM.png](/assets/img/notes/introduction-to-neuroscience/active-neuron-properties.png)

$$
V_m = \lambda^2\frac{d^2V_M}{dx^2}-\tau_m\frac{dV_M}{dt}
$$

lamda squared in the length constant of propagation $$R_m/R_i$$. R_i is the intracellular resistance

$$
R_m=\frac{r_m}{2\pi a}\\R_i =\frac{r_i}{\pi a^2}\\\lambda^2 = \frac{r_m a}{r_i 2}
$$

a is the radius of the neuron represented as a cylinder.

![Screenshot 2023-05-14 at 9.04.52 AM.png](/assets/img/notes/introduction-to-neuroscience/ion-channel-potentials.png)

K wants to make the cell membrane more negative, Na wants to make it more positive. This is because the concentration gradients are reversed (there is more K inside and more Na outside).

Voltage gated ion channel: channel is open only when the cell is depolarized. Includes an activation gate and an inactivation gate. Separate channels for sodium and potassium

1. Rest: $$g_K > 0$$; $$g_{Na} = 0$$; small inward $$I_K$$
2. Rising: $$g_K > 0$$; $$g_{Na} >> 0$$; small outward $$I_K$$; large inward $$I_{Na}$$; voltage gated ion channels open
3. Overshoot: $$g_K > 0$$; $$g_{Na} >>> 0$$; small outward $$I_K$$; small inward $$I_{Na}$$; Concentration gradient of Na decreases
4. Falling: $$g_K >> 0$$; $$g_{Na} > 0$$; large outward $$I_K$$; small inward $$I_{Na}$$; Sodium channels deactivating, potassium channels activate causing hyperpolarization
5. After-hyperpolarization: $$g_K >> 0$$; $$g_{Na} > 0$$; small outward $$I_K$$; no $$I_{Na}$$; voltage gated channels are closed

Action potential is binary, it is the same regardless of the size of the stimulus. It is caused by transient changes in conductances. Relatively few ions cross the membrane, the ion concentrations don’t change so the Nernst potentials $$E_k$$ $$E_{Na}$$ don’t change.

Hodgkins-Huxley equations: 4 ODEs that explain the action potential

Action potential moves in space through the axon. The depolarization of a part of the axon, triggers the next part to start. Positive feedback loop causes the propagation.

![Screenshot 2023-05-14 at 8.09.12 PM.png](/assets/img/notes/introduction-to-neuroscience/action-potential-propagation.png)

Wider axons cause faster propagation. Another solution is myelination. Glial cells are cells that aid neurons. Myelin increases the length constants.

Node of ranvier: gaps between myelin with a lot of voltage gated ion channels. APs go through these nodes, and are propagated through the myelin instead of the axon.

Myelination is costly and only used when necessary.

# Anatomy of Synapses (6)

dendrite: receive signals from synapses of other neurons and propagate to the cell body

soma: neuron cell body

axon hillock: triggers action potentials

presynaptic neuron sends AP to postsynaptic neuron. This terminology is with respect to a particular synapse

vesicles: transmit electrical activity in a chemical synapse

synaptic cleft: the gap between neurons in a synapse

# Chemical Synaptic Transmission (7)

1. AP arrives and presynaptic cell is depolarized
2. Voltage gated Calcium channels open. Calcium is used for signaling because there is a lot outside the cell, but little inside
3. Calcium triggers exocytosis
4. Vesicles fuse with the membrane open at the cleft, release neurotransmitters
   1. SNARE proteins facilitate the fusion. These proteins are enabled by calcium receptors
   2. Calcium increases the probability of fusion, but the fusion is probabilistic Poisson process
5. NTs bing to the postsynaptic ionotropic and metabotropic receptors
6. Downstream effects, ions entry, second messenger cascades, changes in gene expression
7. Transporters reuptake NTs to terminate synaptic activity, this can also happen in the postsynaptic cell or by glial cells
8. Transmitters are recycled back into vesicles

Common NTs

- Monoamine: derived from amino acids
  - Acetylcholine (ACh): controls muscle movement
  - GABA: inhibits neurons in brain
  - Glutamate: excites neurons in the brain
- Serotonin: involved in depression
- Dopamine: reward / addiction

## Postsynaptic Receptors (8)

- Ionotropic receptor
  - Transmitter-gated ion channels
  - Glutamate receptors; both depolarize the cell and are permeable to Na and K
    - AMPA
    - NMDA
      - Activates only if the cell is already depolarized
      - permeable to Ca
      - activated when there are multiple APs in quick succession; the effects get more significant as more NMDA receptors are active
      - A can only activate C after B depolarized C, this is a logic gate. B can add AMPA receptors through A through calcium influx. This is an example of synaptic plasticity and associative learning.
        ![Screenshot 2023-05-21 at 11.40.57 AM.png](/assets/img/notes/introduction-to-neuroscience/synaptic-plasticity-logic-gate.png)
    - GABA receptors / Glycine receptors
      - Cl channels
      - different blockers and agonists (activators)
    - ACh
      - blocker: curare
- Metabotropic receptor
  - G-protein coupled receptor (GPCR)
    - NT binds to GPCR then activates or deactivates an ion channel
    - humans have 1000 types
  - G-protein coupled effector system
    - NT binds to GPCR, G-protein activates enzymes that cause a cascade
    - May trigger membrane potential or gene expression
  # Types of Synapses (9)
  Electrical synapses
  Held together by gap junction channels
  Much faster than chemical synapses, but less control/change of the properties. This makes it weaker for learning.
  Axodendendritic: axon to dendrite synapse
  Axosomatic: axon to cell body
  Axoaxonal: axon to axon
  chlorine inhibits, sodium excites
  APs are triggered by the sum of the potentials (positive and negative) in all synapses
  Mixed chemical electrical synapses
  Glial cells can influence synapses with NTs
  Heterosynaptic interaction is where two neurons form different types of synapses to the same postsynapse
  Co-release: vesicles have mixes of NTs
  Co-transmision: different vesicles have different NTs
  # Recreation Drugs (10)
  Drugs abuse the pleasure-reward system.
  medial forebrain bundle (MFB): reward
  ventral tegmental area (VTA): unexpected reward
  ### Dopamine System
  cocaine blocks dopamine reuptake, causing it to be active for longer
  amphetamine: false substrate that takes space in vesicles, blocking dopamine reuptake
  ### Serotonin System
  LSD: agonist of serotonin receptors, activates these receptors to mimic the effects of serotonin
  PCP: Antagonist of NMDA glutamate receptors
  MDMA: Antagonist, false substrate
  ### Other NTs
  Caffeine: antagonist of A1 receptors
  THC: Agonist of CB1 receptors
  # Addiction and Learning (11)
  Natural learning: RL, learning with rewards for good things and penalties (pain) for bad things
  Drugs can hack the rewards
  Naloxone: competitive antagonist for opioid receptors, can prevent overdose
  # Sensing and Action (12)
  Loop between actions and sensory perceptions
  # Coordinates (13)
  ![Screenshot 2023-05-28 at 7.49.11 AM.png](/assets/img/notes/introduction-to-neuroscience/brain-coordinates-sagittal.png)
  ![Screenshot 2023-05-28 at 7.51.22 AM.png](/assets/img/notes/introduction-to-neuroscience/brain-coordinates-bipedal.png)
  Coordinates are rotated for the brain because humans are bipedal.
  Medial - Lateral axis, medial is closer to the line of symmetry or spine
  Coronal section: front to back vertical slicing
  Horizontal section: parallel to gravity slicing
  Sagittal section: left to right vertical
  # **Electrical Methods in Neuroscience (14)**
  **Intracellular recording**
  Cell recordings are done by attaching a pipette to the membrane of the cell. This can attach to the membrane in cell-attached recording, or break it in whole-cell recording. These methods require the pipette to contain a solution with similar composition as the inside of the cell. Applying suction to a single ion channel is called patch clamping. These techniques are all very difficult since they require a lot of precision and are prone to error.
  **Extracellular Recordings**
  Metal tip on a probe can detect action potentials of nearby neurons. This only measures the induced electrical field so it is a noisier signal. However this allows measuring multiple neurons.
  **Multi-electrode Recordings**
  Can use an array of electrodes to measure many more neurons. It can also make it easier to localize different neurons by seeing the differences in amplitude between different electrodes.
  Utah arrays: Grid of about 10x10 electrodes
  Neuropixels probe: thousands of recording sites in a small thread. It can measure thousands of cells and creates large amounts of data.
  **Spike Sorting**
  Signal processing is important to separate different action potentials.
  **Raster Plots**
  Used to visualize spikes in a group of neurons over time
  # Optical **Methods in Neuroscience (15)**
  **Calcium Imaging**
  Calcium concentration is a proxy for electrical depolarization. Can use fluorescent calcium sensitive dyes, but it is hard to inject dyes. Calcium sensitive proteins can be used instead.
  GCaMP: protein where GFP (green fluorescent protein) is activated with four calcium ions
  GCaMP reporting requires measuring change in fluorescence not absolute fluorescence. This requires either modifying the DNA of the animal which is difficult, or viral biodelivery.
  **Optogenetics**
  Channelrhodopsin is a light activated ion channel. Flashing a blue light caused an AP.
  Haklorhodopsin is a light sensitive chloride pump. Yellow light reduces the magnitude of APs.
  Can add these proteins to only specific types of neurons. This is added specificity not available with electrical stimulation. Can also use precise lasers with spatial and temporal precision to isolate the activation.
  Optical methods are less biased to more active neurons.
  # **Non-invasive Methods in Neuroscience, fMRI, MEG, EEG (16)**
  MRI:
  fMRI: Blood Oxygenation Level-Dependent (BOLD) effect. Oxygenated hemoglobin has a different magnetic effect that can be measured
  Relation between BOLD and neural activity is unclear
  Restriction in experiments because the subjects have to lay still in the fMRI tube
  EEG: attaches electrodes to head
  MEG can sit outside tube with magnetic measurement
  # **Visual perception, organization of the retina, photoreceptors and rhodopsin (17)**
  Retina → Thalamus / LGN → cortex V1 → V2 → V3/V4
  ![Screenshot 2024-01-14 at 11.08.12 AM.png](/assets/img/notes/introduction-to-neuroscience/visual-pathway-retina-to-cortex.png)
  Cortex projects back to LGN more than the inverse.
  Retina is at the back of they and gets excited by light and then transfers information through the optic nerve.
  ![Screenshot 2024-01-14 at 11.19.01 AM.png](/assets/img/notes/introduction-to-neuroscience/retina-structure.png)
  Photoreceptor cells come in two types: rods and cones. Rods are not color sensitive but are very fast and sensitive to low light. Cones are slower but color receptive. Octopus doesn’t have a blind spot because the photoreceptors are facing the lens and the ganglion cells are in deeper layers.
  There is a blind spot in each eye where the optic nerve connects.
  Light receptors are packed with photosensitive rhodopsins (GPCR).
  Retinal: light sensitive pigment that absorbs energy from photons. It is bound to rhodopsin which propagates also has a conformational change.
  Photoreceptors have graded potentials instead of spikes. They hyperpolarize when activated. They are normally depolarized.
  Our vision works in a wide range of light (luminance) conditions. We can adjust by pupil dilation (10x) switching to rod based vision (1000x), adjusting retinal levels (100x). It takes 30 mins to fully adjust.
  There are 120 million rods and 6 million cones. Cones in the center of the retina.
  ![Screenshot 2024-01-14 at 11.47.17 AM.png](/assets/img/notes/introduction-to-neuroscience/rod-cone-distribution.png)
  # **Color vision, Rod and cone cells in the retina (18)**
  Rhodopsin comes in different variations to shift the absorption spectrum of the retinal to different colors in the visible spectrum. Retinal by itself peaks with UV light. Rods have only one type of rhodopsin so it is not color sensitive.
  Different colors of light activate the receptors to different levels. Color can be discerned by comparing the relative activations of receptors. We have three types of photoreceptors.
  The eye can only see color in a small spot where the cones are. The brain fills in the rest with color.
  Some humans have more than 3 photoreceptors: [https://en.wikipedia.org/wiki/Tetrachromacy](https://en.wikipedia.org/wiki/Tetrachromacy)
  # **Visual computations and circuits, Receptive fields, the Sparse coding hypothesis (19)**
  ![Screenshot 2024-01-14 at 4.38.02 PM.png](/assets/img/notes/introduction-to-neuroscience/receptive-fields.png)
  Retinal Ganglion Cells RGC
  on-center RGC activate when there is light in the middle and no light on the outside. It is like a convolution kernel. off-center RGC is most activate for the inverse (dark in the middle).
  Horizontal cells inhibit neighbors to increase contrast.
  LGN seem to do spatiotemporal convolutions. This means it responds to particular light patterns.
  V1 neurons show directional selectivity. It performs pooling on LGN cells. V1 cells are organized by the orientation of their sensitivity
  Sparse coding hypothesis: images can be decomposed into a sum of sparse basis functions. This would minimize the number of neurons activated to represent an image.
  The higher you go through the visual pathway the more complex the receptive fields.
  # **Why is everything everywhere (in the brain)? (20)**
  Function is not separated in the nervous system. V1 cells are used in other functions such as action.
  # **The chemical senses: taste and smell (21)**
  Gustatory receptors: taste; water soluble molecules
  Olfactory receptors: smell, volatile airborne molecules
  # **Taste sense, Gustatory receptors (22)**
  Taste buds contains TRCs (taste receptor cell). Gustatory nerve propagates the signal.
  ATP is used as a neurotransmitter.
  Many TRCs are GPCRs.
  umami, sweet, bitter, sour, sodium
  Spicy: capsaicin binds to TRPV1 receptors which is not a TRC. These channels are also activated by hot temperatures.
  Taste and smells elicit powerful memories and emotions. Different parts of the gustatory cortex project to different parts of the amygdala.
  # **Smell sense, odorant receptors neurons (23)**
  Olfactory bulb: region of brain
  Humans don’t have a significantly weaker sense of smell compared to animals.
  Dogs have nose shapes that allow for better sniffing. When dogs exhale they don’t disrupt the object they are trying to smell.
  Animals smell in stereo. Humans don’t have smell directionality. They have smell organs that are offset like eyes. This allows for stereo smell.
  Each olfactory receptor neuron expresses a single odor receptor and projects to the olfactory bulb. The olfactory bulb contains three glomerulus. Each of these are connected to different neurons.
  Odorant receptor proteins are GPCRs. Each type of olfactory receptor cell responds to different chemical compounds to different degrees. This creates a multidimensional representation of a smell. This is referred to as a population code, since a single neuron doesn’t identify a single specific smell.
  Is this a result of chemistry, in that a receptor protein is not specific to a single smell.
  The olfactory bulb directly connects to the piriform cortex. The mitral cells of the glomerulus does this projection.
  Lateral inhibitory circuits in the olfactory bulb sharpen our sense of smell. If one glomerulus is activated, its neighbors are inhibited.
  # **Somatic Senses (24)**
  **Touch**
  Somatic senses are hard to study because they are coupled with motor systems so require movement.
  ![Screenshot 2024-02-04 at 9.45.17 AM.png](/assets/img/notes/introduction-to-neuroscience/touch-dermatomes.png)
  The dorsal root ganglion (DRG) in the spine connects to sense receptors. Each segment is responsible for a different area of skin.
  DRG to muscle spindle has different neurons with different levels of myelination.
  Different tactile neurons have different receptive field locations, field sizes, sensitivity, and timescales of adaptation. Neurons stop firing after you touch something for a certain amount of time.
  Embodied computation: firing properties are a factory of biomechanics of the cell and tissue (stiffness of the skin) and mechanotransduction channels.
  Different receptors have different adaptation / firing patterns to stimulus. Some respond to change, others respond constantly.
  **Proprioception**
  Sense of location of body.
  Muscle spindles sense muscle length and rate of change.
  Golgi tendon organs sense muscle force and its rate of change.
  Joint receptors sense extreme joint positions.
  Mechanotransduction relies of the Piezo protein and the TRP channels.
  **Temperature/pain/itch**
  Transient receptor potential (TRP) channels.
  These receptors are also involved in taste, which is why mint tastes cold and spice tastes hot.
  Pain and itch have distinct pathways. These pathways are connected. They can inhibit each other.
  # **Somatosensory Systems (25)**
  Dorsal column / lemniscal pathway: touch and proprioception
  Anterolateral column / spinothalamic pathway: pain and temperature
  The spine itself has mechanosensation, meaning it has some sensory neurons.
  S1: primary somatosensory cortex
  Sensory Homunculus: diagram of human where the body regions are scaled with respect to their size in S1.
  M1: primary motor cortex. There is also a motor homunculus
  Phantom Limb
  After amputation, missing nerve endings get remapped to other parts of the body. The region of S1 that does to the arm will be remapped to the face since it is next to it in S1.
  Rubber Hand Illusion
  Humans can learn to use proprioception on external entities. Scientists can manipulate a subject to think a rubber hand is their actual hand.
  # **Common concepts in sensory neurobiology (26)**
  Neurons can be described by their receptive fields (spatial/temporal)
  # **Movement and Motor Control (27)**
  Outputs of the brain: motor system, autonomic nervous system (internal organs), neuroendocrine system (hormones, digestion)
  ![Screenshot 2024-02-10 at 12.32.29 PM.png](/assets/img/notes/introduction-to-neuroscience/motor-control-system.png)
  Motor control system projects down, whereas sensory systems project up to the brain. It is an output system.
  1. Action potential travels down motor neuron
  2. Acetylcholine is released at neuromuscular junction
  3. Acetylcholine activates Na channels in muscle to depolarize
  4. Depolarization of T-tubule triggers release of Ca from sarcoplasmic reticulum into cytocol
  5. Elevated Ca causes muscle contraction
     There are many synapses at the neuromuscular junction to increase reliability.
     Old experiments show that the brain is needed to initiate movement, but not for maintaining it. A cat can walk with a severed spinal cord once initiated on a treadmill.
     Central Pattern Generator (CPG) can create walking through neural oscillators. Each leg is at a different phase of oscillation with mutual inhibition. There is also sensory feedback to react to environmental changes.
     Skipping gait is energetically favorable for kids because they are smaller. This is why adults choose not to skip. However, skipping is favorable in low gravity on the moon.
  # **Movement Initiation(28)**
  Cerebellum is required for fine control of movement. Cerebellum defects lead to slower and uncoordinated movement, and delay in initiation.
  Purkinje cell has a large number of synapses with parallel fibers. There is another cell that follows the purkinje cell and acts as an inhibitor.
  Basal ganglia is important for the initiation and selection of motor programs. Parkinsons and Huntingtons are disorders of the basal ganglia.
  Deep brain stimulation (DBS) treatment for Parkinson’s stimulates subthalamic nucleus (STN) in the basal ganglia.
  Population activity of frontal cortical neurons encodes movement trajectories. Movement isn’t dependent on singular neurons.
  # **Navigation (29)**
  Many cues and senses are used for navigation.
  Ants can integrate their paths to find a direct path back to their nest.
  Ant odometer experiment: ants had their legs amputated or artificially extended with stilts. Ants with short legs undershoot their walk back to the nest. This suggests that ants count their steps for navigation. However, they can learn to adjust to their new leg lengths.
  # **Hippocampus, Place Cells (30)**
  Patient H.M. Henry Molaison.
  Had his hippocampi removed to treat epilepsy. Could not form new memories.
  Taxi drivers have enlarged hippocampi. Is neurogenesis in adults?
  Hippocampus circuitry: Dentate gyrus → CA3 → CA1
  In rats, different cells in CA1 fire when the rat is in different locations. These cells track the location of the rat. These cells have place fields that tile the environment and overlap. These fields are constructed and reconstructed during exploration.
  # **Grid Cells of the Entorhinal Cortex (31)**
  Grid cells in the Entorhinal cortex connect to the place cells. These cells fire at different places that form a gird.
  Along the dorsal-ventral axis the cells change from having a fine scaled grid to a coarse scaled grid. This forms a population code. This also functions as an error correcting code.
  # **Memory and Navigation (32)**
  Sequential place cell firing is replayed to strengthen the memory of the sequence of actions/movements
