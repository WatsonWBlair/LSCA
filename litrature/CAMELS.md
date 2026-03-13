# Abstract

We present CAMELS, a Multimodal Understanding (MU) pipeline and alignment framework designed to enable real-time, naturalistic, Human–Computer Interaction (HCI). Traditional Automatic Speech Recognition (ASR) pipelines rely on converting audio features into textual transcripts as the primary intermediate representation of user utterances. Although this allows text-based LLMs to process user utterances, it introduces a number of critical flaws. First, it discards critical non-textual semantic cues such as tone, speech patterns, and contextual behavioral signals. Secondly, these pipelines rely on components that are not designed to handle streaming interactions. Our approach addresses these limitations by using multi-modal encoding to preserve semantic information as well as a novel temporal chunking strategy to enable real-time comprehension of user interactions.

CAMELS improves training efficiency by leveraging Parameter Efficient Fine Tuning (PEFT) strategies to project multimodal embeddings onto a shared latent space. It further enables online processing of user utterances via temporal chunking and Agentic AI, a stateful agent that tracks and updates conversational context within Latent Representational Space (LRS). CAMELS models conversations in real-time as Hypergraph Neural Networks (HGNNs) that capture higher-order dependencies across speakers, modalities, and temporal segments. These HGNN representations are then used by multi-agent systems to generate naturalistic, context-aware responses across evolving multimodal interactions. We hypothesize that operating directly in LRS will reduce computational overhead sufficiently to enable real-time simulation-based learning exercises. By unifying multimodal signals in a shared latent space, CAMELS provides a practical framework for low-latency, context-aware conversational AI, with planned evaluations targeting multimodal dialogue understanding benchmarks.

**Keywords:** *Conversational AI, Multimodal Understanding, Large Concept Model, Latent Space, Coordinated Learning, Latent Embedding, Automatic Speech Recognition*

---

# 1. Introduction

***Why is this an important field of research?***

Conversational AI is an ever more present feature in our day to day lives, providing a seamless interaction with AI powered systems. Conversational AI also presents a large opportunity for automation; virtual assistants can take reservations, schedule doctors appointments, and help users externalize ideas and concepts.

***What is the problem we are solving?***

However, even though ASR pipelines enable voice-to-voice interactions with AI chatbots, they fall short in several regards:

- Cascade architectures introduce latency and a vulnerability to transcription errors.
- Semantic information that cannot be well represented in transcription is discarded.
- Most architectures do not support streaming, requiring users to complete a full utterance before the ASR pipeline can begin executing.
- Building Foundational Multimodal Models is computationally expensive.
- Struggle with highly multimodal data and complex relationships between modalities.
- Lack of semantically labeled data for complex interactions.

***What are some examples of projects that already address these issues? How do they fall short?***

To mitigate these issues we introduce a Multimodal Understanding Alignment Framework for Conversational Agents in Multimodal Embedded Latent Space (MUAF\_CAMELS).

\[Elevator Pitch for the project\]

***How does it mitigate each issue?***

**System Latency**

Operating in a latent modality shows efficiency gains in several recent studies.

\[CITE STUDIES AND SPECIFIC FINDINGS\]

This work contributes to the growing body of evidence that latent modalities are computationally efficient. We show efficiency by execution time as well as token throughput during benchmarking.

**Semantic Drift**

We include a reconstruction step in training cycles. This forces the latent representations to be coherent and extractable. This helps ensure systems remain interpretable.

**Discarded Semantic Data**

We allow for an arbitrary number of time-aligned encoders. Allowing for task-specific encoder swarms enables implementations to be tailored to specific use-cases and data-availability.

**Streaming Architecture**

We use a chunking strategy with tunable stride and overlap parameters. We show that hypergraph structures are able to model the complex set of naturalistic communication cues, allowing us to predict the end of a user's utterance and begin to generate a response before the user completes their utterance.

\[Discuss sliding q-former and attention mechanism\]

**Fine Tuning Costs**

Our architecture takes advantage of Parameter Efficient Fine Tuning Adapters to fine tune a model to a given modality while training only 1% of available weights. \[CITE\]

Our framework efficiently adapts existing backbone models to latent space. Reducing the computational and complexity cost of aligning to latent modalities enables further research into the field.

By leveraging PEFT Adapters, we reduce the computational cost of defining latent representational spaces and aligning models to them. This allows us to perform extensive ablative studies on both the disposition of the representational space, and the composition and architecture of agentic systems operating inside the representational space.

**Multimodal Understanding**

\[CITE\] shows that Dynamic Hypergraph Neural Networks (DHNN) are capable of modeling complex, multimodal, non-linear relationships.

**Lack of Labeled Data**

Representational learning allows us to utilize self-supervised learning. Contrastive learning should be verified using labeled data, but the non-linear relationships can be mapped without labeled data. We will use labeled data to define an initial representational space, then continue refining the space using self-supervised learning. This ensures we can train using unlabeled data, but allows us to ensure representations carry target semantics.

***How will we show the efficacy of the framework?***

To show the capabilities of the framework, we align a multimodal understanding pipeline that enables conversational interactions between CAMELS and users.

We will show the efficacy of the framework via the performance of aligned CAMELS systems at multimodal understanding benchmarks by official score, token usage, and execution time.

- MME-Emotion
- MME-Unify

We also provide benchmarking for various backbone models to evaluate their performance in the latent modality. Our benchmark results are based on the best-performing configuration for the benchmark set.

***Who is impacted by this problem being solved? How are they impacted?***

By lowering the barrier of entry to building multi-agent, multimodal systems we enable further research into Representational Modality's role in Multimodal Understanding.

***How does our work differ from earlier research?***

We provide a comprehensive framework for aligning and evaluating multimodal, multi-agent systems.

Our contributions are summarized as follows:

1. Alignment framework for information preserving bottleneck representational modality.
2. Hypergraph Context Management Architecture.
3. Multi-Agentic State Space Architecture.

The rest of the paper is organised as follows. Section 2 introduces Foundational Concepts and Related Works. Section 3 provides an overview of our methodology, benchmarks, and datasets. Section 4 reviews our experimental results. Finally we draw conclusions and outline further work in Section 5.

---

# 2. Foundational Concepts

For a full literature review of surveyed papers, see Appendix A.

***What background knowledge do you need to be able to understand the project?***

## 2.1 Representation Learning

***Provide a three sentence overview of the topic.***

Encoder-Decoder networks can capture information preserving representations of higher-dimensional data. These lower dimensional representations are generally more semantically dense, and can be created in a way that preserves representation interpretability. By learning a shared representation across multiple modalities, we enable AI systems to reason across modalities.

***Is there a particular work related to this topic that impacts our project?***

- \[Sonar\] shows the efficacy of shared latent spaces in translation tasks.
- \[Coconut\] shows that information dense representational modalities are computationally efficient.
- \[Multimodal Coordinated Representation Learning Based on Evidence Theory\] allows for uncertainty aware encoding fusion.
- \[Locatello et al\] shows that unsupervised disentangled representations may fail to capture target semantic attributes.
- \[Disentangled Alignment via Supervised Contrastive Learning\] shows how representational space can be disentangled in a way that ensures semantics.

***How does this topic impact our project?***

By decoupling encoding/decoding and inference tasks, we enable iterative improvement and ablation of different portions of our architecture.

## 2.2 Learning Strategies

***Provide 1 paragraph overview of the general topic.***

To overcome a lack of semantically labeled data, we identify a number of semi and self-supervised learning strategies. We execute training tasks on the bulk of unlabeled data, then evaluate performance against labeled data.

### 2.2.1 Contrastive Learning

***Provide a three sentence overview of the topic.***

Contrastive Learning is a Self-Supervised learning algorithm that learns to group similar items together and dissimilar ones further apart in geometric space.

***How does this topic impact our project?***

Contrastive learning will allow us to learn the complex non-linear relationships between modalities. Because it is self-supervised, the lack of labeled data does not impact encoder/decoder alignment.

***Is there a particular work related to this topic that impacts our project?***

- \[MoCo\] Allows for contrastive learning with sets of positive matches and a cross-modal stabilization mechanism.
- \[CITE\] shows how GAN and Reconstruction Loss can be used to ensure the latent space is information preserving.
- \[Disentangled Alignment via Supervised Contrastive Learning\] shows reconstruction loss can be used to disentangle latent spaces via supervised learning.

### 2.2.2 Generative Adversarial Networks

***Provide a three sentence overview of the topic.***

GANs rely on training models in an adversarial pair. Generators produce synthetic data, and Discriminators differentiate between synthetic and authentic data. These methods excel at producing realistic synthetic data.

***How does this topic impact our project?***

For conversational facilitation, it's critical that we can generate realistic responses to user interactions.

***Is there a particular work related to this topic that impacts our project?***

- \[CITE\] shows that GANs are effective at generating realistic content.
- \[CITE\] shows that GAN discriminators can be used to power RL loops.

### 2.2.3 Reinforcement Learning

***Provide a three sentence overview of the topic.***

RL uses iterative loops with a scoring function to drive model tuning. One of the benefits is that tuning can occur as the model is in use.

***How does this topic impact our project?***

This is our main training mechanism for agentic systems. We should design it so that RL loops continue during system operation.

***Is there a particular work related to this topic that impacts our project?***

- \[CITE\] shows that RL drives coherence across long-context tasks.

## 2.3 Adapter PEFT

***Provide a three sentence overview of the topic.***

Using adapters to project the last latent representation of a backbone model allows us to translate a model's internal understanding of a piece of data into a shared modality. Additionally, because we are adapting pretrained functionality, aligning backbone models to a representational modality is a relatively cheap task.

***How does this topic impact our project?***

This lets us efficiently align encoder swarms and agentic systems, allowing for full ablation of the dimensionality of latent representational space. This also reduces computational costs of building multimodal, multi-agent, spaces.

***Is there a particular work related to this topic that impacts our project?***

- \[CITE\] shows how information preserving bottleneck representational spaces can be learned.
- \[CITE\] shows how representations can be learned via self-supervised learning.

## 2.4 Shared Representation Space

***Provide a three sentence overview of the topic.***

Shared representational space refers to a semantically dense, cross-modal, representational space.

***How does this topic impact our project?***

The computational efficiency, and semantic density of representational space is intended to increase system performance. We need to show enough of an improvement to justify the translational loss of transitioning data across the representational space.

***Is there a particular work related to this topic that impacts our project?***

- \[Sonar\] shows how these spaces help translation tasks in low-resource languages.
- \[Coconut\] shows that operating in a representational space is token and compute efficient. We expect that our learned representational spaces will not be as effective as single-model continuous latent space.
- Although \[CITE\] shows how representational spaces can be defined in a self-supervised strategy, \[CITE\] shows that self-supervised methods can result in representations that fail to capture target semantic features. To address this we adopt a two phase approach to defining a shared latent space. We first use supervised methods introduced by \[Disentangled Alignment via Supervised Contrastive Learning\] on all labeled data, then continue unsupervised learning on unlabeled data. We hypothesize that by anchoring semantic representations during supervised training, we ensure that later unsupervised training cycles produce representational spaces that capture target features well. \[ABLATE ON THIS — What is the ratio tradeoff between supervised training and unsupervised training\]

## 2.5 Multimodal Fusion

***Provide a three sentence overview of the topic.***

Several strategies exist for building multimodal understanding systems. Multimodal Fusion combines embeddings to ensure temporal linkage. We explore several strategies including mid-fusion, where embeddings are concatenated, and weighted graph-fusion which relies on hyper-edges to manage temporal linkage between sensory streams.

***How does this topic impact our project?***

How we fuse the embeddings is going to change how the system performs. We'll want to ablate on fusion strategies.

***Is there a particular work related to this topic that impacts our project?***

- \[CITE\] Mid-Fusion

## 2.6 Hypergraph Neural Networks

***Provide a three sentence overview of the topic.***

Hypergraphs are graphs whose edges can connect an arbitrary number of nodes.

***How does this topic impact our project?***

Using hypergraphs allows us to model non-linear, long-context, relationships between multimodal embeddings.

***Is there a particular work related to this topic that impacts our project?***

- \[Dynamic Hypergraph Neural Networks\] shows the benefit of maintaining hypergraphs via a recursive convolution strategy.

---

# 3. Methodology

***What is the problem we are going to be solving?***

In this section we outline the three major functional units of the framework.

***What is the system architecture we are using? (show figure)***

***What are the main components of our solution?***

Our architecture is split cleanly between Encoder/Decoders that define a latent space and transition data into and out of it, and Latent Agents that operate within the latent space. Our encoder swarm consists of 4 units, each encoding a specific feature set from audio or video features. Embeddings are consumed by a Latent Agent that maintains a hypergraph representation of the user interaction. Additional Latent Agents perform convolutions on the hypergraph to simulate interactions or extract information.

## 3.1 Encoder Swarm

***What is this component's architecture? (show figure)***

Show both operational and training architectures.

***What is this component's expected input?***

Encoders execute a 1 dimensional convolution over video and audio content. Encoder pipelines need to be created for the feature set that is to be extracted.

***How will this component compute its output? (provide algorithm pseudo-code or mathematical formulation)***

Adapters project the final latent representation onto the latent space. Contrastive learning is used to define the cross modal relationships.

***How will this component be trained? (Encoder Alignment)***

We will pre-generate tokens for each backbone model. By training only the adapters we not only reduce computational requirements, but the tokenized data is much more compact than raw video.

We calculate loss via \[SHOW EQUATIONS\]

***How will we determine how successful the component is?***

Each encoder needs to be trained with a Decoder. To prove that the latent space represents an information preserving bottleneck, we must be able to reconstruct the source material from the embedding. We can calculate reconstruction drift as a metric of data degradation when passing through the latent space.

***What aspects of this component will we perform ablations on?***

We will ablate on backbone models, targeting 3 backbone variants for each pipeline. We will also ablate on the dimensionality of latent space representations — how do results change as dimensionality increases? Not just a flat 1024, but what about 64x64. We'll also want to explore scaling and weighting between the modalities.

## 3.2 Maintenance Agent

***What is this component's architecture? (show figure)***

Show both operational and alignment architectures.

***What is this component's expected input?***

The maintenance agent receives latent embeddings. It is responsible for updates to the interaction state graph.

***How will this component compute its output? (provide algorithm pseudo-code or mathematical formulation)***

Two pass system. The first integrates new embeddings into the hypergraph. The second pass maintains long-distance relationships within the graph, and collapses graph-nodes as necessary.

***How will this component be trained?***

Similar to how the encoders were trained, we will use pre-generated embeddings to fine-tune pretrained models to the modality.

We align existing LLM backbones via Adapter training. Be specific about how we adapt a model to this task. What kind of model are we adapting? What backbones are good options? Why?

***How will we determine how successful the component is?***

The ability to build and maintain the hypergraph in real time. Measure system lag in ms. Performance at relationship modeling tasks. Are there benchmarks that are particularly well suited to showing hypergraph performance?

Can we examine models operating in the space without a hypergraph state?

***What aspects of this component will we perform ablations on?***

- Latent dimensionality
- Backbone performance

## 3.3 Latent Space Agents

***What is this component's architecture? (show figure)***

***What is this component's expected input?***

Latent Space agents will perform convolutions on the interaction state hypergraph to generate predictions or simulate interactions.

***How will this component compute its output? (provide algorithm pseudo-code or mathematical formulation)***

***How will this component be trained?***

Continue the trend of training on embeddings from the previous step. Treat hypergraph states as training embeddings. Agents should be able to predict/generate utterances. Agents should also be able to reason about the hypergraph.

***How will we determine how successful the component is?***

Performance in Benchmark tasks.

***What aspects of this component will we perform ablations on?***

- Backbone models

---

***How will we evaluate overall system performance?***

Lag time, token usage, performance on benchmark tasks.

***What benchmarks fit our core use case best?***

Pretty much any multimodal understanding benchmark. EEM-Emotion and EEM-Unified are excellent targets.

## 3.4 Datasets

***What kinds of data support system performance and evaluation?***

Multimodal understanding.

### 3.4.1 Candor

***Write a paragraph that explains the dataset and why it's a relevant fit for us.***

Largest collection of remote interactions between strangers. Is annotated with time-aligned transcripts, and 200+ question survey from both participants.

***How did we acquire the dataset?***

Dataset was requested and granted. We will need to re-request the data to get the full set; we currently have 10% of available data.

***What data-wrangling steps were done on the dataset?***

Repackaged, metadata converted to standard format.

***What labels/features did we extract from the dataset?***

Questionnaire, transcript, audio, video.

### 3.4.2 Seamless Interaction

***Write a paragraph that explains the dataset and why it's a relevant fit for us.***

4000 hour dataset of in-person dyadic interactions. Only 4.5 hours of annotated interactions. We should plan to use the 4000 hours for self-supervised learning then validate performance on the annotated interactions.

***How did we acquire the dataset?***

Publicly available from Facebook and HuggingFace API.

***What data-wrangling steps were done on the dataset?***

Download interaction pair. Crop video to be mid-chest and up. Repackage files into common format.

***What labels/features did we extract from the dataset?***

Time-Aligned Transcription, Body position, face position.

### 3.4.3 VT-SSum

- ***Write a paragraph that explains the dataset and why it's a relevant fit for us.***
- ***How did we acquire the dataset?***
- ***What data-wrangling steps were done on the dataset?***
- ***What labels/features did we extract from the dataset?***

### 3.4.4 EEM-Emotion

- ***Write a paragraph that explains the dataset and why it's a relevant fit for us.***
- ***How did we acquire the dataset?***
- ***What data-wrangling steps were done on the dataset?***
- ***What labels/features did we extract from the dataset?***

### 3.4.5 EEM-Unified

- ***Write a paragraph that explains the dataset and why it's a relevant fit for us.***
- ***How did we acquire the dataset?***
- ***What data-wrangling steps were done on the dataset?***
- ***What labels/features did we extract from the dataset?***

---

# 4. Experiments and Results

***Provide a one paragraph introduction to the experiments we conducted and benchmarks we measured against.***

## 4.1 Ablations

***What ablative studies were conducted? What were the impacts?***

- Dimensions of Latent Space
- Chunk sizing
- etc.

## 4.2 Benchmarks

***What benchmarks did we evaluate against?***

***How did we do?***

---

# 5. Conclusions

***What are the conclusions we can draw from our results?***

***What further research is indicated by our results?***

---

# 6. Future Work

***What topics require further investigation?***

***What are our known-unknowns?***

***What topics deserve more detailed attention?***

---

# References

[1] L. Barrault, et al, "Large Concept Models: Language Modeling in a Sentence Representation Space," FAIR at Meta, Dec. 12, 2024. [Online]. Available: https://github.com/facebookresearch/large_concept_model

[2] P.A. Duquenne, H. Schwenk, B. Sagot, "SONAR: Sentence-Level Multimodal and Language-Agnostic Representations | Research - AI at Meta," *ai.meta.com*. https://ai.meta.com/research/publications/sonar-sentence-level-multimodal-and-language-agnostic-representations/

[3] E. Rosenfeld, P. Nakkiran, H. Pouransari, O. Tuzel, and F. Faghri, "APE: Aligning Pretrained Encoders to Quickly Learn Aligned Multimodal Representations," *arXiv.org*, 2022. https://arxiv.org/abs/2210.03927

[5] W. Li, D. Han, J. Dezert, and Y. Yang, "Multimodal Coordinated Representation Learning Based on Evidence Theory," *2024 27th International Conference on Information Fusion (FUSION)*, pp. 1–6, Jul. 2024, doi: https://doi.org/10.23919/fusion59988.2024.10706295

---

# Appendix

## A. Detailed Literature Review

**Format:**

### PAPER TITLE

1. What is the problem?
2. Why is it important?
3. What are the challenges?
4. What are the existing solutions (other publications) and what are the limitations of existing solutions?
5. What is the contribution of this work?

---

## B. Datasets

Given our focus on human-computer interaction, we opted to select datasets consisting of dyadic (between two participants) dialogues. This section provides an overview of the selected datasets.

### B.1 Seamless Interaction

A large-scale collection of over 4,000 hours of face-to-face interaction footage from over 4,000 participants in diverse contexts. Seamless Interaction provides full-body video as well as microphone recordings of speech.

As the dataset's literature points out, in-person dialogues differ from those facilitated via video conferencing. As such this dataset is a little off target for our intended use case.

The data consists of full body video, isolated audio, time-encoded transcript, and metadata capturing the personality and relationship between participants. Personality metadata captures 5 features: Agreeableness, Conscientiousness, Extraversion, Neuroticism, and Openness. Relationship metadata captures the degree of familiarity between the participants.

A subset of interactions were annotated both by the participants (first party) and external observers (third party). Annotations capture moments of interest, speaker internal state, internal state rationale, and Visual Elements. 4.74 hours of dyadic interactions were annotated.

The full dataset size is nearly 27 TB.

### B.2 Candor

An 850-hour corpus with more than 1 terabyte of audio, video, and transcripts, with moment-to-moment measures of vocal, facial, and semantic expression, together with an extensive survey of speakers' post-conversation reflections. Conversations were captured between January and November 2020; six rounds of data collection yielded a total of 1,656 dyadic conversations recorded over video chat.

The dataset is labeled via pre and post conversational survey of participants, providing a coarse-grained view of the participants' internal state going into and coming out of the conversations.

This dataset most closely matches our target use case of facilitating simulation-based learning in video-conferencing contexts.

---

## Data Wrangling

The general goal of data-wrangling operations in this project are focused on acquiring relevant data for development and training while maintaining a minimal memory footprint. Each dataset required slightly different data-wrangling steps, as defined below.

### Seamless Interaction

Seamless\_Interaction's codebase provides data-fetching utilities for acquiring interaction pairs (both sides of the conversation).

Data wrangling steps included discarding the bottom two thirds of the video, deemed unnecessary for our use case.

Wrangled conversation pairs take up roughly 300MB of disk space.

### Candor

Candor is available upon request. As of February '26 the author was extremely responsive.

Data wrangling for the dataset consisted primarily of discarding redundant/unnecessary data.

Wrangled conversation pairs take up roughly 200MB of disk space.

