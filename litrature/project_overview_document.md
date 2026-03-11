# Conversational Agents in Multimodal Embedded Latent Space

As AI becomes ever more present in our daily lives, achieving naturalistic Human-AI interaction has become a critical part of driving technology adoption. Traditional approaches leverage Automatic Speech Recognition(ASR) Pipelines to transform audio input into a textual transcript for use in inference. However, this approach discards semantic information that does not translate into a text-based modality; such as tone of voice or speech patterns. Multimodal Understanding(MU) is a subdomain of Natural Language Understanding(NLU) that seeks to integrate multiple types of data simultaneously to a holistic comprehension of user utterances, similar to human perception. We propose to combine recent advances build an online video MU pipeline capable of real-time voice-to-voice conversations with users in a video call context.


## Goals
We propose to improve on existing research presented in the paper “Multimodal Coordinated Representational Learning Based on Evidence Theory” in the following ways:
- Enable ablation studies of the system via`Adapted Pretrained Encoders` architecture to reduce training costs by 99%
- Enable a streaming architecture by sampling inputs at discrete intervals to enable predictive modeling for user utterances. 
- Examining the interpretability of End-to-End vs Cascade Architectures. 
- Implement inference and pattern recognition functionality within ‘Evidence Space` (Embedding Vector Space) to enable “Large Concept Model” inference capabilities.

## Objectives

#### Training and Alignment
* Outcome:
1) Enable rapid alignment of multi-modal encoder modules.
2) Enable distillation learning.
* Time Frame:
1) Complete by March 17th
* Measures:
1) Relative training costs and impacts between APE vs full tuning architectures.
* Actions:
1) Implement APE architecture (Already Completed for SHARD)
2) Identify specific training data-sets and benchmark targets. 
3) Define training methodology.
4) Ablation Study to exhaustively examine impacts of APE and adapter architectural features.
#### Streaming Architecture
* Outcome:
1) Enable a real-time streaming architecture for video-chat like interactions that supports predictive modeling of user utterances from multimodal inputs 
* Time Frame:
1) Complete by March 17th
* Measure:
1) Evaluate conversational latency using ablation studies between streaming and non-streaming architectures at sampling intervals. 
* Action: 
1) Design a processing pipeline to sample input at discrete time intervals
2) Implement video embedding memory bank to efficiently store embedded features for downstream inference and generation tasks.
3) Identify optimal sampling interval for utterance prediction modelling using ablation studies 
4) Ablation studies between streaming and non-streaming architecture for evaluation conversational latency and also check utterance prediction performance 
#### Pattern Recognition 
* Outcome:
Develop uncertainty-aware evidential embeddings to enable direct inference in belief space
Explore compositional reasoning over multimodal representations vis-à-vis Large Concept Model-style semantic generalization.
* Time Frame:
Complete by April 21st.
* Measure:
Concept Alignment, Brier Score, Robustness under modality ablation, Conflict metric sensitivity, compositional reasoning accuracy
* Actions:
Curate and prepare multimodal training data.
Design an efficient evidential data storage strategy.
Learn uncertainty-aware concept embeddings.
Implement evidential pattern recognition.
Conduct robustness & compositional tests.
#### Interpretability
* Outcome:
1) Develop a  "White-Box" reasoning framework that justifies interventions by tracing decisions back to specific multimodal evidence and resolved conflicts.
* Time Frame:
1) Complete April 21st
* Measure: 
1) Conflict Metric (K): Numerical value (0 to 1) representing the degree of disagreement between modalities. 
2) Belief-Plausibility Interval: The width of the [Bel,Pl] range, where a wider gap indicates higher "Ignorance" or lack of data.
3) Concept Alignment : The Cosine Similarity between the model’s predicted concept vector and the human-verified Target Vector.
4) Brier Score: Evaluates the calibration of the model's Belief (Bel) score to ensure the AI's confidence matches its actual accuracy.
* Actions:
1) Convert concept vectors into  Basic Belief Assignments (BBA) using Dempster-Shafer Theory
2) Modality Ablation Studies: remove specific inputs (video or audio) to measure the impact on accuracy and determine which modality provides the most critical evidence for the model's decision.
3) Generate Target Vectors by encoding human-verified transcripts from the MELD/IEMOCAP datasets into the SONAR space, then validate the model's accuracy by measuring the Cosine Similarity between its predicted concepts and the target vectors.
4) Evidence Tracing: Build a log that connects every intervention to a specific spike in the Conflict Metric (K), providing a clear audit trail for the model's behavior.

## Success Criteria: 
- Show improved performance on emotion recognition/classification
- Reduce conversational delay time to 300ms
- Demonstrate ability of economically bootstrap cross-modal latent spaces. 
- Interoperability measure
- Users rating of the conversations they have. Metrics like fluidity, conversational gaps, human-like-interactions, ect…
- System performance at conversational-specific tasks:
1) prediction tasks:
* utterance end points
* utterance semantics


## Assumptions, Risks, Obstacles: 
* Assumptions
1) We can leverage existing tooling in https://github.com/WatsonWBlair/cs627 (Multimodal Joint Representation Learning Via APE Swarm)
2) That we will be able to get volunteers to use the app, and get consent to record video of volunteers for training data.
3) Also assuming we can sufficiently sanitize the data to remove any privacy concerns.
4) That we can get IRB approval or Exemption
5) Finishing in the defined time-frame
6) Have access to GPU

* Risks
1) The scope of the project is too big for what we can actually accomplish
2) Data pipeline processing time and training time constraints.
3) The resources required to run the system make it hard to have people use it.

* Obstacles
1) IRB review / approval / exemption.
2) The amount of time that can be devoted to the project.(Addressed)
3) The team committed to at least 15 hours per week. 
4) Dependency on collaboration with Global Pathways
5) Access to Participants and Training Data.
6) Outreach to student clubs could help remediate

* Challenges
1) Learning new/unfamiliar tools and technologies.
