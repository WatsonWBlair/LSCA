# Data Wrangling Planning and Overview

This file provides an overview of our selected data sources and the wrangling steps involved with them. 

The goal of all data wrangling tasks is to produce audio and video file pairs that closely resemble the raw video footage that will be captured by a web-cam in model operation. ie: mid-chest and up centered on the speakers face, and cropped to the appropriate aspect ratio.

## Sources
Several sources have been identifed as highly relevant to our use case.

#### Seamless_Interaction
A collection of over 4,000 hours of in-person, face-to-face, interaction footage from more than 4,000 participants in diverse contexts.

It is worth noting that in-person interactions differ from those facilitated by Zoom. From the Dataset’s research paper: 
>> …research has found that remote interactions (e.g., audio and video conferencing) often differ substantially from in-person interactions in terms of turn-taking (Tian et al., 2024b),[and] gaze patterns (Horstmann and Linke, 2022)...

However the overall goals of the dataset are highly aligned with our research goals:

* Embodied AI and Virtual Agents
- Train agents that display natural gestures
- Model turn-taking dynamics and interaction rhythms
- Generate contextually appropriate responses to human behavior
* Multimodal Understanding
- Analyze cross-modal correlations between speech, gesture, and expressions
- Extract behavioral patterns from large-scale interaction data
- Develop models to understand social dynamics
* Human-Computer Interaction
- Design interfaces that respond to subtle human cues
- Improve telepresence technologies with better behavioral modeling
- Create more natural conversational agents

Paper: [Seamless Interaction: Dyadic Audiovisual Motion Modeling and Large-Scale Dataset](./litrature/DataSets/Seamless%20Interaction-%20Dyadic%20Audiovisual%20Motion%20Modeling%20and%20Large-Scale%20Dataset.pdf)

###### Wrangling Process
1) Crop images to show mid-chest and up (frame face as for video call)
2) Re-Package relevant training data objects into a directory that is labeled in a coherent way. 
3) Set up a master file that explains how individual segments are stitched together.

The idea is that we can stitch the conversation back together programmatically.



#### CANDOR Corpus
A collection of 1650 conversations that strangers had over video chat with rich metadata information obtained from pre-conversation and post-conversation surveys.

This dataset is a perfect match for us, but requires submitting an access request. The access has been granted and Were getting 
