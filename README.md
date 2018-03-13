This folder contains resources for the following project:

##### Predicting final discharge diagnoses from intake medical notes.

The below project description can be found at : https://docs.google.com/document/d/1N0HwZGDdrVbHLN3LPpZTc5tDIis6ngKMpEWcIk1-Uto/edit#

"Current models of care rely on physicians to call in specialists to address specific medical needs. This consultation process takes time and can, at times, delay treatments that can help patients. A technical barrier to identifying consultants is being able to automatically identify relevant consultants from the given clinical data for a patient. Specifically, we want to predict from an intake clinical note what final diagnoses a patient will have and from that prediction, create clinical decision support tools to alert specialists of patients who could benefit from their highly focused knowledge. This work will require the latest in text classification technologies.

The student will work in concert with our data scientists toward developing and benchmarking relevant models."

In particular, there is:

- An course project done by a colleague and I on ICD9 code prediction. We implemented a hierarchical RNN model with attention in PyTorch on the medical discharge summaries in the MIMIC III dataset. In the 'hagru' folder.
- Some papers on text classification, in particular sentence-level CNNs, non-neural network models, and paper on multi-label classification tasks. In 'literature' folder. The 'review' document provides a brief review of this papers and how to use them. 
