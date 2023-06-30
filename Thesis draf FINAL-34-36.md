<a name="br1"></a> 

Appendix A

Appendix A: Code Structure

The code for replicating this work can be found via the following github link:

https://github.com/BasvanRzdl/ABSQ2023. This appendix explains the structure of the code

and the function of each ﬁle. An explanation of how to use the code can be found in Appendix

B.

• main LCRRotHopPP.py: Main ﬁle to train and evaluate LCR-Rot-hop++ using a

Keras HyperBand tuner. Loads training and test data. Saves necessary embeddings,

probabilities and, the true and predicted sentiment polarities.

• AspEntQuaNet

– asp ent qua net.py: File for the AspEntQuaNet model, simultaneously also for

AspGiniQuaNet and AspClassErrQuaNet.

– baselines.py: File containing functions to compute baselines: CC, PCC, ACC,

PACC.

– main asp class err qua net.py: Main ﬁle to train and evaluate AspClassErrQuaNet

using a Keras HyperBand tuner. Loads embeddings, probabilities, and, the true and

predicted sentiment polarities.

– main asp gini qua net.py; Main ﬁle to train and evaluate AspGiniQuaNet using a

Keras HyperBand tuner. Loads embeddings, probabilities, and, the true and predicted

sentiment polarities.

– main asp ent qua net.py: Main ﬁle to train and evaluate AspClassErrQuaNet us-

ing a Keras HyperBand tuner. Loads embeddings, probabilities, and, the true and

predicted sentiment polarities.

– measures.py: File containing functions to compute evaluation metrics

– subdatasets.py: File containing functions to create subdatasets and training data.

• models

– layers

∗ attention.py: File containing the attention mechanisms/layers classes.

∗ embedding.py: File containing the class to load the BERT embeddings.

– LCRRotHopPP.py: File for the LCR-Rot-hop++ model.

26



<a name="br2"></a> 

Appendix A Appendix A: Code Structure

27

• utils

– analysis.py: File containing function to calculate quantiﬁcations and prevalence

statistics.

– data loader.py: File containing functions to load various data.

– download data.py: File containing links to the SemEval datasets

– SemEval to CSV.py: File used to convert SemEval datasets from .xml format to

.csv format.

– separate categories.py: File used to separate csv test dataset into the diﬀerent

aspect categories. Saves a new ﬁle for every aspect category.



<a name="br3"></a> 

Appendix B

Appendix B: Replication guide

for code

1\. Download Bert Pre-processing and bert en uncased from the links in embedding.py and

change the paths.

2\. Download SemEval datasets from links in download data.py

3\. Convert SemEval datasets from .xml to .csv with SemEval to CSV.py.

4\. Split test data into categories with separate categories.py.

5\. main LCRRotHopPP.py

(a) Change paths: training data and test data.

(b) Change paths: saving the results.

6\. main asp ent qua net.py, main asp gini qua net.py, main asp class err qua net.py

(a) Change paths: the loading the training and test data.

28

