# Code Structure

1. **main_LCRRotHopPP.py:** Main ﬁle to train and evaluate LCR-Rot-hop++ using a Keras HyperBand tuner. Loads training and test data. Saves necessary embeddings, probabilities and, the true and predicted sentiment polarities
2. **AspEntQuaNet**
	- **asp_ent_qua_net.py:** File for the quantification model, can be used for all sorting measures
	- **baselines.py:** File containing functions to compute baselines: CC, PCC, ACC, PACC
	- **main_asp_class_err_qua_net.py:** Main ﬁle to train and evaluate (Asp)ClassErrQuaNet using a Keras HyperBand tuner. Loads embeddings, probabilities, and, the true and predicted sentiment polarities
	- **main_asp_gini_qua_net.py:** Main ﬁle to train and evaluate (Asp)GiniQuaNet using a Keras HyperBand tuner. Loads embeddings, probabilities, and, the true and predicted sentiment polarities
	- **main_asp_ent_qua_net.py:** Main ﬁle to train and evaluate (Asp)EntQuaNet using a Keras HyperBand tuner. Loads embeddings, probabilities, and, the true and predicted sentiment polarities
 	- **measures.py:** File containing functions to compute evaluation measures
  	- **subdatasets.py:** File containing functions to create subdatasets and training data  
3. **models**
	- **layers**
		- **attention.py:** File containing the attention mechanisms/layers classes
    	- **embedding.py:** File containing the class to load the BERT embeddings
	- **LCRRotHopPP.py:** File for the LCR-Rot-hop++ model
4. **utils**
	- **analysis.py:** File containing function to calculate quantiﬁcations and prevalence statistics
 	- **data_loader.py:** File containing functions to load various data
  	- **download_data.py:** File containing links to the SemEval datasets
  	- **SemEval_to_CSV.py:** File used to convert SemEval datasets from .xml format to .csv format
  	- **separate_categories.py:** File used to separate csv test dataset into the diﬀerent aspect categories. Saves a new ﬁle for every aspect category


# Replication guide for code

1. Download Bert Pre-processing and bert en uncased from the links in embedding.py and change the paths
2. Download SemEval datasets from links in download data.py
3. Convert SemEval datasets from .xml to .csv with SemEval to CSV.py
4. Split test data into categories with separate categories.py
5. main LCRRotHopPP.py
	- **Change paths:** training data and test data.
  	- **Change paths:** saving the results.
6. main asp ent qua net.py, main asp gini qua net.py, main asp class err qua net.py
  	- **Change paths:** the loading the training and test data.

