import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text # necessary for the BERT layers
import ssl
import certifi

ssl_context = ssl.create_default_context()
ssl_context.load_verify_locations(certifi.where())

class BERTEmbedding(tf.keras.layers.Layer):
    def __init__(self, **kwargs) -> None:
        # https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3
        preprocess_path = "/Users/basvanroozendaal/Downloads/DATA THESIS/Embeddings/bert_en_uncased_preprocess_3"
        # https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4
        bert_path = "/Users/basvanroozendaal/Downloads/DATA THESIS/Embeddings/bert_en_uncased_L-12_H-768_A-12_4"
        self.embedding_dim = 768

        print("Loading BERT preprocessing layer", end='\r')
        self.preprocessor = hub.KerasLayer(hub.load(preprocess_path))
        print("BERT preprocessing layer loaded", end='\r')

        print("Loading BERT layer", end='\r')
        self.bert = hub.KerasLayer(hub.load(bert_path))
        print("BERT layer loaded", end='\r')
        super().__init__(**kwargs)
    
    def call(self, inputs):
        preproc = self.preprocessor(inputs)
        embeddings = self.bert(preproc)

        # We use the average of the last four hidden layers.
        # Other methods may result in better performance
        average_last_four = tf.math.reduce_mean(embeddings['encoder_outputs'][-4:], axis=0)
        return average_last_four