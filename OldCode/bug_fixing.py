from OldCode.fake_data import create_testing_input
from subdatasets import create_trainings_data
from asp_ent_qua_net import AspEntQuaNet
from tensorflow_addons.metrics import F1Score
from measures import relative_absolute_error

fake_data = create_testing_input(amount=300)

training_data = create_trainings_data(all_data=fake_data, sub_dataset_percentage=0.4, number_of_subdatasets=40, sort_key='entropy')
y_data = training_data.pop('true_prevalence')

layer1 = {"neurons": 256, "activation_function": "relu", "drop_out": 0.01}
layer2 = {"neurons": 128, "activation_function": "relu", "drop_out": 0.01}
layer3 = {"neurons": 64, "activation_function": "relu", "drop_out": 0.01}

hidden_layers = [layer1, layer2, layer3]

f1 = F1Score(num_classes=3, average='macro', name='f1')

model = AspEntQuaNet(hidden_layers=hidden_layers,use_quantification_statistics=True,use_prevalence_statistics=True,number_of_classes=3, bilstm_output_neurons=256, subdataset_size=120, vector_size=14)
model.compile(optimizer='adam', loss=relative_absolute_error, loss_weights=1, metrics = ['acc', f1], run_eagerly=True)
model.fit(training_data, y_data, epochs=2, steps_per_epoch=10, batch_size=4)
model.summary()

print(model.predict(training_data))