import random
random.seed(0)


def create_testing_input(amount: int) -> dict:
    testing_input = {
        'y_true': create_predictions(amount),
        'y_pred': create_predictions(amount),
        'probabilities': create_probabilities(amount),
        'embeddings': [[random.uniform(0, 1) for _ in range(10)] for _ in range(amount)]
    }
    return testing_input


def create_probabilities(amount: int) -> list:
    probabilities = list()

    for i in range(amount):
        prob_vector = [random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)]
        probabilities.append(prob_vector)

    return probabilities


def create_predictions(amount: int):
    pred = [-1, 0, 1]
    predictions = [random.choice(pred) for _ in range(amount)]
    return predictions


