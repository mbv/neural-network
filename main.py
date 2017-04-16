from network import NeuralNetwork
from training_data import TRAINING_DATA
from test_data import TEST_DATA

LEARNING_RATE = 0.1
MOMENTUM_RATE = 0.8

FIRST_LAYER = 5 * 7
SECOND_LAYER = 14
OUTPUT_LAYER = 6


def format_output(l):
    return ['{:.2f}'.format(i) for i in l]


def main():
    print('LEARNING_RATE', LEARNING_RATE)
    print('MOMENTUM_RATE', MOMENTUM_RATE)
    print('teaching')
    network = NeuralNetwork((FIRST_LAYER, SECOND_LAYER, OUTPUT_LAYER),
                            learning_rate=LEARNING_RATE,
                            momentum=MOMENTUM_RATE)
    network.teach(TRAINING_DATA, 1000)

    print('checking')
    for item in TEST_DATA:
        print(format_output(network.calculate(item['input'])), format_output(item['output']))

if __name__ == "__main__":
    main()
