from network import NeuralNetwork

LEARNING_RATE = 0.1
MOMENTUM_RATE = 0.8

FIRST_LAYER = 5 * 7
SECOND_LAYER = 14
OUTPUT_LAYER = 5

data = [
    {
        'input': [
            4, 4, 4, 4, 4,
            4, 0, 0, 0, 4,
            4, 0, 0, 0, 4,
            4, 0, 0, 0, 4,
            4, 0, 0, 0, 4,
            4, 0, 0, 0, 4,
            4, 4, 4, 4, 4
        ],
        'output': [1, 0, 0, 0, 0]
    },
    {
        'input': [
            0, 0, 0, 0, 4,
            0, 0, 0, 0, 4,
            0, 0, 0, 0, 4,
            0, 0, 0, 0, 4,
            0, 0, 0, 0, 4,
            0, 0, 0, 0, 4,
            0, 0, 0, 0, 4
        ],
        'output': [0, 1, 0, 0, 0]
    },
    {
        'input': [
            0, 0, 4, 4, 4,
            0, 0, 0, 0, 4,
            0, 0, 0, 4, 0,
            0, 0, 4, 0, 0,
            0, 4, 0, 0, 0,
            4, 0, 0, 0, 0,
            4, 4, 4, 4, 4
        ],
        'output': [0, 0, 1, 0, 0]
    },
    {
        'input': [
            0, 4, 4, 4, 0,
            4, 0, 0, 0, 4,
            0, 0, 0, 0, 4,
            0, 0, 0, 4, 0,
            0, 0, 0, 0, 4,
            4, 0, 0, 0, 4,
            0, 4, 4, 4, 0
        ],
        'output': [0, 0, 0, 1, 0]
    },
    {
        'input': [
            4, 4, 4, 4, 4,
            4, 0, 0, 0, 0,
            4, 0, 0, 0, 0,
            4, 4, 4, 4, 0,
            0, 0, 0, 0, 4,
            0, 0, 0, 0, 4,
            4, 4, 4, 4, 0
        ],
        'output': [0, 0, 0, 0, 1]
    }
]

test_data = [
    {
        'input': [
            4, 4, 4, 4, 4,
            4, 1, 1, 1, 4,
            4, 1, 1, 1, 4,
            4, 1, 1, 1, 4,
            4, 1, 1, 1, 4,
            4, 1, 1, 1, 4,
            4, 4, 4, 4, 4
        ],
        'output': [1, 0, 0, 0, 0]
    },
    {
        'input': [
            4, 4, 4, 4, 4,
            4, 1, 1, 1, 4,
            4, 1, 1, 1, 4,
            4, 1, 1, 1, 4,
            4, 1, 1, 1, 4,
            4, 1, 1, 1, 4,
            4, 4, 4, 1, 4
        ],
        'output': [1, 0, 0, 0, 0]
    },
    {
        'input': [
            0, 0, 0, 0, 4,
            0, 0, 0, 4, 4,
            0, 0, 4, 0, 4,
            0, 0, 0, 0, 4,
            0, 0, 0, 0, 4,
            0, 0, 0, 0, 4,
            0, 0, 0, 0, 4
        ],
        'output': [0, 1, 0, 0, 0]
    },
    {
        'input': [
            1, 0, 4, 4, 4,
            1, 0, 0, 0, 4,
            0, 0, 0, 4, 0,
            0, 0, 4, 0, 0,
            0, 4, 0, 0, 0,
            4, 0, 0, 0, 4,
            4, 4, 4, 4, 4
        ],
        'output': [0, 0, 1, 0, 0]
    },
    {
        'input': [
            4, 4, 4, 4, 0,
            0, 0, 0, 0, 4,
            0, 0, 0, 0, 4,
            0, 0, 0, 4, 0,
            0, 0, 0, 0, 4,
            0, 0, 0, 0, 4,
            4, 4, 4, 4, 0
        ],
        'output': [0, 0, 0, 1, 0]
    },
    {
        'input': [
            4, 4, 4, 4, 4,
            4, 0, 0, 0, 0,
            4, 1, 1, 1, 0,
            4, 4, 4, 4, 4,
            0, 4, 0, 1, 4,
            0, 0, 2, 0, 4,
            4, 4, 4, 4, 0
        ],
        'output': [0, 0, 0, 0, 1]
    }
]


def format_output(l):
    return ['%.2f' % i for i in l]


def main():
    print('LEARNING_RATE', LEARNING_RATE)
    print('MOMENTUM_RATE', MOMENTUM_RATE)
    print('При вычислении отображается текущая среднеквадратичная ошибка')
    network = NeuralNetwork((FIRST_LAYER, SECOND_LAYER, OUTPUT_LAYER), learning_rate=LEARNING_RATE,
                            momentum=MOMENTUM_RATE)
    network.teach(data, 10000)

    print('Проверка обучения')
    for item in test_data:
        print(format_output(network.calculate(item['input'])), format_output(item['output']))

if __name__ == "__main__":
    main()
