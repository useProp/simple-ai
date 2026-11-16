import random
import math

HEIGHT = 7
WIDHT = 5
SIZE = HEIGHT * WIDHT
DIGITS = {
    0: [" ### ",
        "#   #",
        "#   #",
        "#   #",
        "#   #",
        "#   #",
        " ### "],
    1: ["  #  ",
        " ##  ",
        "  #  ",
        "  #  ",
        "  #  ",
        "  #  ",
        " ### "],
    2: [" ### ",
        "#   #",
        "    #",
        "   # ",
        "  #  ",
        " #   ",
        "#####"],
    3: [" ### ",
        "#   #",
        "    #",
        "  ## ",
        "    #",
        "#   #",
        " ### "],
    4: ["   # ",
        "  ## ",
        " # # ",
        "#  # ",
        "#####",
        "   # ",
        "   # "],
    5: ["#####",
        "#    ",
        "#### ",
        "    #",
        "    #",
        "#   #",
        " ### "],
    6: [" ### ",
        "#    ",
        "#    ",
        "#### ",
        "#   #",
        "#   #",
        " ### "],
    7: ["#####",
        "    #",
        "   # ",
        "  #  ",
        "  #  ",
        "  #  ",
        "  #  "],
    8: [" ### ",
        "#   #",
        "#   #",
        " ### ",
        "#   #",
        "#   #",
        " ### "],
    9: [" ### ",
        "#   #",
        "#   #",
        " ####",
        "    #",
        "    #",
        " ### "],
}

random.seed(1)

def to_vector(rows):
    vector = []
    for row in rows:
        for ch in row:
            vector.append(1.0 if ch == '#' else 0.0)
    return vector

def show(vector):
    for r in range(HEIGHT):
        line = ""
        for c in range(WIDHT):
            line += "#" if vector[r * WIDHT + c] > 0.5 else "."
        print(line)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def neuron(inputs, weights, bias):
    total = bias
    for x, w in zip(inputs, weights):
        total += x * w
    return sigmoid(total)

def make_layer(n_inputs, n_neurons):
    weights = []
    for _ in range(n_neurons):
        weights.append([random.uniform(-1, 1) for _ in range(n_inputs)])
    return { "w": weights, "b": [0.0] * n_neurons }

def layer_forward(layer, inputs):
    outputs = []
    for weights, bias in zip(layer['w'], layer['b']):
        outputs.append(neuron(weights, inputs, bias))
    return outputs

def make_net():
    return { "hidden": make_layer(35, 8), "output": make_layer(8, 10) }

def forward(net, inputs):
    hidden = layer_forward(net['hidden'], inputs)
    output = layer_forward(net['output'], hidden)

    return hidden, output

def predict(net, inputs):
    hidden, output = forward(net, inputs)
    return output.index(max(output))

net = make_net()
for digit, rows in DIGITS.items():
    print(digit, '->', predict(net, to_vector(rows)))

