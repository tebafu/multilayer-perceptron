from mlp import Mlp

if __name__ == '__main__':
    model = Mlp(layer_layout=(4, 15, 8, 2))

    model.structure_visualization()

    for idx, layer in enumerate(model.weights):
        print(f'\nLayer {idx}: ')
        for idx2, node in enumerate(layer):
            print(f'the weights of node {idx2} are: ')
            print(node)
