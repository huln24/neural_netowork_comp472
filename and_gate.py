from argparse import ArgumentParser
import math

def sigmoid_func(x):
    return 1 / (1 + math.e ** -x)
    
def boolean_and(x1, x2):
    weight1 = 20
    weight2 = 20
    bias = -30
    return sigmoid_func(weight1 * x1 + weight2 * x2 + bias)
    

def main():
    parser = ArgumentParser()
    parser.add_argument("-x1", type=int, choices={0, 1}, required=True)
    parser.add_argument("-x2", type=int, choices={0, 1}, required=True)
    args = parser.parse_args()
    
    x1 = args.x1
    x2 = args.x2
    
    print(f"{x1} and {x2} => {round(boolean_and(x1, x2))}")
    
    
if __name__ == "__main__":
    main()
    