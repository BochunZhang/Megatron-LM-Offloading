from patch import *

output = TextOutput(path='./logs')

config = Config()


def func_a(times=2):
    if times > 0:
        func_a(times-1)
        func_a(times-1)

def func_b():
    func_a()
    func_a()

def main():
    func_b()

if __name__ == "__main__":
    with PyCallGraph(output=output, config=config):
        main()