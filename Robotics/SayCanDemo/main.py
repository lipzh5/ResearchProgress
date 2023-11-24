# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import numpy as np
    print_hi('PyCharm')
    testx = np.linspace(-1, 1, 10)
    t1 = testx[None, :]
    print(t1.shape)
    t2 = testx[...]
    print(t2.shape, t2)
    t3=testx[:]
    print(t3.shape, t3)
    t4 = np.expand_dims(testx, 0)
    print(t4.shape)
    # testxx = testx[..., None]
    # print(testx, type(testx), testx.shape)
    # print('===\n')
    # print(testxx, testxx.shape)
    # y = np.expand_dims(testx, -1)
    # print(y, y.shape)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
