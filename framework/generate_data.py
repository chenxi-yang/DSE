import pandas as pd
import random

# therostat example

def therostat(x, theta):
    i = 0
    isOn = 0
    while(i < 40):
        if(isOn < 0.5):
            x = x - 0.1 * (x - 60)
            if(x < theta):
                isOn = 1.0
            else:
                isOn = 0.0
        else:
            x = x - 0.1 * (x - 60) + 5
            if(x < 80):
                isOn = 1.0
            else:
                isOn = 0.0

        i = i + 1
    
    return x


# def data_generator(l, r, root, num):
#     data_list = list()

#     for i in range(num):
#         x = random.uniform(l, r)
#         y = f(x, theta)

#         data = [x, y]
#         data_list.append(data)
    
#     return pd.DataFrame(data_list, columns=['x', 'y'])











