import sys
import constants

print(constants.status)
if constants.status == 1:
    from modules1 import *
elif constants.status == 2:
    from modules2 import *

def outer_test(x):
    return test(x) 