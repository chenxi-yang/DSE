from modules1 import *
import sys

print(test(1))

# del sys.modules['modules1']
# sys.modules['modules1'] = __import__('modules2')  
from modules2 import *

print(test(1))

from modules1 import *
print(test(1))

if __name__ == "__main__":
    for i in range(2):
        from modules2 import *
        print(test(1))
        from modules1 import *
        print(test(1))
