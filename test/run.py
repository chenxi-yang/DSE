
import sys
import constants

if __name__ == "__main__":
    
    constants.status = 1
    from program import *

    print(outer_test(1))

    constants.status = 2
    from program import *

    print(outer_test(1))


