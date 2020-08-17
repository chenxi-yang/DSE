def func3(x, theta):
    i = 0
    isOn = 0
    while(i < 40):
        if(isOn <  0.5):
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
        assert(x < 120)
    
    return abs(x - Interval(55.0, 78.0))


theta in [0.0, 1.0]

def func0(x, theta):
    if (x < theta):
        x = x - 5
    else:
        x = x + 5
    
    assert(x < 7)

    return abs(x - [4, 6.5])



theta in [30, 40]

def func1(x, theta):
    x = Interval(7, 10)
    while(x < theta){
        x = x + 5
    }
    assert(x < 41)

    return abs(x - 38)


theta in [10, 25]

def func2(x, theta):
    x = Interval(7, 10)
    while(x < theta){
        x = x + theta
    }
    assert(x < 32)
    
    return abs(x - 28)



theta in [75, 85]

def func3(x, theta):
    i = 0
    isOn = 0
    while(i < 40):
        if(isOn <= 0.5):
            x = x - 0.1 * (x - 60)
            if(x < 68):
                isOn = 1.0
            else:
                isOn = 0.0
        else:
            x = x - 0.1 * (x - 60) + 5
            if(x < theta):
                isOn = 1.0
            else:
                isOn = 0.0

        i = i + 1
        assert(x < 120)
    
    return abs(x - 75)