
def thermostat(lin):
    x = lin
    x_max = -100000
    x_min = 100000
    tOff = 66.0
    tOn = 80.0
    isOn = 0.0

    for i in range(40):
        if isOn <= 0.5:
            x = x - 0.1 * (x - lin)
            if x <= tOff:
                isOn = 1.0
            else:
                isOn = 0.0
        else:
            x = x - 0.1 * (x - lin) + 5.0
            if x <= tOn:
                isOn = 1.0
            else:
                isOn = 0.0
        x_min = min(x, x_min)
        x_max = max(x, x_max)
    
    return lin, x, x_min, x_max


