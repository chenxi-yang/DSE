
def thermostat(lin):
    x = lin
    x_max = -100000
    x_min = 100000
    tOff = 78.0
    tOn = 66.0
    isOn = 0.0

    for i in range(40):
        if isOn <= 0.5: # ifblock1
            x = x - 0.1 * (x - lin)
            if x <= tOn: # ifelse_tOn
                isOn = 1.0
            else:
                isOn = 0.0
        else: # ifblock2
            x = x - 0.1 * (x - lin) + 5.0
            if x <= tOff:
                isOn = 1.0
            else:
                isOn = 0.0
        # print(f"x: {x}, isOn:{isOn}")
        x_min = min(x, x_min)
        x_max = max(x, x_max)
    # exit(0)
    
    return lin, x, x_min, x_max


