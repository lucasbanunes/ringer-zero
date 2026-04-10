import matplotlib.pyplot as plt


def get_color_cycle():
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return color_cycle
