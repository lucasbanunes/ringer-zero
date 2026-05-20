from keras.ops import custom_gradient, clip

@custom_gradient
def straight_through_clip(x, x_min, x_max):
    def grad(dy):
        return dy
    return clip(x, x_min, x_max), grad