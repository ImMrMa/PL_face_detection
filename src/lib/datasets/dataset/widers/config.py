class Config(object):
    def __init__(self):
        self.iter_per_epoch = 4000

        self.point = 'center'  # or 'top', 'bottom
        self.scale = 'hw'  # or 'w', 'hw'
        self.num_scale = 2  # 1 for height (or width) prediction, 2 for height+width prediction
        self.offset = True  # append offset prediction or not
        self.down = 4  # downsampling rate of the feature map for detection
        self.radius = 2  # surrounding areas of positives for the scale map
        # setting for data augmentation
        self.use_horizontal_flips = True
        self.brightness = (0.5, 2)
        self.size_train = (704,704)
        self.img_channel_mean = [103.939, 116.779, 123.68]
