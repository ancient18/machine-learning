import numpy as np
import mnist


class Conv:
    def __init__(self, num_filters):
        self.num_filters = num_filters
        self.filters = np.random.randn(num_filters, 3, 3)/9

    def iterate_regions(self, image):
        h, w = image.shape
        for i in range(h-2):
            for j in range(w-2):
                im_region = image[i:i+3, j:j+3]
                yield im_region, i, j

    def forward(self, input):
        h, w = input.shape
        output = np.zeros((h-2, w-2, self.num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(im_region*self.filters, axis=(1, 2))
        return output


class MaxPool:

    def iterate_regions(self, image):
        h, w, _ = image.shape
        new_h = h//2
        new_w = w//2

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i*2):(i*2+2), (j*2):(j*2+2)]
                yield im_region, i, j

    def forward(self, input):
        h, w, num_filters = input.shape
        output = np.zeros((h//2, w//2, num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(im_region, axis=(0, 1))

        return output


class SoftMax:
    def __init__(self, input_len, nodes):
        self.weights = np.random.randn(input_len, nodes)/input_len
        self.biases = np.zeros(nodes)

    def forward(self, input):
        input = input.flatten()

        input_len, nodes = self.weights.shape

        totals = np.dot(input, self.weights)+self.biases
        exp = np.exp(totals)

        return exp/np.sum(exp, axis=0)

test_images = mnist.test_images()[:1000]
test_labels = mnist.test_labels()[:1000]

train_images = mnist.train_images()
train_labels = mnist.train_labels()
softmax = SoftMax(13 * 13 * 8, 10)
conv = Conv(8)
pool = MaxPool()
output = conv.forward(train_images[0])
output = pool.forward(output)
output = softmax.forward(output)
# print(output.shape)  # (26, 26, 8)



