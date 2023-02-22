"""
2d Max Pooling Module.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import numpy as np


class MaxPooling:
    """
    Max Pooling of input
    """

    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        """
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        """
        out = None
        #############################################################################
        # TODO: Implement the max pooling forward pass.                             #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #############################################################################

        # conv_output x conv_output x n
        # h,w = (h,w + pad_h,pad_w - k / stride ) + 1

        # need...
        #   - width
        #   - height

        # return output by max pooling with kernel_size and stride

        n, c, h, w = np.shape(x)
        H_out = (h - self.kernel_size) // self.stride + 1
        W_out = (w - self.kernel_size) // self.stride + 1

        # create empty 2d array for future store
        out = np.empty((n, c, H_out, W_out))

        # loop and compute
        for n_idx in range(n):
            for c_idx in range(c):
                for h_idx in range(H_out):
                    for w_idx in range(W_out):
                        h1 = h_idx * self.stride
                        h2 = h1 + self.kernel_size
                        w1 = w_idx * self.stride
                        w2 = w1 + self.kernel_size
                        slicer = x[n_idx, c_idx, h1:h2, w1:w2]
                        out[n_idx, c_idx, h_idx, w_idx] = np.max(slicer)

        # print(out)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = (x, H_out, W_out)
        return out

    def backward(self, dout):
        """
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return:
        """
        x, H_out, W_out = self.cache
        #############################################################################
        # TODO: Implement the max pooling backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                     #
        #       2) You may find np.unravel_index useful                             #
        #############################################################################

        # https: // numpy.org / doc / stable / reference / generated / numpy.unravel_index.html
        #   - converts flat index or array of flat indices int o tuple of coordinates
        #   - clearly useful for image convolution

        # ...it would make more sense to unravel the index in the forward pass and store it in the cache
        #     for backwards pass...?

        dx = np.zeros_like(x)
        n, c, _, _ = np.shape(x)
        for n_idx in range(n):
            for c_idx in range(c):
                for h_idx in range(H_out):
                    for w_idx in range(W_out):
                        h1 = h_idx * self.stride
                        h2 = h1 + self.kernel_size
                        w1 = w_idx * self.stride
                        w2 = w1 + self.kernel_size
                        scalar_idx = np.argmax(x[n_idx, c_idx, h1:h2, w1:w2])
                        idx = np.unravel_index(scalar_idx, (self.kernel_size, self.kernel_size))
                        upstream_der = dout[n_idx, c_idx, h_idx, w_idx]
                        dx[n_idx, c_idx, h1:h2, w1:w2][idx] = upstream_der

        self.dx = dx

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
