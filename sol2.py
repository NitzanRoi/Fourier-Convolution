import numpy as np
from skimage import color
from skimage import io
from scipy import signal


GREYSCALE_CODE = 1
RGB_CODE = 2
MAX_PIXEL_VAL = 255
DERIVATIVE_KERNEL = np.array([1, 0, -1])
DERIVATIVE_KERNEL_SIZE = (3, 3)
INVALID_BLUR_KERNEL_MSG = "Error: the given kernel size is not valid"
GAUSSIAN_BASIC_KERNEL = np.array([1, 1])


def read_image(filename, representation):
    """
    this function reads an image file and returns it in a given representation
    filename is the image
    representation code: 1 is greyscale, 2 is RGB
    returns an image
    """
    final_img = io.imread(filename)
    if (representation == GREYSCALE_CODE):
        final_img = color.rgb2gray(final_img)
    else:
        final_img /= MAX_PIXEL_VAL
    return final_img.astype(np.float64)


def create_vandermonde(is_dft, signal_len):
    """
    creates vandermonde matrix
    """
    if (is_dft):
        sign_mul = -2
    else:
        sign_mul = 2
    w = np.power(np.e, ((sign_mul * np.pi * np.complex(0, 1)) / signal_len))
    w_vector = np.array([np.power(w, np.arange(signal_len))])
    return np.vander(w_vector.ravel(), increasing=True)


def DFT_bonus(signal):
    """
    bonus function.
    transforms 2-D discrete signal to its Fourier representation
        signal - matrix of float64 numbers with shape (M, N)
        returns complex fourier signal
    """
    return np.dot(create_vandermonde(True, signal.shape[0]),
                  np.dot(signal, create_vandermonde(True, signal.shape[1])))


def IDFT_bonus(fourier_signal):
    """
    bonus function.
    transforms 2-D discrete Fourier representation to its original signal
        fourier_signal - matrix of 128complex numbers with shape (M, N)
        returns the restored original signal
    """
    normal_factor = 1 / (fourier_signal.shape[0] * fourier_signal.shape[1])
    return np.dot(create_vandermonde(False, fourier_signal.shape[0]),
                  np.dot(fourier_signal,
                         create_vandermonde(False, fourier_signal.shape[1]))) * normal_factor


def DFT(signal):
    """
    transforms 1-D discrete signal to its Fourier representation
        signal - array of float64 numbers with shape (N, 1)
        returns complex fourier signal
    """
    signal_len = signal.shape[0]
    vandermonde = create_vandermonde(True, signal_len)
    return np.dot(vandermonde, signal)


def IDFT(fourier_signal):
    """
    transforms 1-D discrete Fourier representation to its original signal
        fourier_signal - array of 128complex numbers with shape (N, 1)
        returns the restored original signal
    """
    signal_len = fourier_signal.shape[0]
    normal_factor = 1 / signal_len
    vandermonde = create_vandermonde(False, signal_len)
    return np.dot(vandermonde, fourier_signal) * normal_factor


def DFT2(image):
    """
    converts 2D discrete signal to its Fourier representation
        image - greyscale image of float64
    implemented with the bonus dft function
    """
    return DFT_bonus(image)


def IDFT2(fourier_image):
    """
    converts 2D Fourier representation to its original signal
        fourier_image - array of complex128
    implemented with the bonus idft function
    """
    return IDFT_bonus(fourier_image)


def conv_der(im):
    """
        im - greyscale image of type float64
    returns the magnitude of image derivatives (float64)
    """
    x_derivative_kernel = np.zeros(DERIVATIVE_KERNEL_SIZE)
    x_derivative_kernel[1, :] = DERIVATIVE_KERNEL
    y_derivative_kernel = np.zeros(DERIVATIVE_KERNEL_SIZE)
    y_derivative_kernel[:, 1] = DERIVATIVE_KERNEL
    x_derivative = signal.convolve2d(im, x_derivative_kernel, mode='same')
    y_derivative = signal.convolve2d(im, y_derivative_kernel, mode='same')
    return np.sqrt(np.abs(x_derivative)**2 + np.abs(y_derivative)**2).astype(np.float64)


def fourier_der(im):
    """
        im - greyscale image of type float64
    returns the magnitude of the image derivatives by using the Fourier transform
    """
    num_of_rows = im.shape[0]
    num_of_cols = im.shape[1]
    im_fourier = DFT2(im)
    mult_factor_x = (2 * np.pi * np.complex(0, 1)) / num_of_rows
    mult_factor_y = (2 * np.pi * np.complex(0, 1)) / num_of_cols
    u_mult_x = np.arange(-1 * num_of_rows / 2, num_of_rows / 2)
    u_mult_x = np.fft.fftshift(u_mult_x.reshape(u_mult_x.shape[0], 1))
    v_mult_y = np.arange(-1 * num_of_cols / 2, num_of_cols / 2)
    v_mult_y = np.fft.fftshift(v_mult_y.reshape(1, v_mult_y.shape[0]))
    x_derivative = mult_factor_x * IDFT2(u_mult_x * im_fourier)
    y_derivative = mult_factor_y * IDFT2(v_mult_y * im_fourier)
    return np.sqrt(np.abs(x_derivative) ** 2 + np.abs(y_derivative) ** 2).astype(np.float64)


def create_gaussian_kernel(kernel_size):
    """
    calculates the gaussian 2D kernel using binomial coefficients and returns it
    """
    gaussian_x = np.zeros((kernel_size, kernel_size))
    gaussian_y = np.zeros((kernel_size, kernel_size))
    binomial_coefficients = GAUSSIAN_BASIC_KERNEL.copy()
    for i in range(kernel_size - 2):
        binomial_coefficients = signal.convolve(binomial_coefficients, GAUSSIAN_BASIC_KERNEL)
    gaussian_x[int(np.floor(kernel_size / 2)), :] = binomial_coefficients
    gaussian_y[:, int(np.floor(kernel_size / 2))] = binomial_coefficients
    gaussian_kernel = signal.convolve2d(gaussian_x, gaussian_y, mode='same')
    if (np.sum(gaussian_kernel) != 0):
        return gaussian_kernel / np.sum(gaussian_kernel)
    return gaussian_kernel


def blur_spatial(im, kernel_size):
    """
    performs image blurring using 2D convolution between image and Gaussian
        im - the image, float64 greyscale
        kernel_size - an odd integer
    returns output as float64 greyscale
    """
    if (kernel_size == 1):
        return im
    if (is_even_number(kernel_size) or kernel_size < 0):
        print(INVALID_BLUR_KERNEL_MSG)
        exit()
    gaussian_kernel = create_gaussian_kernel(kernel_size)
    return signal.convolve2d(im, gaussian_kernel, mode='same').astype(np.float64)


def padding_kernel_helper(wanted_dim_size, kernel_size):
    """
    calculates the number of zeros for kernel padding
    """
    return (np.floor(wanted_dim_size / 2) - ((kernel_size - 1) / 2))


def is_even_number(num):
    """
    return boolean - if num is even number
    """
    return (num % 2 == 0)


def padding_kernel(im_rows_size, im_cols_size, kernel_size, kernel):
    """
    calculates the number of zeros for padding and returns the padded matrix
    """
    pad_left = int(padding_kernel_helper(im_cols_size, kernel_size))
    pad_up = int(padding_kernel_helper(im_rows_size, kernel_size))
    if (is_even_number(im_cols_size)):
        pad_right = pad_left
        pad_left -= 1
    else:
        pad_right = pad_left + 1
        pad_left -= 1
    if (is_even_number(im_rows_size)):
        pad_bottom = pad_up
        pad_up -= 1
    else:
        pad_bottom = pad_up + 1
        pad_up -= 1
    return np.pad(kernel, [(pad_up, pad_bottom), (pad_left, pad_right)], 'constant')


def blur_fourier(im, kernel_size):
    """
    performs image blurring with gaussian kernel in Fourier space
        im - the image, float64 greyscale
        kernel_size - an odd integer
    returns output as float64 greyscale
    """
    if (kernel_size == 1):
        return im
    if (is_even_number(kernel_size) or kernel_size < 0):
        print(INVALID_BLUR_KERNEL_MSG)
        exit()
    fourier_im = DFT2(im)
    gaussian_kernel = create_gaussian_kernel(kernel_size)
    gaussian_kernel = padding_kernel(im.shape[0], im.shape[1], kernel_size, gaussian_kernel)
    gaussian_kernel = np.fft.ifftshift(gaussian_kernel)
    fourier_kernel = DFT2(gaussian_kernel)
    mult_fourier = np.multiply(fourier_im, fourier_kernel)
    return np.real(IDFT2(mult_fourier)).astype(np.float64)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    def tst_fft2(input):
        return np.fft.fft2(input)

    def tst_ifft2(fourier):
        return np.fft.ifft2(fourier)

    def show(im, is_fourier): # display images
        if (is_fourier):
            im=np.log(1+np.abs(im))
        plt.imshow(im, cmap=plt.get_cmap('gray'))
        plt.show()
#
#     # test_img = read_image("./test_images/monkey.jpg", GREYSCALE_CODE)
#     # test_img = read_image("./test_images/view1.jpg", GREYSCALE_CODE)
    test_img = read_image("./test_images/view1.jpg", GREYSCALE_CODE)

    #### dft 1d test - done
    # reshaped_img = test_img[0].reshape(test_img[0].shape[0], 1)  # for dft 1d
    # fourier = DFT(reshaped_img)
    # restored = IDFT(fourier)
    # print(np.allclose(reshaped_img, restored))
    ####

    #### dft 2d test - done
    fourier = DFT2(test_img)
    # restored = IDFT2(fourier)
    # print(type(restored[0][0]))
    # print(np.allclose(test_img, restored))

    fourier_test = tst_fft2(test_img)
    print(np.allclose(fourier,fourier_test))
    # restored_test = tst_ifft2(fourier_test)
    ####

    #### derivatives - done
    # magnitude = conv_der(test_img)
    # magnitude_f = fourier_der(test_img)
    # show(magnitude_f, True)
    ####

    #### blur - done
    # blur_im = blur_spatial(test_img, 11)
    # blur_im_f = blur_fourier(test_img, 11)
    # print(type(blur_im_f[0][0]))
    # print(blur_im_f.shape)
    # print(test_img.shape)
    # show(blur_im_f, True)
    ####