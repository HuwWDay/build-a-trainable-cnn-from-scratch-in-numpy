"""
Build a Trainable CNN from Scratch in NumPy

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - argmax_rows
def argmax_rows(matrix):
    # TODO: return the index of the largest element in each row of a 2D array
    return np.argmax(matrix, axis=1)

# Step 2 - row_max
import numpy as np

def row_max(matrix):
    # TODO: return the maximum value of each row of `matrix` with keepdims True for broadcasting.
    return np.max(matrix, axis=1, keepdims=True)

# Step 3 - row_sum
import numpy as np

def row_sum(matrix):
    """Return per-row sums of a 2D array with shape (N, 1)."""
    # TODO: return the sum along axis 1 keeping the reduced dimension
    return np.sum(matrix,axis=1,keepdims=True)

# Step 4 - exp_shifted
import numpy as np

def exp_shifted(logits):
    """Subtract per-row max from logits and exponentiate elementwise."""
    # TODO: shift each row of logits by its max and return elementwise exp
    return np.exp(logits-np.max(logits,axis=-1,keepdims=True))

# Step 5 - stable_softmax
def stable_softmax(logits):
    # TODO: Compute a numerically stable softmax row-wise over (N, C) logits.
    return exp_shifted(logits)/row_sum(exp_shifted(logits))

# Step 6 - one_hot
import numpy as np

def one_hot(labels, num_classes):
    # 1. Initialize an (N, num_classes) matrix of zeros
    out = np.zeros((len(labels), num_classes), dtype=np.float32)
    
    # 2. Use advanced indexing to set the correct columns to 1.0
    out[np.arange(len(labels)), labels] = 1.0
    
    return out

# Step 7 - gather_true_class_probs
import numpy as np

def gather_true_class_probs(probs, labels):
    """Return probs[i, labels[i]] for every row i as a 1D length-N array."""
    # Create an array of row indices: [0, 1, ..., N-1]
    row_indices = np.arange(len(labels))
    
    # Index into probs simultaneously for rows and columns
    return probs[row_indices, labels]

# Step 8 - cross_entropy_loss
import numpy as np

def cross_entropy_loss(probs, labels, eps=1e-12):
    # TODO: return the mean negative log-likelihood of the true-class probabilities
    return -np.mean(np.log(gather_true_class_probs(probs, labels)+eps))

# Step 9 - accuracy
# ── Step 009  accuracy ──
def accuracy(logits_or_probs, labels):
    # Removed the [] brackets around the boolean comparison
    out = argmax_rows(logits_or_probs) == labels
    return np.mean(out)

# Step 10 - he_std
def he_std(fan_in):
    # TODO: return the He initialization standard deviation sqrt(2 / fan_in).
    return np.sqrt(2/fan_in)

# Step 11 - he_init
import numpy as np

def he_init(shape, fan_in, seed):
    """Sample a weight tensor from a normal distribution scaled by He std using the seed."""
    # 1. Set the legacy global seed
    np.random.seed(seed)
    
    # 2. Calculate the He standard deviation
    he_std = np.sqrt(2.0 / fan_in)
    
    # 3. Sample using the legacy normal function
    return np.random.normal(loc=0.0, scale=he_std, size=shape)

# Step 12 - init_zero_bias
import numpy as np

def init_zero_bias(length):
    # TODO: return a 1D float array of zeros with the given length.
    return np.zeros((length))

# Step 13 - pad_2d
import numpy as np

def pad_2d(images, pad):
    """Zero-pad the spatial (H, W) dims of a 4D (N, C, H, W) tensor by `pad` on each side."""
    # (N_before, N_after), (C_before, C_after), (H_before, H_after), (W_before, W_after)
    pad_width = ((0, 0), (0, 0), (pad, pad), (pad, pad))
    
    # Mode 'constant' with constant_values=0 fills the padded areas with zeros
    return np.pad(images, pad_width, mode='constant', constant_values=0)

# Step 14 - output_spatial_size
def output_spatial_size(input_size, kernel, stride, padding):
    # TODO: return the conv/pool output spatial dimension from input_size, kernel, stride, padding
    return int((input_size+2*padding-kernel)/stride)+1

# Step 15 - im2col
import numpy as np

def im2col(images, kernel_h, kernel_w, stride, padding):
    """Unroll overlapping patches of a 4D image tensor into a 2D column matrix."""
    N, C, H, W = images.shape
    
    # 1. Apply zero padding to the spatial dimensions
    padded = pad_2d(images, padding)
    
    # 2. Calculate the output spatial dimensions
    # I'm explicitly assigning variables here in case your helper expects a specific order!
   # Pass stride BEFORE padding!
    out_h = output_spatial_size(H, kernel_h, stride, padding)
    out_w = output_spatial_size(W, kernel_w, stride, padding)
    
    # 3. Create a placeholder tensor for the extracted patches
    # Shape: (Batch, Channels, Kernel_H, Kernel_W, Out_H, Out_W)
    col = np.zeros((N, C, kernel_h, kernel_w, out_h, out_w), dtype=images.dtype)
    
    # 4. Fast slicing: Only loop over the small kernel dimensions
    for y in range(kernel_h):
        y_max = y + stride * out_h
        for x in range(kernel_w):
            x_max = x + stride * out_w
            
            # Extract the grid of pixels for this specific kernel position across all images
            col[:, :, y, x, :, :] = padded[:, :, y:y_max:stride, x:x_max:stride]
            
    # 5. Transpose to align with the required matrix structure
    # Desired order: sample (0), row pos (4), col pos (5), channel (1), kernel_h (2), kernel_w (3)
    col_transposed = col.transpose(0, 4, 5, 1, 2, 3)
    
    # 6. Flatten into the final 2D matrix
    # Rows: N * out_h * out_w
    # Cols: C * kernel_h * kernel_w
    return col_transposed.reshape(N * out_h * out_w, -1)

# Step 16 - col2im
import numpy as np

def col2im(cols, images_shape, kernel_h, kernel_w, stride, padding):
    """Re-roll a 2D column matrix back into a 4D image tensor, accumulating overlaps."""
    N, C, H, W = images_shape
    
    # 1. Calculate output spatial dimensions (matching our corrected argument order!)
    out_h = output_spatial_size(H, kernel_h, stride, padding)
    out_w = output_spatial_size(W, kernel_w, stride, padding)
    
    # 2. Reshape the 2D matrix back into the 6D spatial layout
    # From: (N * out_h * out_w, C * kernel_h * kernel_w)
    # To:   (N, out_h, out_w, C, kernel_h, kernel_w)
    cols_reshaped = cols.reshape(N, out_h, out_w, C, kernel_h, kernel_w)
    
    # Transpose it back to our looping format: (N, C, kernel_h, kernel_w, out_h, out_w)
    cols_transposed = cols_reshaped.transpose(0, 3, 4, 5, 1, 2)
    
    # 3. Create a padded image placeholder for accumulation
    H_pad = H + 2 * padding
    W_pad = W + 2 * padding
    images_padded = np.zeros((N, C, H_pad, W_pad), dtype=cols.dtype)
    
    # 4. Fast slicing: Accumulate the patch values back into their original locations
    for y in range(kernel_h):
        y_max = y + stride * out_h
        for x in range(kernel_w):
            x_max = x + stride * out_w
            
            # CRITICAL: We use += here to sum the gradients where patches overlap!
            images_padded[:, :, y:y_max:stride, x:x_max:stride] += cols_transposed[:, :, y, x, :, :]
            
    # 5. Crop out the padding to return the original (N, C, H, W) shape
    if padding > 0:
        return images_padded[:, :, padding:padding+H, padding:padding+W]
        
    return images_padded

# Step 17 - conv2d_forward
import numpy as np

def conv2d_forward(x, weights, bias, stride, padding):
    """
    Convolve x with weights using im2col, add bias, 
    return output and a dictionary backprop cache.
    """
    # 1. Extract dimensions
    N, C, H, W = x.shape
    C_out, _, kernel_h, kernel_w = weights.shape
    
    # 2. Compute output spatial dimensions
    out_h = output_spatial_size(H, kernel_h, stride, padding)
    out_w = output_spatial_size(W, kernel_w, stride, padding)
    
    # 3. Unroll the input images using im2col
    # Shape: (N * out_h * out_w, C * kernel_h * kernel_w)
    cols = im2col(x, kernel_h, kernel_w, stride, padding)
    
    # 4. Reshape weights into a 2D matrix for multiplication
    # Original: (C_out, C, kernel_h, kernel_w)
    # Reshaped & Transposed: (C * kernel_h * kernel_w, C_out)
    w_cols = weights.reshape(C_out, -1).T
    
    # 5. Perform the matrix multiplication and add bias (broadcasting handles the bias)
    # Resulting shape: (N * out_h * out_w, C_out)
    out = cols @ w_cols + bias
    
    # 6. Reshape and transpose back into standard 4D output format
    # Step A: Reshape to (N, out_h, out_w, C_out)
    # Step B: Transpose to (N, C_out, out_h, out_w)
    out = out.reshape(N, out_h, out_w, C_out).transpose(0, 3, 1, 2)
    
    # 7. Package the cache perfectly for the backward pass
    cache = {
        "x_shape": x.shape,
        "weights": weights,
        "cols": cols,
        "stride": stride,
        "padding": padding,
        "kernel_h": kernel_h,
        "kernel_w": kernel_w
    }
    
    return out, cache

# Step 18 - conv2d_grad_input
import numpy as np

def conv2d_grad_input(d_out, cache):
    """
    Backpropagate the upstream gradient d_out through the conv input using col2im.
    Returns dx, the gradient of the loss with respect to the input x.
    """
    # 1. Unpack all necessary values from the cache
    x_shape = cache["x_shape"]
    weights = cache["weights"]
    stride = cache["stride"]
    padding = cache["padding"]
    kernel_h = cache["kernel_h"]
    kernel_w = cache["kernel_w"]
    
    N, C_out, out_h, out_w = d_out.shape
    
    # 2. Reshape d_out to match the 2D matrix structure from the forward pass
    # In forward, we did: out.reshape(N, out_h, out_w, C_out).transpose(0, 3, 1, 2)
    # To reverse this, we transpose back to (N, out_h, out_w, C_out) and flatten it
    d_out_reshaped = d_out.transpose(0, 2, 3, 1).reshape(N * out_h * out_w, C_out)
    
    # 3. Prepare the weights matrix
    # weights shape is (C_out, C_in, kernel_h, kernel_w)
    # We want the transpose of w_cols from the forward pass, which is conveniently (C_out, -1)
    w_cols_T = weights.reshape(C_out, -1)
    
    # 4. Compute the gradient w.r.t the unrolled columns (d_cols)
    # Shapes: (N*out_h*out_w, C_out) @ (C_out, C_in*kernel_h*kernel_w) 
    # Result: (N*out_h*out_w, C_in*kernel_h*kernel_w)
    d_cols = d_out_reshaped @ w_cols_T
    
    # 5. Re-roll the column gradients back into the original image layout
    # col2im handles the accumulation (+=) for overlapping patches perfectly!
    dx = col2im(d_cols, x_shape, kernel_h, kernel_w, stride, padding)
    
    return dx

# Step 19 - conv2d_grad_weights
import numpy as np

def conv2d_grad_weights(d_out, cache):
    """
    Compute the gradient of the loss with respect to the convolution weights.
    Returns dW shaped (C_out, C_in, kernel_h, kernel_w).
    """
    # 1. Unpack necessary values from the cache
    cols = cache["cols"]
    weights = cache["weights"]
    
    N, C_out, out_h, out_w = d_out.shape
    
    # 2. Reshape d_out exactly as we did in the input backward pass
    # Transpose to (N, out_h, out_w, C_out) and flatten spatial/batch dimensions
    # Shape: (N * out_h * out_w, C_out)
    d_out_reshaped = d_out.transpose(0, 2, 3, 1).reshape(N * out_h * out_w, C_out)
    
    # 3. Compute the gradient w.r.t the reshaped weights
    # d_out_reshaped.T has shape (C_out, N * out_h * out_w)
    # cols has shape (N * out_h * out_w, C_in * kernel_h * kernel_w)
    # Resulting dw_flat has shape (C_out, C_in * kernel_h * kernel_w)
    dw_flat = d_out_reshaped.T @ cols
    
    # 4. Reshape back into the original 4D weight tensor structure
    dw = dw_flat.reshape(weights.shape)
    
    return dw

# Step 20 - conv2d_grad_bias
import numpy as np

def conv2d_grad_bias(d_out):
    """
    Compute the gradient of the loss with respect to a convolution layer's bias.
    Returns a 1D array of length C_out.
    """
    # Sum over Batch (0), Height (2), and Width (3), leaving only Channels (1)
    return np.sum(d_out, axis=(0, 2, 3))

# Step 21 - conv2d_backward
def conv2d_backward(d_out, cache):
    # TODO: return (dx, dW, db) using the conv2d gradient helpers and the forward cache
    return conv2d_grad_input(d_out, cache), conv2d_grad_weights(d_out, cache), conv2d_grad_bias(d_out)

# Step 22 - maxpool2d_forward
import numpy as np

def maxpool2d_forward(x, kernel, stride):
    """
    Runs 2D max pooling over a 4D image tensor, caching the in-window argmax.
    Returns (out, cache).
    """
    N, C, H, W = x.shape
    
    # 1. Compute output spatial dimensions (padding is explicitly 0)
    # Using the fixed argument order from our previous debugging!
    out_h = output_spatial_size(H, kernel, stride, 0)
    out_w = output_spatial_size(W, kernel, stride, 0)
    
    # 2. Initialize the output and argmax tensors
    out = np.zeros((N, C, out_h, out_w), dtype=x.dtype)
    argmax = np.zeros((N, C, out_h, out_w), dtype=int)
    
    # 3. Slide the window over the spatial grid
    for y in range(out_h):
        y_start = y * stride
        y_end = y_start + kernel
        
        for x_out in range(out_w):
            x_start = x_out * stride
            x_end = x_start + kernel
            
            # Extract the window across all images and channels simultaneously
            # Shape: (N, C, kernel, kernel)
            window = x[:, :, y_start:y_end, x_start:x_end]
            
            # Flatten the spatial dimensions of the window to (N, C, kernel * kernel)
            window_flat = window.reshape(N, C, -1)
            
            # 4. Find the max value and its index
            out[:, :, y, x_out] = np.max(window_flat, axis=-1)
            
            # np.argmax returns indices in the range [0, kernel * kernel - 1]
            argmax[:, :, y, x_out] = np.argmax(window_flat, axis=-1)
            
    # 5. Package the cache for the backward pass
    cache = {
        "x_shape": x.shape,
        "argmax": argmax,
        "kernel": kernel,
        "stride": stride
    }
    
    return out, cache

# Step 23 - scatter_grad_window
import numpy as np

def scatter_grad_window(grad_value, argmax_index, kernel):
    """Place grad_value at the argmax position within a (kernel, kernel) zero array."""
    out = np.zeros(shape=(kernel, kernel))
    
    # Correct row-major unwinding formulas
    row = argmax_index // kernel
    col = argmax_index % kernel 
    
    out[row, col] = grad_value
    return out

# Step 24 - maxpool2d_backward
import numpy as np

def maxpool2d_backward(d_out, cache):
    """
    Routes upstream gradients back through max pooling by calling 
    scatter_grad_window for each output position.
    """
    # 1. Unpack values from the cache
    N, C, H, W = cache["x_shape"]
    argmax = cache["argmax"]
    kernel = cache["kernel"]
    stride = cache["stride"]
    
    _, _, out_h, out_w = d_out.shape
    
    # 2. Initialize the input gradient tensor with zeros
    dx = np.zeros((N, C, H, W), dtype=d_out.dtype)
    
    # 3. Loop through every sample, channel, and spatial output cell
    for n in range(N):
        for c in range(C):
            for y in range(out_h):
                y_start = y * stride
                y_end = y_start + kernel
                
                for x_out in range(out_w):
                    x_start = x_out * stride
                    x_end = x_start + kernel
                    
                    # Extract the single gradient scalar and its won argmax position
                    grad_value = d_out[n, c, y, x_out]
                    argmax_index = argmax[n, c, y, x_out]
                    
                    # Create the (kernel, kernel) window gradient block
                    window_grad = scatter_grad_window(grad_value, argmax_index, kernel)
                    
                    # 4. Accumulate the scattered window gradient into dx
                    dx[n, c, y_start:y_end, x_start:x_end] += window_grad
                    
    return dx

# Step 25 - relu_forward
def relu_forward(x):
    # TODO: Compute the elementwise ReLU and cache the input for backprop.
    out = np.maximum(x, 0)
    cache = {}
    cache["x"] = x
    return out,cache

# Step 26 - relu_backward
import numpy as np

def relu_backward(d_out, cache):
    """Mask the upstream gradient by the positive entries of the cached input."""
    # 1. Retrieve the original pre-activation input from the cache dictionary
    x = cache['x']
    
    # 2. Let the gradient flow where x > 0, otherwise zero it out
    return np.where(x > 0, d_out, 0.0)

# Step 27 - flatten_forward
import numpy as np

def flatten_forward(x):
    """Reshape a 4D feature map into a 2D batch matrix and cache the original shape."""
    # 1. Cache the original 4D shape tuple safely under a string key
    cache = {'x_shape': x.shape}
    
    # 2. Flatten all dimensions after the batch dimension
    # Using -1 tells NumPy to automatically calculate C * H * W for you
    out = x.reshape(x.shape[0], -1)
    
    return out, cache

# Step 28 - flatten_backward
import numpy as np

def flatten_backward(d_out, cache):
    # TODO: reshape the upstream gradient back to the original 4D feature map shape.
    shape = cache["x_shape"]
    return d_out.reshape(shape)

# Step 29 - linear_forward
import numpy as np

def linear_forward(x, weights, bias):
    """
    Perform the affine transform of a batch of feature vectors.
    Inputs:
    - x: Shape (N, D_in)
    - weights: Shape (D_in, D_out)
    - bias: Shape (D_out,)
    
    Returns a tuple of (out, cache).
    """
    # 1. Compute the linear transformation (N, D_in) @ (D_in, D_out) -> (N, D_out)
    # NumPy handles broadcasting the 1D bias vector across all N rows automatically
    out = x @ weights + bias
    
    # 2. Package the exact keys requested for the backward pass
    cache = {
        'x': x,
        'weights': weights
    }
    
    return out, cache

# Step 30 - linear_grad_input
import numpy as np

def linear_grad_input(d_out, cache):
    """Gradient of a linear layer w.r.t. its input X."""
    # TODO: return dL/dX given d_out (N, D_out) and cache['weights'] (D_in, D_out)
    weights = cache["weights"]
    return d_out @ weights.T

# Step 31 - linear_grad_weights
import numpy as np

def linear_grad_weights(x, dout):
    """Gradient of loss wrt linear-layer weights W of shape (D_in, D_out)."""
    # TODO: Compute the gradient of a linear layer's loss wrt its weight matrix W.
    return x.T @ dout

# Step 32 - linear_grad_bias
import numpy as np

def linear_grad_bias(dout):
    # TODO: Compute the bias gradient of a linear layer given upstream gradient dout.
    return np.sum(dout, axis=0)

# Step 33 - linear_backward
def linear_backward(dout, cache):
    # TODO: combine input, weight, and bias gradients for a linear layer using the cache
    x, W = cache["x"], cache["weights"]
    return linear_grad_input(dout, cache), linear_grad_weights(x, dout), linear_grad_bias(dout)

# Step 34 - softmax_cross_entropy_forward
import numpy as np

def softmax_cross_entropy_forward(logits, y):
    """Return the mean cross-entropy loss for logits (N, C) and integer labels y (N,)."""
    loss = cross_entropy_loss(stable_softmax(logits), y)
    
    # np.abs() strictly forces -0.0 to become 0.0
    return np.abs(loss)

# Step 35 - softmax_cross_entropy_backward
import numpy as np

def softmax_cross_entropy_backward(logits, y):
    """Return the fused softmax-cross-entropy gradient of shape (N, C)."""
    # 1. Extract Batch Size (N) and Number of Classes (C) directly from logits
    N, C = logits.shape
    
    # 2. Create the one-hot target matrix using the true number of classes
    yhat = one_hot(y, C)
    
    # 3. Calculate probabilities using your forward pass helper
    probs = stable_softmax(logits)
    
    # 4. Compute the fused gradient
    return (probs - yhat) / N

# Step 36 - sgd_step
import numpy as np

def sgd_step(param, grad, lr):
    # TODO: return the SGD-updated parameter array (param - lr * grad).
    return param-lr*grad

# Step 37 - adam_update_m
import numpy as np

def adam_update_m(m, grad, beta_one):
    # TODO: return the updated first moment estimate using beta_one and grad.
    return beta_one*m+(1-beta_one)*grad

# Step 38 - adam_update_v
import numpy as np

def adam_update_v(v, grad, beta_two):
    # TODO: return the updated Adam second moment estimate as an EMA of squared gradients.
    return beta_two*v + (1-beta_two)*grad**2

# Step 39 - adam_bias_correct
def adam_bias_correct(moment, beta, t):
    # TODO: return moment divided by (1 - beta**t) to undo Adam's zero-init bias.
    return moment/(1-beta**t)

# Step 40 - adam_param_step
import numpy as np

def adam_param_step(param, m_hat, v_hat, lr, eps):
    # TODO: apply one Adam parameter update using bias-corrected moments
    return param-lr*m_hat/(np.sqrt(v_hat)+eps)

# Step 41 - adam_step
import numpy as np

def adam_step(param, grad, m, v, t, lr, beta_one, beta_two, eps):
    """Chain the four Adam helpers and return (new_param, new_m, new_v)"""
    # 1. Update the moving averages
    new_m = adam_update_m(m, grad, beta_one)
    new_v = adam_update_v(v, grad, beta_two)
    
    # 2. Apply bias correction to BOTH moments
    m_hat = adam_bias_correct(new_m, beta_one, t)  # <-- This was missing!
    v_hat = adam_bias_correct(new_v, beta_two, t)
    
    # 3. Take the parameter step using the bias-corrected moments
    new_param = adam_param_step(param, m_hat, v_hat, lr, eps)
    
    return new_param, new_m, new_v

# Step 42 - init_conv_layer
def init_conv_layer(out_channels, in_channels, kernel_size, seed=0):
    """Build He-initialized weights and a zero bias for a single conv layer."""
    # 1. Calculate the proper fan-in for a 2D convolutional filter
    fan_in = in_channels * kernel_size * kernel_size
    
    # 2. Define the 4D shape for the weight tensor
    weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
    
    # 3. Initialize using your upstream helpers
    out = {}
    out["W"] = he_init(weight_shape, fan_in, seed=seed)
    out["b"] = init_zero_bias(out_channels)
    
    return out

# Step 43 - init_linear_layer
def init_linear_layer(in_features, out_features, seed=0):
    """Return a dict containing He-initialized weights and zero bias for a linear layer."""
    # 1. For a linear layer, fan_in is exactly the number of incoming features
    fan_in = in_features
    
    # 2. Define the 2D shape for the weight matrix
    weight_shape = (in_features, out_features)
    
    # 3. Initialize using your upstream helpers
    out = {}
    out["W"] = he_init(weight_shape, fan_in, seed=seed)
    out["b"] = init_zero_bias(out_features)
    
    return out

# Step 44 - init_lenet
def init_lenet(in_channels, num_classes, seed=0):
    """
    Initialize a classic LeNet-5 model with the exact expected channel 
    and feature dimensions.
    """
    params = {}
    
    # 1. Conv Block 1: 6 filters of size 5x5
    params['conv1'] = init_conv_layer(
        out_channels=6, 
        in_channels=in_channels, 
        kernel_size=5, 
        seed=seed
    )
    
    # 2. Conv Block 2: 16 filters of size 5x5 (input channels = 6 from conv1)
    params['conv2'] = init_conv_layer(
        out_channels=16, 
        in_channels=6, 
        kernel_size=5, 
        seed=seed
    )
    
    # 3. Dense FC1: 256 flattened features down to 120 hidden units
    # (256 comes from the 16 channels * 4x4 remaining image size after pooling twice)
    params['fc1'] = init_linear_layer(
        in_features=256, 
        out_features=120, 
        seed=seed
    )
    
    # 4. Dense FC2: 120 hidden units down to the final target class count
    params['fc2'] = init_linear_layer(
        in_features=120, 
        out_features=num_classes, 
        seed=seed
    )
    
    return params

# Step 45 - forward_conv_block
def forward_conv_block(x, W, b, pool_size, stride, pad):
    """Run conv2d -> relu -> maxpool2d and return (out, cache_dict)"""
    cache = {}
    
    # 1. Conv pass uses the provided 'stride'
    out, conv_cache = conv2d_forward(x, W, b, stride, pad)
    cache["conv_cache"] = conv_cache
    
    # 2. ReLU pass
    out, relu_cache = relu_forward(out)
    cache["relu_cache"] = relu_cache 
    
    # 3. CRITICAL FIX: Max pool should use 'pool_size' as its stride!
    out, pool_cache = maxpool2d_forward(out, pool_size, pool_size)  # <-- Changed 'stride' to 'pool_size'
    cache["pool_cache"] = pool_cache 
    
    return out, cache

# Step 46 - forward_classifier_block
def forward_classifier_block(x, fc1, fc2):
    """Run flatten -> linear -> relu -> linear and return logits plus a cache dict."""
    cache = {}
    
    # 1. Flatten Pass
    out, flatten_cache = flatten_forward(x)
    cache["flatten_cache"] = flatten_cache 
    
    # 2. First Linear Layer (Fully-Connected 1)
    out, fc1_cache = linear_forward(out, fc1["W"], fc1["b"])
    cache["fc1_cache"] = fc1_cache
    
    # 3. ReLU Activation Pass
    out, relu_cache = relu_forward(out)
    cache["relu_cache"] = relu_cache
    
    # 4. Second Linear Layer (Fully-Connected 2 / Logits)
    logits, fc2_cache = linear_forward(out, fc2["W"], fc2["b"])
    cache["fc2_cache"] = fc2_cache
    
    return logits, cache

# Step 47 - lenet_forward
def lenet_forward(x, params):
    """
    Run two conv blocks then the classifier block and return (logits, caches).
    
    Inputs:
        x: Input image tensor of shape (N, C, H, W)
        params: Dictionary containing 'conv1', 'conv2', 'fc1', 'fc2'
    """
    caches = {}
    
    # 1. Convolutional Block 1
    # Specs: pool_size=2, stride=1, pad=0
    out, block1_cache = forward_conv_block(
        x, 
        params['conv1']['W'], 
        params['conv1']['b'], 
        pool_size=2, 
        stride=1, 
        pad=0
    )
    caches['block1'] = block1_cache
    
    # 2. Convolutional Block 2
    # Specs: pool_size=2, stride=1, pad=0
    out, block2_cache = forward_conv_block(
        out, 
        params['conv2']['W'], 
        params['conv2']['b'], 
        pool_size=2, 
        stride=1, 
        pad=0
    )
    caches['block2'] = block2_cache
    
    # 3. Dense Classifier Block
    # Extracts features, flattens, and passes through the FC layers
    logits, classifier_cache = forward_classifier_block(
        out, 
        params['fc1'], 
        params['fc2']
    )
    caches['classifier'] = classifier_cache
    
    return logits, caches

# Step 48 - backward_conv_block
def backward_conv_block(dout, cache):
    """
    Backprop dout through the cached pool, relu, and conv layers in reverse order.
    
    Inputs:
        dout: Upstream gradient matching the block output tensor shape.
        cache: Dict containing 'conv_cache', 'relu_cache', and 'pool_cache'.
        
    Returns:
        tuple: (dx, dW, db) representing gradients w.r.t block input, weights, and bias.
    """
    # 1. Unpack individual layer caches from the master cache dictionary
    pool_cache = cache["pool_cache"]
    relu_cache = cache["relu_cache"]
    conv_cache = cache["conv_cache"]
    
    # 2. Backpropagate through the 2D Max Pooling layer
    # Output shape: matches the pre-pooled feature map shape
    dpool = maxpool2d_backward(dout, pool_cache)
    
    # 3. Backpropagate through the ReLU nonlinearity mask
    # Output shape: matches the pre-activation convolution output shape
    drelu = relu_backward(dpool, relu_cache)
    
    # 4. Backpropagate through the 2D Convolution layer
    # Using your optimized gradient helpers to get input, weight, and bias gradients
    dx = conv2d_grad_input(drelu, conv_cache)
    dW = conv2d_grad_weights(drelu, conv_cache)
    db = conv2d_grad_bias(drelu)
    
    return dx, dW, db

# Step 49 - backward_classifier_block
def backward_classifier_block(dlogits, cache):
    """
    Walk the classifier block backward using your verified modular helper functions.
    
    Inputs:
        dlogits: Upstream gradient matching the shape of output logits (N, D_out)
        cache: Dict containing 'flatten_cache', 'fc1_cache', 'relu_cache', 'fc2_cache'
        
    Returns:
        dict: Structured gradients for the block input ('dx') and parameters ('fc1', 'fc2')
    """
    # 1. Unpack individual layer caches from the master cache dictionary
    fc2_cache = cache["fc2_cache"]
    relu_cache = cache["relu_cache"]
    fc1_cache = cache["fc1_cache"]
    flatten_cache = cache["flatten_cache"]
    
    # 2. Backpropagate through the second linear layer (fc2) using linear_backward
    # Returns the gradient flowing into ReLU, along with fc2 parameter gradients
    drelu_post, dfc2_dW, dfc2_db = linear_backward(dlogits, fc2_cache)
    
    # 3. Backpropagate through the ReLU activation function
    drelu_pre = relu_backward(drelu_post, relu_cache)
    
    # 4. Backpropagate through the first linear layer (fc1) using linear_backward
    # Returns the gradient flowing into Flatten, along with fc1 parameter gradients
    dflatten, dfc1_dW, dfc1_db = linear_backward(drelu_pre, fc1_cache)
    
    # 5. Backpropagate through the Flatten layer using flatten_backward
    # Seamlessly reshapes the 2D gradient matrix back into the original 4D tensor shape
    dx = flatten_backward(dflatten, flatten_cache)
    
    # 6. Assemble the final requested nested dictionary structure
    grads = {
        "dx": dx,
        "fc1": {
            "dW": dfc1_dW,
            "db": dfc1_db
        },
        "fc2": {
            "dW": dfc2_dW,
            "db": dfc2_db
        }
    }
    
    return grads

# Step 50 - lenet_backward
def lenet_backward(dlogits, caches):
    """
    Walk classifier and conv block caches in reverse to assemble all gradients.
    
    Inputs:
        dlogits: Upstream gradient at the logits layer of shape (N, num_classes).
        caches: Dictionary containing 'block1', 'block2', and 'classifier' caches.
        
    Returns:
        dict: A dictionary containing parameter gradients for all layers:
              'conv1', 'conv2', 'fc1', 'fc2'.
    """
    # 1. Backpropagate through the Dense Classifier Block
    # This calls your verified modular block function and returns a dict with:
    # 'dx' (gradient at flatten input), 'fc1' (subdict with dW/db), and 'fc2' (subdict with dW/db)
    classifier_grads = backward_classifier_block(dlogits, caches['classifier'])
    
    # 2. Backpropagate through Convolutional Block 2
    # We pass the 4D input gradient flowing out of the classifier block ('dx')
    dblock2_in, dW_conv2, db_conv2 = backward_conv_block(classifier_grads['dx'], caches['block2'])
    
    # 3. Backpropagate through Convolutional Block 1
    # We pass the gradient flowing out of block 2 into block 1
    _, dW_conv1, db_conv1 = backward_conv_block(dblock2_in, caches['block1'])
    
    # 4. Assemble every parameter gradient into the final master dictionary
    grads = {
        'conv1': {
            'dW': dW_conv1,
            'db': db_conv1
        },
        'conv2': {
            'dW': dW_conv2,
            'db': db_conv2
        },
        'fc1': classifier_grads['fc1'],
        'fc2': classifier_grads['fc2']
    }
    
    return grads

# Step 51 - lenet_predict
def lenet_predict(x, params):
    # TODO: Return the argmax class index per sample from a LeNet forward pass.
    out = lenet_forward(x, params)[0]
    return argmax_rows(out)

# Step 52 - build_synthetic_image_dataset
import numpy as np

def build_synthetic_image_dataset(num_samples, num_classes, image_size, in_channels=1, seed=0):
    """
    Produce a tiny labeled image set for sanity-checking the CNN.
    Returns a tuple of (x, y) where:
      - x: shape (num_samples, in_channels, image_size, image_size)
      - y: shape (num_samples,) containing integer labels in [0, num_classes)
    """
    # 1. Initialize the single recommended random number generator
    rng = np.random.default_rng(seed)
    
    # 2. Draw integer labels uniformly across the classes
    y = rng.integers(0, num_classes, size=num_samples)
    
    # 3. Initialize random NCHW pixel values using a standard normal distribution
    x = rng.standard_normal(size=(num_samples, in_channels, image_size, image_size))
    
    # 4. Make classes linearly separable by calculating the baseline shifts
    # Formula: k - (num_classes - 1) / 2
    shifts = y.astype(float) - (num_classes - 1) / 2.0
    
    # 5. Reshape shifts to (N, 1, 1, 1) to broadcast smoothly across 
    # channels, height, and width simultaneously
    x += shifts.reshape(-1, 1, 1, 1)
    
    return x, y

# Step 53 - shuffle_indices
import numpy as np

def shuffle_indices(n, seed=0):
    """Return a reproducible permutation of [0, n) using legacy global seeding."""
    # 1. Set the global legacy random seed
    np.random.seed(seed)
    
    # 2. Return the legacy permutation sequence
    return np.random.permutation(n)

# Step 54 - train_test_split
import numpy as np

def train_test_split(x, y, test_fraction=0.2, seed=0):
    """Partition x and y into train and test halves using a shared shuffled order."""
    n = len(y)
    
    # 1. Get the shared shuffled sequence by passing the total count and seed
    order = shuffle_indices(n, seed=seed)
    
    # 2. Calculate the exact integer split point for the test set
    num_test = int(n * test_fraction)
    
    # 3. Slice the shuffled indices into distinct train and test pools
    test_indices = order[:num_test]
    train_indices = order[num_test:]
    
    # 4. Use advanced indexing to split x and y simultaneously
    x_train, y_train = x[train_indices], y[train_indices]
    x_test, y_test = x[test_indices], y[test_indices]
    
    return x_train, y_train, x_test, y_test

# Step 55 - iterate_minibatches
import numpy as np

def iterate_minibatches(x, y, batch_size, seed=0):
    """
    Yield shuffled mini-batches of features and labels for one epoch of training.
    Drops any trailing partial batches.
    """
    N = x.shape[0]
    
    # 1. Get the shuffled index sequence for ALL samples in the dataset
    indices = shuffle_indices(N, seed=seed)
    
    # 2. Calculate the total number of complete mini-batches we can form
    # Using floor division (//) automatically handles dropping the trailing partial batch
    num_batches = N // batch_size
    
    # 3. Loop through and yield each batch chunk sequentially
    for b in range(num_batches):
        start_idx = b * batch_size
        end_idx = start_idx + batch_size
        
        # Pull out the target chunk of shuffled indices
        batch_indices = indices[start_idx:end_idx]
        
        # Yield the perfectly synchronized feature and label pairs
        yield x[batch_indices], y[batch_indices]

# Step 56 - train_step
def train_step(params, opt_state, x, y, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, t=1):
    """
    Run one full training iteration of the network.
    Maps gradient keys ('dW', 'db') correctly to parameter keys ('W', 'b').
    """
    # 1. Forward Pass
    logits, caches = lenet_forward(x, params)
    
    # 2. Compute the scalar loss value
    loss = softmax_cross_entropy_forward(logits, y)
    
    # 3. Compute the initial error signal at the logits boundary
    dlogits = softmax_cross_entropy_backward(logits, y)
    
    # 4. Backpropagate dlogits to collect all parameter gradients
    grads = lenet_backward(dlogits, caches)
    
    # 5. Optimization Pass
    new_params = {}
    new_opt_state = {}
    
    for layer in ['conv1', 'conv2', 'fc1', 'fc2']:
        new_params[layer] = {}
        new_opt_state[layer] = {}
        
        for pname in ['W', 'b']:
            w = params[layer][pname]
            
            # CRITICAL FIX: Prepend 'd' to match gradient keys ('dW' or 'db')
            dw = grads[layer]['d' + pname]  
            
            m = opt_state[layer][pname]['m']
            v = opt_state[layer][pname]['v']
            
            # Compute updates via your adam_step helper
            new_w, new_m, new_v = adam_step(
                w, dw, m, v, t, 
                lr=lr, beta_one=beta1, beta_two=beta2, eps=eps
            )
            
            new_params[layer][pname] = new_w
            new_opt_state[layer][pname] = {'m': new_m, 'v': new_v}
            
    return new_params, new_opt_state, loss

# Step 57 - train_one_epoch
def train_one_epoch(params, opt_state, x, y, batch_size, lr, beta_one, beta_two, eps, step_counter, seed=0):
    """
    Perform one full pass over the training data by iterating shuffled minibatches
    and applying a single Adam update per batch.
    
    Returns:
        tuple: (updated_params, updated_opt_state, final_step_counter, loss_list)
    """
    losses = []
    
    # 1. Iterate through perfectly aligned, shuffled minibatches
    for xb, yb in iterate_minibatches(x, y, batch_size, seed=seed):
        
        # 2. Advance the step counter by one for Adam bias correction
        step_counter += 1
        
        # 3. Apply the atomic training step to optimize parameters on this batch
        params, opt_state, loss = train_step(
            params, 
            opt_state, 
            xb, 
            yb, 
            lr=lr, 
            beta1=beta_one, 
            beta2=beta_two, 
            eps=eps, 
            t=step_counter
        )
        
        # 4. Collect the scalar loss value from this minibatch
        losses.append(loss)
        
    return params, opt_state, step_counter, losses

# Step 58 - train_loop
import numpy as np

def train_loop(params, x_train, y_train, num_epochs, batch_size, lr=1e-3, beta_one=0.9, beta_two=0.999, eps=1e-8, seed=0):
    """
    Drive the full multi-epoch training routine for the LeNet model.
    
    Inputs:
        params: Dictionary containing initial model parameters ('conv1', 'conv2', 'fc1', 'fc2').
        x_train: Complete training dataset features.
        y_train: Complete training dataset integer labels.
        num_epochs: Total number of complete passes over the dataset.
        batch_size: Total number of rows per mini-batch.
        lr, beta_one, beta_two, eps: Hyperparameters routed to Adam.
        seed: Base random seed to ensure reproducible shuffling.
        
    Returns:
        tuple: (params, loss_history) where loss_history is a flat list of all batch-level losses.
    """
    # 1. Initialize the Adam optimizer state to mirror the parameter tree structure exactly
    opt_state = {}
    for layer, layer_params in params.items():
        opt_state[layer] = {}
        for pname, pval in layer_params.items():
            opt_state[layer][pname] = {
                'm': np.zeros_like(pval),
                'v': np.zeros_like(pval)
            }
            
    # 2. Setup global tracking variables across all epochs
    step_counter = 0
    loss_history = []
    
    # 3. Iterate through the training epochs
    for epoch in range(num_epochs):
        # Generate a distinct, reproducible shuffle seed for the current epoch
        epoch_seed = seed + epoch
        
        # Execute one full epoch pass over the dataset
        params, opt_state, step_counter, epoch_losses = train_one_epoch(
            params, 
            opt_state, 
            x_train, 
            y_train, 
            batch_size, 
            lr, 
            beta_one, 
            beta_two, 
            eps, 
            step_counter, 
            seed=epoch_seed
        )
        
        # Flatten and concatenate batch losses into the global history list
        loss_history.extend(epoch_losses)
        
    return params, loss_history

# Step 59 - evaluate
def evaluate(params, x, y):
    # Fixed argument order: (x, params) instead of (params, x)
    guess = lenet_predict(x, params)
    return np.mean(guess == y)

