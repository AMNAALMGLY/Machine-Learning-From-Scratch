#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive Python implementation of a convolutional layer.
    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.
    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        
    During padding, 'pad' zeros should be placed symmetrically (i.e., equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.
    Returns an array.
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    """
    out = None

    N, C, H, W = x.shape
    num_filters, _, filter_height, filter_width = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']

    # Check dimensions.
    assert (W + 2 * pad - filter_width) % stride == 0, 'width does not work'
    assert (H + 2 * pad - filter_height) % stride == 0, 'height does not work'
         
    outputsize= ((H+ 2 * pad - filter_height) // stride) +1
    
    

    out =torch.zeros(N,num_filters,outputsize,outputsize)
    x=x.clone()
    x=torch.nn.functional.pad(x,(pad,pad,pad,pad))
   
    for point in  range(N):
      
       
        for f in range(num_filters):
            for i in range(outputsize):
                for j in range(outputsize):
                     
                     slide=x[point,:,i*stride:((i*stride)+w.shape[2]),j*stride:(j*stride)+w.shape[2]]*w[f,:,:,:]

                     out[point,f,i,j]= torch.sum(slide)+b[f]
    return out            

