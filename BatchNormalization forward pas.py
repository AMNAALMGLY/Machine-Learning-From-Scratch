#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class MyBatchNorm2d(nn.Module):
    """Simple implementation of batch normalization."""

    def __init__(self, num_channels, momentum=0.1, epsilon=1e-5):
        super(MyBatchNorm2d, self).__init__()

        # Initialize bias and gain parameters.
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

        # Initialize moving averages.
        self.epsilon = epsilon
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros((1, num_channels, 1, 1)))
        self.register_buffer('running_var', torch.ones((1, num_channels, 1, 1)))
        
    def forward(self, x):
        # Check that input is of correct size.
        assert x.dim() == 4, 'input should be NCHW'
        assert x.size(1) == self.gamma.numel()
        #START_GRADING
        mean=torch.mean(x,dim=(0,2,3),keepdim=True)
        
        
        var=torch.var(x,dim=(0,2,3),keepdim=True)
       
        
        
        if mode:
            self.running_mean=(1-self.momentum)*self.running_mean+self.momentum*mean
            self.running_var=(1-self.momentum)*self.running_var+self.momentum*var
            output=(x-mean)/torch.sqrt(var+self.epsilon)
            
        else:
             output=(x-self.running_mean)/(torch.sqrt(self.running_var+self.epsilon))
        y=self.gamma*output+self.beta    
        return  y
        #END_GRADING
    


# Use this code to test if your implementation is correct.
batch_size, num_channels, im_size = 32, 8, 6
batchnorm1 = nn.BatchNorm2d(num_channels)
batchnorm2 = MyBatchNorm2d(num_channels)
for key, param in batchnorm1.named_parameters():
    if key == 'weight':
        param.data.fill_(1.0)  # undo random initialization in nn.BatchNorm2d
for mode in [True, False]:     # test in training and evaluation mode
    batchnorm1.train(mode=mode)
    batchnorm2.train(mode=mode)
    for _ in range(5):
        x = torch.randn(batch_size, num_channels, im_size, im_size) + 10.0
        out1 = batchnorm1(x)
        out2 = batchnorm2(x)
      
       # print(out2.shape)
        assert (batchnorm1.running_mean - batchnorm2.running_mean.squeeze()).abs().max() < 1e-5,             'running mean is incorrect (%s mode)' % ('train' if mode else 'eval')
        assert (batchnorm1.running_var - batchnorm2.running_var.squeeze()).abs().max() < 1e-5,             'running variance is incorrect (%s mode)' % ('train' if mode else 'eval')
        assert (out1 - out2).abs().max() < 5e-3,             'normalized output is incorrect (%s mode)' % ('train' if mode else 'eval')
print('All OK!')

