#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from torch.autograd import Function

class ReverseLayerF(Function):
    
    @staticmethod
    def forward(ctx, input_, alpha):
        ctx.alpha = alpha
        ctx.save_for_backward(input_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output*ctx.alpha
        return grad_input, None


"""
def forward(ctx, x, alpha):
        ctx.alpha = alpha
        
        return x.view_as(x)
    
    def backward(ctx, grad_output):
        output = grad_output.neg()*ctx.alpha
        
        return output, None
        """