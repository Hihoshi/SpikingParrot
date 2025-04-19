import torch
import math


class ATan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, alpha=2.0):
        ctx.save_for_backward(input_)
        ctx.alpha = alpha
        out = (input_ > 0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        alpha = ctx.alpha
        grad = (
            alpha
            / (2 * (1 + (math.pi / 2 * alpha * input_).pow(2)))
            * grad_output
        )
        return grad, None


class BinarizeLayer(torch.nn.Module):
    def __init__(self, alpha=2.0):
        super(BinarizeLayer, self).__init__()
        self.alpha = alpha

    def forward(self, input_):
        return ATan.apply(input_, self.alpha)


if __name__ == "__main__":
    # Example usage:
    # Create an instance of the BinarizeLayer
    atan_layer = BinarizeLayer()

    # Generate some sample input data
    input_data = torch.tensor([-0.5, 0.2, 0.8], requires_grad=True)

    # Pass the input data through the layer
    output = atan_layer(input_data)
    print(output)

    # Generate some sample gradients
    grad_output = torch.tensor([1.0, 1.0, 1.0])

    # Back propagate the gradients
    output.backward(grad_output)

    # Access the gradients of the input data
    print(input_data.grad)
