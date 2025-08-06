import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(
            self.target_layer.register_backward_hook(backward_hook)
        )

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def __call__(self, input_tensor, gender_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor, gender_tensor)

        output.sum().backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        grad_cam_map = torch.sum(weights * self.activations, dim=1)
        grad_cam_map = F.relu(grad_cam_map)

        grad_cam_map_min = grad_cam_map.min(dim=1, keepdim=True)[0].min(
            dim=2, keepdim=True
        )[0]
        grad_cam_map_max = grad_cam_map.max(dim=1, keepdim=True)[0].max(
            dim=2, keepdim=True
        )[0]
        grad_cam_map = (grad_cam_map - grad_cam_map_min) / (
            grad_cam_map_max - grad_cam_map_min + 1e-8
        )

        return grad_cam_map.cpu().numpy()
