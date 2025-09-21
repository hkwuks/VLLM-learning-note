# CPU Offload

由于大模型的参数量实在太大，所以如果想在单机上运行较大的模型，那么有一个可选的方案就是CPU Offload。

vllm也实现了一个简单的CPU Offload机制，可以通过`--cpu-offload-gb`启用。

> 官方文档：https://docs.vllm.ai/en/latest/getting_started/examples/basic.html#cpu-offload

主要是通过这个[PR](https://github.com/vllm-project/vllm/pull/6496)，添加了一个func叫`maybe_offload_to_cpu`：

- [model_executor/models/utils.py#L487-L540](https://github.com/vllm-project/vllm/blob/82fbeae92b86e404829a01441334a9505e8b190d/vllm/model_executor/models/utils.py#L487-L540)

```python
import torch

def maybe_offload_to_cpu(module: torch.nn.Module) -> torch.nn.Module:
    device = next(module.parameters()).device

    if device == torch.device('cpu'):
        return module

    global _CPU_OFFLOAD_MAX_BYTES, _CPU_OFFLOAD_BYTES
    if _CPU_OFFLOAD_BYTES >= _CPU_OFFLOAD_MAX_BYTES:
        return module

    # 对于CPU来说，不支持pin_memory，在CpuPlatform类中明确返回False
    # 这里应该是在判断GPU是否支持pin_memory，以便使用torch的pin_memory
    pin_memory = is_pin_memory_available()

    # offload parameters to CPU
    # use pin_memory if possible, which helps cudagraph capture speed
    offloaded_parameters = False
    for p in module.parameters():
        if _CPU_OFFLOAD_BYTES >= _CPU_OFFLOAD_MAX_BYTES:
            # 他是做了一个单参数的offload
            # 所以可能一个module中有的参数offload了，有的没有
            break

        # `torch.empty_like` does not support `pin_memory` argument
        cpu_data = torch.empty_strided(size=p.data.size(),
                                       stride=p.data.stride(),
                                       dtype=p.data.dtype,
                                       layout=p.data.layout,
                                       device='cpu',
                                       pin_memory=pin_memory)
        cpu_data.copy_(p.data)
        p.data = cpu_data
        _CPU_OFFLOAD_BYTES += p.data.numel() * p.data.element_size()
        offloaded_parameters = True

    if offloaded_parameters:
        original_forward = module.forward

        def forward(*args, **kwargs):
            module.forward = original_forward
            device_state = {
                # here we blindly call `to(device)`
                # if the parameter is already on the device, it will be a no-op
                k: v.to(device, non_blocking=True)
                for k,v in module.state_dict().items()
            }

            output = functional_call(module,
                                     device_state,
                                     args=args,
                                     kwargs=kwargs)
            module.forward = forward
            return output

        module.forward = forward

    return module
```

这个函数只有一个地方调用：

