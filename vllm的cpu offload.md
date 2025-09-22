# CPU Offload

由于大模型的参数量实在太大，所以如果想在单机上运行较大的模型，那么有一个可选的方案就是CPU Offload。

vllm也实现了一个简单的CPU Offload机制，可以通过`--cpu-offload-gb`启用。

> 官方文档：https://docs.vllm.ai/en/latest/getting_started/examples/basic.html#cpu-offload

主要是通过这个[PR](https://github.com/vllm-project/vllm/pull/6496)，添加了一个function叫`maybe_offload_to_cpu`：

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

    # 对于CPU来说，在CpuPlatform类中明确返回False
    # 但是在torch.empty_strided()接口中明确pin_memory只在CPU上受支持
    # 所以这里需要后续去深入看一下
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

- [model_executor/models/utils.py#L556-L560](https://github.com/vllm-project/vllm/blob/82fbeae92b86e404829a01441334a9505e8b190d/vllm/model_executor/models/utils.py#L556-L560)

```python
modules = torch.nn.ModuleList(
    [PPMissingLayer() for _ in range(start_layer)] + [
        maybe_offload_to_cpu(layer_fn(prefix=f"{prefix}.{idx}"))
        for idx in range(start_layer, end_layer)] + [
        PPMissingLayer() for _ in range(end_layer, num_hidden_layers)]
)
```

CPU Offload的流程可以概括为：

1. 将传入的`cpu_offload_gb`读取为`_CPU_OFFLOAD_MAX_BYTES`。
1. 然后在构建Module时对每个Layer里面的参数从前往后依次塞到CPU的pin_memory上，并累加参数大小，直到超过用户配置的大小。
1. 替换对应Module的`forward`函数，新的forward函数增加的功能就是每次forward的时候将CPU上的参数复制到GPU上，计算完毕后释放

值得学习的是：这里的替换逻辑非常的高超，通过临时替换forward函数配合`functional_call`函数，巧妙的解决了原始forward函数调用的问题。