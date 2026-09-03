"""Cross-process CUDA VMM sharing probe for WSL.

This avoids the legacy cudaIpcOpenMemHandle path used by torch.multiprocessing.
"""

from __future__ import annotations

import json
import os
from multiprocessing import reduction

import torch
import torch.multiprocessing as mp
from cuda.bindings import driver as cuda


def check(result):
    error, *values = result
    if error != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver call failed: {error}")
    if not values:
        return None
    return values[0] if len(values) == 1 else tuple(values)


def allocation_properties():
    properties = cuda.CUmemAllocationProp()
    properties.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    properties.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    properties.location.id = 0
    properties.requestedHandleTypes = (
        cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    )
    return properties


def access_descriptor():
    descriptor = cuda.CUmemAccessDesc()
    descriptor.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    descriptor.location.id = 0
    descriptor.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    return descriptor


class CudaArray:
    def __init__(self, pointer: int, elements: int):
        self.__cuda_array_interface__ = {
            "shape": (elements,),
            "strides": None,
            "typestr": "<f4",
            "data": (pointer, False),
            "version": 3,
            "stream": 1,
        }


def map_allocation(handle, size: int, elements: int):
    address = check(cuda.cuMemAddressReserve(size, 0, 0, 0))
    check(cuda.cuMemMap(address, size, 0, handle, 0))
    check(cuda.cuMemSetAccess(address, size, [access_descriptor()], 1))
    tensor = torch.as_tensor(CudaArray(int(address), elements), device="cuda")
    return address, tensor


def child_main(duplicated_fd, size: int, elements: int, result_queue) -> None:
    torch.cuda.set_device(0)
    check(cuda.cuInit(0))
    fd = duplicated_fd.detach()
    try:
        handle = check(
            cuda.cuMemImportFromShareableHandle(
                fd,
                cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
            )
        )
        address, tensor = map_allocation(handle, size, elements)
        before = float(tensor.sum().cpu())
        tensor.add_(10)
        torch.cuda.synchronize()
        result_queue.put({"before": before, "after": float(tensor.sum().cpu())})
        check(cuda.cuMemUnmap(address, size))
        check(cuda.cuMemAddressFree(address, size))
        check(cuda.cuMemRelease(handle))
    finally:
        os.close(fd)


def main() -> None:
    torch.cuda.set_device(0)
    check(cuda.cuInit(0))
    properties = allocation_properties()
    granularity = check(
        cuda.cuMemGetAllocationGranularity(
            properties,
            cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
        )
    )
    elements = 8
    requested = elements * 4
    size = ((requested + granularity - 1) // granularity) * granularity
    handle = check(cuda.cuMemCreate(size, properties, 0))
    address, tensor = map_allocation(handle, size, elements)
    tensor.copy_(torch.arange(elements, dtype=torch.float32, device="cuda"))
    torch.cuda.synchronize()
    fd = check(
        cuda.cuMemExportToShareableHandle(
            handle,
            cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
            0,
        )
    )

    context = mp.get_context("spawn")
    result_queue = context.SimpleQueue()
    process = context.Process(
        target=child_main,
        args=(reduction.DupFd(fd), size, elements, result_queue),
    )
    process.start()
    child_result = result_queue.get()
    process.join(60)
    torch.cuda.synchronize()
    result = {
        **child_result,
        "parent_after": float(tensor.sum().cpu()),
        "child_exitcode": process.exitcode,
        "granularity": granularity,
    }
    print(json.dumps(result))

    os.close(fd)
    check(cuda.cuMemUnmap(address, size))
    check(cuda.cuMemAddressFree(address, size))
    check(cuda.cuMemRelease(handle))


if __name__ == "__main__":
    main()
