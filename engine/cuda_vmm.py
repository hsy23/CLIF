"""CUDA VMM tensors exportable across processes through POSIX file descriptors."""

from __future__ import annotations

import math
import os
import uuid
from dataclasses import dataclass
from multiprocessing import reduction
from typing import Any, Sequence

import torch
from cuda.bindings import driver as cuda


def check_cuda(result):
    error, *values = result
    if error != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver call failed: {error}")
    if not values:
        return None
    return values[0] if len(values) == 1 else tuple(values)


def _allocation_properties():
    properties = cuda.CUmemAllocationProp()
    properties.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    properties.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    properties.location.id = 0
    properties.requestedHandleTypes = (
        cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    )
    return properties


def _access_descriptor():
    descriptor = cuda.CUmemAccessDesc()
    descriptor.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    descriptor.location.id = 0
    descriptor.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    return descriptor


class _CudaArray:
    def __init__(self, pointer: int, elements: int, typestr: str):
        self.__cuda_array_interface__ = {
            "shape": (elements,),
            "strides": None,
            "typestr": typestr,
            "data": (pointer, False),
            "version": 3,
            "stream": 1,
        }


def _storage_description(dtype: torch.dtype) -> tuple[str, torch.dtype]:
    if dtype == torch.bfloat16:
        return "<u2", torch.uint16
    descriptions = {
        torch.float32: ("<f4", torch.float32),
        torch.float16: ("<f2", torch.float16),
        torch.int64: ("<i8", torch.int64),
        torch.int32: ("<i4", torch.int32),
        torch.uint8: ("|u1", torch.uint8),
    }
    if dtype not in descriptions:
        raise TypeError(f"unsupported VMM tensor dtype: {dtype}")
    return descriptions[dtype]


def _tensor_from_pointer(pointer: int, shape: Sequence[int], dtype: torch.dtype):
    elements = math.prod(shape)
    typestr, storage_dtype = _storage_description(dtype)
    wrapper = _CudaArray(pointer, elements, typestr)
    tensor = torch.as_tensor(wrapper, device="cuda")
    if dtype == torch.bfloat16:
        tensor = tensor.view(torch.bfloat16)
    elif tensor.dtype != storage_dtype:
        raise RuntimeError(f"CUDA array produced {tensor.dtype}, expected {storage_dtype}")
    return wrapper, tensor.reshape(tuple(shape))


def _map_handle(handle, size: int, shape: Sequence[int], dtype: torch.dtype):
    address = check_cuda(cuda.cuMemAddressReserve(size, 0, 0, 0))
    check_cuda(cuda.cuMemMap(address, size, 0, handle, 0))
    check_cuda(cuda.cuMemSetAccess(address, size, [_access_descriptor()], 1))
    wrapper, tensor = _tensor_from_pointer(int(address), shape, dtype)
    return address, wrapper, tensor


@dataclass
class ImportedCudaTensor:
    allocation_id: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    size: int
    handle: Any
    address: Any
    wrapper: Any
    tensor: torch.Tensor

    @classmethod
    def from_spec(cls, spec: dict) -> "ImportedCudaTensor":
        check_cuda(cuda.cuInit(0))
        fd = spec["fd"].detach()
        try:
            handle = check_cuda(
                cuda.cuMemImportFromShareableHandle(
                    fd,
                    cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                )
            )
        finally:
            os.close(fd)
        dtype = getattr(torch, spec["dtype"])
        shape = tuple(spec["shape"])
        address, wrapper, tensor = _map_handle(handle, spec["size"], shape, dtype)
        return cls(
            allocation_id=spec["allocation_id"],
            shape=shape,
            dtype=dtype,
            size=spec["size"],
            handle=handle,
            address=address,
            wrapper=wrapper,
            tensor=tensor,
        )

    def close(self) -> None:
        check_cuda(cuda.cuMemUnmap(self.address, self.size))
        check_cuda(cuda.cuMemAddressFree(self.address, self.size))
        check_cuda(cuda.cuMemRelease(self.handle))


class ExportableCudaTensor:
    def __init__(self, shape: Sequence[int], dtype: torch.dtype, *, allocation_id: str | None = None):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for exportable VMM tensors")
        torch.cuda.set_device(0)
        check_cuda(cuda.cuInit(0))
        self.shape = tuple(int(dimension) for dimension in shape)
        self.dtype = dtype
        self.allocation_id = allocation_id or str(uuid.uuid4())
        requested_size = math.prod(self.shape) * torch.empty((), dtype=dtype).element_size()
        properties = _allocation_properties()
        granularity = check_cuda(
            cuda.cuMemGetAllocationGranularity(
                properties,
                cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            )
        )
        self.size = ((requested_size + granularity - 1) // granularity) * granularity
        self.handle = check_cuda(cuda.cuMemCreate(self.size, properties, 0))
        self.address, self.wrapper, self.tensor = _map_handle(
            self.handle,
            self.size,
            self.shape,
            self.dtype,
        )
        self.fd = check_cuda(
            cuda.cuMemExportToShareableHandle(
                self.handle,
                cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                0,
            )
        )

    def export_spec(self) -> dict:
        return {
            "allocation_id": self.allocation_id,
            "shape": self.shape,
            "dtype": str(self.dtype).removeprefix("torch."),
            "size": self.size,
            "fd": reduction.DupFd(self.fd),
        }

    def close(self) -> None:
        os.close(self.fd)
        check_cuda(cuda.cuMemUnmap(self.address, self.size))
        check_cuda(cuda.cuMemAddressFree(self.address, self.size))
        check_cuda(cuda.cuMemRelease(self.handle))
