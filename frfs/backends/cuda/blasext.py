# -*- coding: utf-8 -*-

import numpy as np
import pycuda.driver as cuda
from pycuda.gpuarray import GPUArray
from pycuda.reduction import ReductionKernel

from frfs.backends.cuda.provider import CUDAKernelProvider, get_grid_for_block
from frfs.backends.base import ComputeKernel
from frfs.nputil import npdtype_to_ctype


class CUDABlasExtKernels(CUDAKernelProvider):
    def axnpby(self, *arr, subdims=None):
        if any(arr[0].traits != x.traits for x in arr[1:]):
            raise ValueError('Incompatible matrix types')

        nv = len(arr)
        nrow, ldim, dtype = arr[0].traits
        ncola, ncolb = arr[0].ioshape[1:]

        # Render the kernel template
        src = self.backend.lookup.get_template('axnpby').render(
            subdims=subdims or range(ncola), ncola=ncola, nv=nv
        )

        # Build the kernel
        kern = self._build_kernel('axnpby', src,
                                  [np.int32]*3 + [np.intp]*nv + [dtype]*nv)

        # Determine the grid/block
        block = (128, 1, 1)
        grid = get_grid_for_block(block, ncolb, nrow)

        class AxnpbyKernel(ComputeKernel):
            def run(self, queue, *consts):
                args = list(arr) + list(consts)

                # changing from prepared_async_call (queue.cuda_stream_comp,)
                kern.prepared_call(grid, block,
                                         nrow, ncolb, ldim, *args)

        return AxnpbyKernel()

    def axnpby_dgfs_full(self, *arr, subdims=None):
        if any(arr[0].traits != x.traits for x in arr[1:]):
            raise ValueError('Incompatible matrix types')

        nv = len(arr)
        nrow, ldim, dtype = arr[0].traits
        ncola, ncolb = arr[0].ioshape[1:]
        size = nrow*ldim

        # Render the kernel template
        src = self.backend.lookup.get_template('axnpby_dgfs_full').render(
            nv=nv
        )

        # Build the kernel
        kern = self._build_kernel('axnpby_dgfs_full', src,
                                  [np.int32]*2 + [np.intp]*nv + [dtype]*nv)

        # Determine the grid/block
        block = (128, 1, 1)
        grid = get_grid_for_block(block, size)

        class AxnpbyDGFSFullKernel(ComputeKernel):
            def run(self, queue, *consts):
                args = list(arr) + list(consts)

                # changing from prepared_async_call (queue.cuda_stream_comp,)
                kern.prepared_call(grid, block,
                                         size, *args)

        return AxnpbyDGFSFullKernel()

    def copy_to_reg(self, *arr, subdims=None):
        nrow, ldimr, dtype = arr[1].traits
        _, ldimf, _ = arr[0].traits
        ncolar, ncolb = arr[1].ioshape[1:]
        ncolaf, _ = arr[0].ioshape[1:]

        # Render the kernel template
        src = self.backend.lookup.get_template('copy_dgfs').render(
            subdims=subdims or range(ncolar), ncola0=ncolaf, ncola1=ncolar
        )

        #print(arr[0].traits)
        #print(arr[0].ioshape)
        #print(arr[1].traits)
        #print(arr[1].ioshape)
        #raise ValueError('copy_to_reg')

        # Build the kernel
        kern = self._build_kernel('copy_dgfs', src,
            [np.int32]*4 + [np.intp]*2 + [np.int32]*4)

        # Determine the grid/block
        block = (128, 1, 1)
        grid = get_grid_for_block(block, ncolb, nrow)

        class CopyToDGFSKernel(ComputeKernel):
            def run(self, queue, *consts):
                args = list(arr) + list(consts)

                kern.prepared_call(grid, block, 
                                         nrow, ncolb, ldimf, ldimr, *args)

        return CopyToDGFSKernel()


    def copy_from_reg(self, *arr, subdims=None):
        nrow, ldimr, dtype = arr[0].traits
        _, ldimf, _ = arr[1].traits
        ncolar, ncolb = arr[0].ioshape[1:]
        ncolaf, _ = arr[1].ioshape[1:]

        # Render the kernel template
        src = self.backend.lookup.get_template('copy_dgfs').render(
            subdims=subdims or range(ncolar), ncola0=ncolar, ncola1=ncolaf
        )

        # Build the kernel
        kern = self._build_kernel('copy_dgfs', src,
            [np.int32]*4 + [np.intp]*2 + [np.int32]*4)

        # Determine the grid/block
        block = (128, 1, 1)
        grid = get_grid_for_block(block, ncolb, nrow)

        class CopyFromDGFSKernel(ComputeKernel):
            def run(self, queue, *consts):
                args = list(arr) + list(consts)

                kern.prepared_call(grid, block, 
                                         nrow, ncolb, ldimr, ldimf, *args)

        return CopyFromDGFSKernel()

    def copy(self, dst, src):
        if dst.traits != src.traits:
            raise ValueError('Incompatible matrix types')

        class CopyKernel(ComputeKernel):
            def run(self, queue):
                cuda.memcpy_dtod_async(dst.data, src.data, dst.nbytes,
                                       stream=queue.cuda_stream_comp)

        return CopyKernel()

    def errest(self, x, y, z, *, norm):
        if x.traits != y.traits != z.traits:
            raise ValueError('Incompatible matrix types')

        nrow, ldim, dtype = x.traits
        ncola, ncolb = x.ioshape[1:]

        # Reduction block dimensions
        block = (128, 1, 1)

        # Determine the grid size
        grid = get_grid_for_block(block, ncolb)

        # Empty result buffer on host with shape (nvars, nblocks)
        err_host = cuda.pagelocked_empty((ncola, grid[0]), dtype, 'C')

        # Device memory allocation
        err_dev = cuda.mem_alloc(err_host.nbytes)

        # Get the kernel template
        src = self.backend.lookup.get_template('errest').render(
            norm=norm, ncola=ncola, sharesz=block[0]
        )

        # Build the reduction kernel
        rkern = self._build_kernel(
            'errest', src, [np.int32]*3 + [np.intp]*4 + [dtype]*2
        )

        # Norm type
        reducer = np.max if norm == 'uniform' else np.sum

        class ErrestKernel(ComputeKernel):
            @property
            def retval(self):
                return reducer(err_host, axis=1)

            def run(self, queue, atol, rtol):
                rkern.prepared_async_call(grid, block, queue.cuda_stream_comp,
                                          nrow, ncolb, ldim, err_dev, x, y, z,
                                          atol, rtol)
                cuda.memcpy_dtoh_async(err_host, err_dev,
                                       queue.cuda_stream_comp)

        return ErrestKernel()
