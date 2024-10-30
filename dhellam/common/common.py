import torch
def get_cublas_workspace_size_bytes() -> None:
    """Return 32 MiB if using hopper, 4 MiB for all other architectures."""
    if torch.cuda.get_device_properties(torch.cuda.current_device()).major >= 9:
        return 33_554_432
    return 4_194_304
def get_workspace() -> torch.Tensor:
    """Returns workspace for cublas."""
    
    _cublas_workspace = torch.empty(
        get_cublas_workspace_size_bytes(), dtype=torch.uint8, device=f"cuda:{torch.cuda.current_device()}"
    )
    return _cublas_workspace

def print_rank0(*args)->None:
    if not torch.distributed.is_initialized() or torch.distributed.get_rank()==0:
        print(*args)

def _get_qkv_layout(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        qkv_format: str = 'sbhd',
    ) -> str:
    """Get qkv layout.

    Parameters
    ----------
    q: torch.Tensor
        Query tensor.
    k: torch.Tensor
        Key tensor.
    v: torch.Tensor
        Value tensor.
    qkv_format: str, default = `sbhd`
        Dimension format for `q`, `k` and `v`, {`sbhd`, `bshd`, `thd`}. `s` stands for
        the sequence length dimension, `b` batch size, `h` the number of attention heads,
        `d` head size, and `t` the total number of sequences in a batch, i.e.
        `t = sum(s_i) for i = 0...b-1`.

    Returns
    ----------
    qkv_layout: str
       Memory layout of `q`, `k` and `v`. Each `qkv_format` can be mapped to one of five
       memory layouts. For example, `sb3hd` means `q`, `k`, `v` are created as one chunk
       of memory and that they are interleaved in the `2`nd dimension. `sbhd_sbh2d` means
       `q` and `kv` are created in two chunks and that `q` itself is contiguous and `k`, `v`
       are interleaved with each other in the `3`rd dimension, `k = kv[:,:,:,0,:]` and
       `v = kv[:,:,:,1,:]`.
       Mapping:
       `sbhd`: {`sb3hd`, `sbh3d`, `sbhd_sb2hd`, `sbhd_sbh2d`, `sbhd_sbhd_sbhd`}
       `bshd`: {`bs3hd`, `bsh3d`, `bshd_bs2hd`, `bshd_bsh2d`, `bshd_bshd_bshd`}
       `thd` : {`t3hd`, `th3d`, `thd_t2hd`, `thd_th2d`, `thd_thd_thd`}
    """

    check_last_dim_contiguous = all(x.stride(-1) == 1 for x in [q, k, v])
    assert check_last_dim_contiguous, "q, k and v must have stride 1 in their last dimension!"

    def run_iteratively(q, k, v):
        data_ptr = q.untyped_storage().data_ptr()
        check_ptrs_qkv = all(x.untyped_storage().data_ptr() == data_ptr for x in [q, k, v])
        data_ptr = k.untyped_storage().data_ptr()
        check_ptrs_kv = all(x.untyped_storage().data_ptr() == data_ptr for x in [k, v])

        stride = q.stride()
        check_strides_qkv = all(stride == x.stride() for x in [q, k, v])
        stride = k.stride()
        check_strides_kv = all(stride == x.stride() for x in [k, v])

        shape = q.shape
        check_shapes_qkv = all(shape == x.shape for x in [q, k, v])
        shape = k.shape
        check_shapes_kv = all(shape == x.shape for x in [k, v])

        last_dim_size = q.shape[-1]
        check_last_dim_offsets_qkv = all(i * last_dim_size == x.storage_offset()
                            for i, x in enumerate([q, k, v]))
        last_dim_size = k.shape[-1]
        check_last_dim_offsets_kv = all(i * last_dim_size == x.storage_offset()
                            for i, x in enumerate([k, v]))

        last_two_dims_size = q.shape[-1] * q.shape[-2]
        check_last_two_dims_offsets_qkv = all(i * last_two_dims_size == x.storage_offset()
                            for i, x in enumerate([q, k, v]))
        last_two_dims_size = k.shape[-1] * k.shape[-2]
        check_last_two_dims_offsets_kv = all(i * last_two_dims_size == x.storage_offset()
                            for i, x in enumerate([k, v]))

        if (check_ptrs_qkv and check_strides_qkv and check_shapes_qkv
            and check_last_two_dims_offsets_qkv
            and not check_last_dim_offsets_qkv):
            # sb3hd, bs3hd, t3hd
            qkv_layout = qkv_format[:-2] + '3' + qkv_format[-2:]
        elif (check_ptrs_qkv and check_strides_qkv and check_shapes_qkv
            and check_last_dim_offsets_qkv):
            # sbh3d, bsh3d, th3d
            qkv_layout = qkv_format[:-1] + '3' + qkv_format[-1:]
        elif (check_ptrs_kv and check_strides_kv and check_shapes_kv
            and check_last_two_dims_offsets_kv
            and not check_last_dim_offsets_kv):
            # sbhd_sb2hd, bshd_bs2hd, thd_t2hd
            qkv_layout = qkv_format + '_' + qkv_format[:-2] + '2' + qkv_format[-2:]
        elif (check_ptrs_kv and check_strides_kv and check_shapes_kv
            and check_last_dim_offsets_kv):
            # sbhd_sbh2d, bshd_bsh2d, thd_th2d
            qkv_layout = qkv_format + '_' + qkv_format[:-1] + '2' + qkv_format[-1:]
        elif check_strides_kv and check_shapes_kv:
            # sbhd_sbhd_sbhd, bshd_bshd_bshd, thd_thd_thd
            qkv_layout = '_'.join(list([qkv_format])*3)
        else:
            qkv_layout = 'not_supported'

        return qkv_layout

    qkv_layout = run_iteratively(q, k, v)
    if qkv_layout == 'not_supported':
        # force q,k,v to be contiguous and run get_layout again
        q, k, v = [x.contiguous() for x in [q, k, v]]
        qkv_layout = run_iteratively(q, k, v)
    if qkv_layout == 'not_supported':
        raise Exception("The provided qkv memory layout is not supported!")

    return qkv_layout, q, k, v
