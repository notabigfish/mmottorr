from __future__ import annotations

import functools
import torch

# Basis bitmasks for Cl(3,0,1): e0,e1,e2,e3 -> bits 0..3
BLADE_NAMES = [
    "1", "e0", "e1", "e01", "e2", "e02", "e12", "e012",
    "e3", "e03", "e13", "e013", "e23", "e023", "e123", "e0123",
]

# even motor basis order: [1, e23, e31, e12, e01, e02, e03, e0123]
# canonical sorted masks use e13 instead of e31; we map e31->e13 with sign in se3 mapping.
EVEN_MOTOR_INDICES = [0, 12, 10, 6, 3, 5, 9, 15]
_METRIC = [0.0, 1.0, 1.0, 1.0]


def blade_grade(mask: int) -> int:
    return int(mask.bit_count())


def reverse_sign(mask: int) -> int:
    k = blade_grade(mask)
    return -1 if ((k * (k - 1) // 2) % 2) else 1


def _basis_gp(a_mask: int, b_mask: int) -> tuple[int, float]:
    sign = 1.0

    for i in range(4):
        if (a_mask >> i) & 1:
            lower_bits_in_b = b_mask & ((1 << i) - 1)
            if (lower_bits_in_b.bit_count() % 2) == 1:
                sign *= -1.0

    repeated = a_mask & b_mask
    for i in range(4):
        if (repeated >> i) & 1:
            m = _METRIC[i]
            if m == 0.0:
                return 0, 0.0
            sign *= m

    out_mask = a_mask ^ b_mask
    return out_mask, sign


def build_geometric_product_table(
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    out_index = torch.zeros((16, 16), dtype=torch.long, device=device)
    sign = torch.zeros((16, 16), dtype=dtype, device=device)

    for i in range(16):
        for j in range(16):
            om, s = _basis_gp(i, j)
            out_index[i, j] = int(om)
            sign[i, j] = float(s)

    return out_index, sign


@functools.lru_cache(maxsize=16)
def _cached_tables(device_str: str, dtype_str: str):
    device = torch.device(device_str)
    dtype = getattr(torch, dtype_str)
    return build_geometric_product_table(device=device, dtype=dtype)


def _get_tables(device: torch.device, dtype: torch.dtype):
    return _cached_tables(str(device), str(dtype).split(".")[-1])


def geometric_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.shape[-1] != 16 or b.shape[-1] != 16:
        raise ValueError("geometric_product expects [...,16] tensors")

    a, b = torch.broadcast_tensors(a, b)
    flat_a = a.reshape(-1, 16)
    flat_b = b.reshape(-1, 16)

    out_idx, sgn = _get_tables(flat_a.device, flat_a.dtype)
    prod = (flat_a[:, :, None] * flat_b[:, None, :]) * sgn[None, :, :]

    res = torch.zeros((flat_a.shape[0], 16), dtype=flat_a.dtype, device=flat_a.device)
    res.scatter_add_(1, out_idx.reshape(-1)[None, :].expand(flat_a.shape[0], -1), prod.reshape(flat_a.shape[0], -1))
    return res.reshape(a.shape)


def reverse(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] != 16:
        raise ValueError("reverse expects [...,16]")
    signs = torch.tensor([reverse_sign(i) for i in range(16)], dtype=x.dtype, device=x.device)
    return x * signs


def scalar_part(x: torch.Tensor) -> torch.Tensor:
    return x[..., 0]


def even_motor_to_full(m8: torch.Tensor) -> torch.Tensor:
    if m8.shape[-1] != 8:
        raise ValueError("even_motor_to_full expects [...,8]")
    M = torch.zeros(*m8.shape[:-1], 16, dtype=m8.dtype, device=m8.device)
    for i, idx in enumerate(EVEN_MOTOR_INDICES):
        M[..., idx] = m8[..., i]
    return M


def full_to_even_motor(M: torch.Tensor) -> torch.Tensor:
    if M.shape[-1] != 16:
        raise ValueError("full_to_even_motor expects [...,16]")
    return torch.stack([M[..., idx] for idx in EVEN_MOTOR_INDICES], dim=-1)


def normalize_motor(M: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    G = geometric_product(M, reverse(M))
    s = scalar_part(G).abs().clamp_min(eps).sqrt()
    return M / s.unsqueeze(-1)


def fix_motor_sign(M: torch.Tensor) -> torch.Tensor:
    s = torch.where(M[..., :1] < 0, -1.0, 1.0)
    return M * s


def motor_inverse(M: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return reverse(normalize_motor(M, eps=eps))


def quaternion_from_matrix(R: torch.Tensor) -> torch.Tensor:
    # Stable branchless-ish conversion from projected rotation matrix
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    qw = torch.sqrt((1.0 + tr).clamp_min(1e-8)) * 0.5
    qx = (R[..., 2, 1] - R[..., 1, 2]) / (4.0 * qw.clamp_min(1e-8))
    qy = (R[..., 0, 2] - R[..., 2, 0]) / (4.0 * qw.clamp_min(1e-8))
    qz = (R[..., 1, 0] - R[..., 0, 1]) / (4.0 * qw.clamp_min(1e-8))
    q = torch.stack([qw, qx, qy, qz], dim=-1)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    q = torch.where(q[..., :1] < 0, -q, q)
    return q


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def se3_to_motor(T: torch.Tensor) -> torch.Tensor:
    if T.shape[-2:] != (4, 4):
        raise ValueError("se3_to_motor expects [...,4,4]")
    R = T[..., :3, :3]
    t = T[..., :3, 3]

    qr = quaternion_from_matrix(R)
    tq = torch.cat([torch.zeros_like(t[..., :1]), t], dim=-1)
    qd = 0.5 * quaternion_multiply(tq, qr)

    # Mapping to even motor coefficients [1,e23,e31,e12,e01,e02,e03,e0123].
    # Canonical stored basis uses e13 for e31, so we encode e31 with sign flip into e13 slot.
    # m8 = torch.stack([
    #     qr[..., 0],   # 1
    #     qr[..., 1],   # e23
    #     -qr[..., 2],  # e31 -> -e13 in canonical ordering
    #     qr[..., 3],   # e12
    #     -qd[..., 1],  # e01
    #     -qd[..., 2],  # e02
    #     -qd[..., 3],  # e03
    #     -qd[..., 0],  # e0123
    # ], dim=-1)
    m8 = torch.stack([
        qr[..., 0],   # 1
        -qr[..., 1],  # e23 (修改：加负号)
        qr[..., 2],   # e31 -> -e13 (修改：去掉负号，-q_y * e31 = q_y * e13)
        -qr[..., 3],  # e12 (修改：加负号)
        -qd[..., 1],  # e01 (同步保持上文对偶部分的负号修复)
        -qd[..., 2],  # e02 
        -qd[..., 3],  # e03 
        -qd[..., 0],  # e0123 
    ], dim=-1)
    M = even_motor_to_full(m8)
    M = normalize_motor(M)
    M = fix_motor_sign(M)
    return M


def point_from_xyz(xyz: torch.Tensor) -> torch.Tensor:
    if xyz.shape[-1] != 3:
        raise ValueError("point_from_xyz expects [...,3]")
    x, y, z = xyz.unbind(dim=-1)
    P = torch.zeros(*xyz.shape[:-1], 16, dtype=xyz.dtype, device=xyz.device)
    # P = e123 + x*e032 + y*e013 + z*e021
    # Canonical masks are sorted: e023(mask13), e013(mask11), e012(mask7).
    # e032 = -e023, e021 = -e012.
    P[..., 14] = 1.0          # e123
    P[..., 13] = -x           # e032 -> -e023
    P[..., 11] = y            # e013
    P[..., 7] = -z            # e021 -> -e012
    return P


def xyz_from_point(P: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if P.shape[-1] != 16:
        raise ValueError("xyz_from_point expects [...,16]")
    w = P[..., 14].clamp_min(eps)
    x = -P[..., 13] / w
    y = P[..., 11] / w
    z = -P[..., 7] / w
    return torch.stack([x, y, z], dim=-1)


def sandwich(M: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    Minv = motor_inverse(M)
    return geometric_product(geometric_product(M, X), Minv)


def sandwich_points(M: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
    one_point = xyz.ndim >= 1 and xyz.shape[-1] == 3 and (xyz.ndim == 1 or xyz.shape[-2] != 3)
    if xyz.shape[-1] != 3:
        raise ValueError("sandwich_points expects [...,3] or [...,N,3]")

    if xyz.ndim == 2:
        xyz_in = xyz.unsqueeze(0)  # [1,N,3]
    elif xyz.ndim == 1:
        xyz_in = xyz.view(1, 1, 3)
    else:
        xyz_in = xyz

    if M.shape[-1] != 16:
        raise ValueError("sandwich_points motor must be [...,16]")

    if M.ndim == 1:
        M = M.view(1, 16)

    B = M.shape[0]
    if xyz_in.shape[0] != B:
        if xyz_in.shape[0] == 1:
            xyz_in = xyz_in.expand(B, *xyz_in.shape[1:])
        elif B == 1:
            M = M.expand(xyz_in.shape[0], 16)
        else:
            raise ValueError("batch mismatch between motor and points")

    P = point_from_xyz(xyz_in)
    M_exp = M[:, None, :].expand(P.shape[0], P.shape[1], 16)
    Pp = sandwich(M_exp, P)
    out = xyz_from_point(Pp)

    if xyz.ndim == 1:
        return out[0, 0]
    if xyz.ndim == 2 and M.shape[0] == 1:
        return out[0]
    return out


def motor_to_features(M: torch.Tensor, include_full: bool = False) -> torch.Tensor:
    M = fix_motor_sign(normalize_motor(M))
    return M if include_full else full_to_even_motor(M)


def unit_motor_regularization(M: torch.Tensor) -> torch.Tensor:
    G = geometric_product(M, reverse(M))
    norm_term = (G[..., 0] - 1.0) ** 2
    nonscalar_term = (G[..., 1:] ** 2).sum(dim=-1)
    return (norm_term + nonscalar_term).mean()
