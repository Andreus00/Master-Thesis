import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

import sys
sys.path.append("./ext/gaussian-splatting/")

from scene.colmap_loader import rotmat2qvec, qvec2rotmat


class GaussianTransformUtils:
    @staticmethod
    def translation(xyz, x: float, y: float, z: float):
        if x == 0. and y == 0. and z == 0.:
            return xyz

        return xyz + torch.tensor([[x, y, z]], device=xyz.device)

    @staticmethod
    def rescale(xyz, scaling, factor: float):
        if factor == 1.:
            return xyz, scaling
        scaling += np.log(factor)
        return xyz * factor, scaling

    @staticmethod
    def rx(theta):
        theta = torch.tensor(theta)
        return torch.tensor([[1, 0, 0],
                             [0, torch.cos(theta), -torch.sin(theta)],
                             [0, torch.sin(theta), torch.cos(theta)]], dtype=torch.float)

    @staticmethod
    def ry(theta):
        theta = torch.tensor(theta)
        return torch.tensor([[torch.cos(theta), 0, torch.sin(theta)],
                             [0, 1, 0],
                             [-torch.sin(theta), 0, torch.cos(theta)]], dtype=torch.float)

    @staticmethod
    def rz(theta):
        theta = torch.tensor(theta)
        return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                             [torch.sin(theta), torch.cos(theta), 0],
                             [0, 0, 1]], dtype=torch.float)

    @classmethod
    def rotate_by_euler_angles(cls, xyz, rotation, x: float, y: float, z: float):
        """
        rotate in z-y-x order, radians as unit
        """

        if x == 0. and y == 0. and z == 0.:
            return

        # rotate
        rotation_matrix = cls.rx(x) @ cls.ry(y) @ cls.rz(z)
        xyz, rotation = cls.rotate_by_matrix(
            xyz,
            rotation,
            rotation_matrix.to(xyz),
        )

        return xyz, rotation

    @staticmethod
    def transform_shs(features, rotation_matrix):
        """
        https://github.com/graphdeco-inria/gaussian-splatting/issues/176#issuecomment-2147223570
        """

        try:
            from e3nn import o3
            import einops
            from einops import einsum
        except:
            print("Please run `pip install e3nn einops` to enable SHs rotation")
            return

        if features.shape[1] == 1:
            return features

        features = features.clone()

        shs_feat = features[:, 1:, :]

        ## rotate shs
        P = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=shs_feat.dtype, device=shs_feat.device)  # switch axes: yzx -> xyz
        inversed_P = torch.tensor([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ], dtype=shs_feat.dtype, device=shs_feat.device)
        permuted_rotation_matrix = inversed_P @ rotation_matrix @ P
        rot_angles = o3._rotation.matrix_to_angles(permuted_rotation_matrix.cpu())

        # Construction coefficient
        D_1 = o3.wigner_D(1, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device=shs_feat.device)
        D_2 = o3.wigner_D(2, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device=shs_feat.device)
        D_3 = o3.wigner_D(3, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device=shs_feat.device)

        # rotation of the shs features
        one_degree_shs = shs_feat[:, 0:3]
        one_degree_shs = einops.rearrange(one_degree_shs, 'n shs_num rgb -> n rgb shs_num')
        one_degree_shs = einsum(
            D_1,
            one_degree_shs,
            "... i j, ... j -> ... i",
        )
        one_degree_shs = einops.rearrange(one_degree_shs, 'n rgb shs_num -> n shs_num rgb')
        shs_feat[:, 0:3] = one_degree_shs

        if shs_feat.shape[1] >= 4:
            two_degree_shs = shs_feat[:, 3:8]
            two_degree_shs = einops.rearrange(two_degree_shs, 'n shs_num rgb -> n rgb shs_num')
            two_degree_shs = einsum(
                D_2,
                two_degree_shs,
                "... i j, ... j -> ... i",
            )
            two_degree_shs = einops.rearrange(two_degree_shs, 'n rgb shs_num -> n shs_num rgb')
            shs_feat[:, 3:8] = two_degree_shs

            if shs_feat.shape[1] >= 9:
                three_degree_shs = shs_feat[:, 8:15]
                three_degree_shs = einops.rearrange(three_degree_shs, 'n shs_num rgb -> n rgb shs_num')
                three_degree_shs = einsum(
                    D_3,
                    three_degree_shs,
                    "... i j, ... j -> ... i",
                )
                three_degree_shs = einops.rearrange(three_degree_shs, 'n rgb shs_num -> n shs_num rgb')
                shs_feat[:, 8:15] = three_degree_shs

        return features
    
    @classmethod
    def vectorized_transform_shs(self, _features, _rotation_matrix):
        """
        https://github.com/graphdeco-inria/gaussian-splatting/issues/176#issuecomment-2147223570
        """

        try:
            from e3nn import o3
            import einops
            from einops import einsum
        except:
            print("Please run `pip install e3nn einops` to enable SHs rotation")
            return

        if _features.shape[1] == 1:
            return _features

        _features = _features.clone()
        
        for i in range(_features.shape[0]):
            
            shs_feat = _features[i, 1:, :].unsqueeze(0)

            ## rotate shs
            P = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=shs_feat.dtype, device=shs_feat.device)  # switch axes: yzx -> xyz
            inversed_P = torch.tensor([
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0],
            ], dtype=shs_feat.dtype, device=shs_feat.device)
            permuted_rotation_matrix = inversed_P @ _rotation_matrix[i] @ P
            rot_angles = o3._rotation.matrix_to_angles(permuted_rotation_matrix.cpu())

            # Construction coefficient
            D_1 = o3.wigner_D(1, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device=shs_feat.device)
            D_2 = o3.wigner_D(2, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device=shs_feat.device)
            D_3 = o3.wigner_D(3, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device=shs_feat.device)

            # rotation of the shs features
            one_degree_shs = shs_feat[:, 0:3]
            one_degree_shs = einops.rearrange(one_degree_shs, 'n shs_num rgb -> n rgb shs_num')
            one_degree_shs = einsum(
                D_1,
                one_degree_shs,
                "... i j, ... j -> ... i",
            )
            one_degree_shs = einops.rearrange(one_degree_shs, 'n rgb shs_num -> n shs_num rgb')
            shs_feat[:, 0:3] = one_degree_shs

            if shs_feat.shape[1] >= 4:
                two_degree_shs = shs_feat[:, 3:8]
                two_degree_shs = einops.rearrange(two_degree_shs, 'n shs_num rgb -> n rgb shs_num')
                two_degree_shs = einsum(
                    D_2,
                    two_degree_shs,
                    "... i j, ... j -> ... i",
                )
                two_degree_shs = einops.rearrange(two_degree_shs, 'n rgb shs_num -> n shs_num rgb')
                shs_feat[:, 3:8] = two_degree_shs

                if shs_feat.shape[1] >= 9:
                    three_degree_shs = shs_feat[:, 8:15]
                    three_degree_shs = einops.rearrange(three_degree_shs, 'n shs_num rgb -> n rgb shs_num')
                    three_degree_shs = einsum(
                        D_3,
                        three_degree_shs,
                        "... i j, ... j -> ... i",
                    )
                    three_degree_shs = einops.rearrange(three_degree_shs, 'n rgb shs_num -> n shs_num rgb')
                    shs_feat[:, 8:15] = three_degree_shs
                
            _features[i, 1:, :] = shs_feat.squeeze(0)

        return _features

    @classmethod
    def vectorized_transform_shs_2(self, _features, _rotation_matrices):
        """
        https://github.com/graphdeco-inria/gaussian-splatting/issues/176#issuecomment-2147223570
        """

        try:
            from e3nn import o3
            import einops
            from einops import einsum
        except:
            print("Please run `pip install e3nn einops` to enable SHs rotation")
            return

        if _features.shape[1] == 1:
            return _features

        _features = _features.clone()
        
        shs_feat = _features[:, 1:, :]

        ## rotate shs
        P = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=shs_feat.dtype, device=shs_feat.device)  # switch axes: yzx -> xyz
        inversed_P = torch.tensor([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ], dtype=shs_feat.dtype, device=shs_feat.device)
        
        permuted_rotation_matrices = torch.einsum('ij,bjk,kl->bil', inversed_P, _rotation_matrices, P)
        # matrix to angles
        rot_angles = torch.stack(o3._rotation.matrix_to_angles(permuted_rotation_matrices.cpu()), dim=1)

        # Construction coefficient
        D_1 = o3.wigner_D(1, rot_angles[:, 0], -rot_angles[:, 1], rot_angles[:, 2]).to(device=shs_feat.device)
        D_2 = o3.wigner_D(2, rot_angles[:, 0], -rot_angles[:, 1], rot_angles[:, 2]).to(device=shs_feat.device)
        D_3 = o3.wigner_D(3, rot_angles[:, 0], -rot_angles[:, 1], rot_angles[:, 2]).to(device=shs_feat.device)

        # rotation of the shs features
        one_degree_shs = shs_feat[:, 0:3, :]
        # one_degree_shs = einops.rearrange(one_degree_shs, 'batch shs_num rgb -> batch rgb shs_num')
        one_degree_shs = torch.einsum(
            "bij,bjr->bir",
            D_1,
            one_degree_shs,
        )
        one_degree_shs = einops.rearrange(one_degree_shs, 'batch rgb shs_num -> batch shs_num rgb')
        shs_feat[:, 0:3, :] = one_degree_shs

        if shs_feat.shape[1] >= 4:
            two_degree_shs = shs_feat[:, 3:8, :]
            # two_degree_shs = einops.rearrange(two_degree_shs, 'batch shs_num rgb -> batch rgb shs_num')
            two_degree_shs = torch.einsum(
                'bij,bjr->bir',
                D_2,
                two_degree_shs,
            )
            # two_degree_shs = einops.rearrange(two_degree_shs, 'batch rgb shs_num -> batch shs_num rgb')
            shs_feat[:, 3:8, :] = two_degree_shs

            if shs_feat.shape[1] >= 9:
                three_degree_shs = shs_feat[:, 8:15, :]
                # three_degree_shs = einops.rearrange(three_degree_shs, 'batch shs_num rgb -> batch rgb shs_num')
                three_degree_shs = torch.einsum(
                    'bij,bjr->bir',
                    D_3,
                    three_degree_shs,
                )
                # three_degree_shs = einops.rearrange(three_degree_shs, 'batch rgb shs_num -> batch shs_num rgb')
                shs_feat[:, 8:15, :] = three_degree_shs

        _features[:, 1:, :] = shs_feat

        return _features

    @classmethod
    def rotate_by_wxyz_quaternions(cls, xyz, rotations, features, quaternions: torch.tensor = None, rotation_matrix=None):
        if quaternions is None and rotation_matrix is None:
            raise ValueError("Please provide either quaternions or rotation matrix")
        # convert quaternions to rotation matrix
        if rotation_matrix is not None:
            quaternions = torch.tensor(rotmat2qvec(rotation_matrix.cpu().numpy()), dtype=torch.float, device=xyz.device)
        else:
            rotation_matrix = torch.tensor(qvec2rotmat(quaternions.cpu().numpy()), dtype=torch.float, device=xyz.device)

        if torch.all(quaternions == 0.) or torch.all(quaternions == torch.tensor(
                [1., 0., 0., 0.],
                dtype=quaternions.dtype,
                device=quaternions.device,
        )):
            return xyz, rotations, features

        # rotate xyz
        xyz = torch.matmul(xyz, rotation_matrix.T)
        # rotate gaussian quaternions
        rotations = torch.nn.functional.normalize(cls.quat_multiply(
            rotations,
            quaternions,
        ))

        features = cls.transform_shs(features, rotation_matrix)

        return xyz, rotations, features
    
    @classmethod
    def vectorized_rotmat2qvec(self, R):
        Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.reshape(-1, 9).transpose()
        zeros = np.zeros(Rxx.shape)
        K = np.array([
            [Rxx - Ryy - Rzz, zeros, zeros, zeros],
            [Ryx + Rxy, Ryy - Rxx - Rzz, zeros, zeros],
            [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, zeros],
            [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
        K = K.transpose(2, 0, 1)
        eigvals, eigvecs = np.linalg.eigh(K)
        qvecs = eigvecs[:, [3, 0, 1, 2]]
        qvec = []
        for idx, q in enumerate(qvecs):
            qvec.append(q[:, np.argmax(eigvals[idx])])
        qvec = np.array(qvec)
        mask = qvec[:, 0] < 0
        qvec[mask] *= -1
        return qvec

    @classmethod
    def vectorized_rotate_by_wxyz_quaternions(cls, xyz, rotations, features, rotation_matrix):
        '''
        This variant of the rotation function allows for a mask to be applied to the rotation, and
        a different rotation applied to every gaussian.

        '''
        if rotation_matrix is not None:
            quaternions = torch.tensor(cls.vectorized_rotmat2qvec(rotation_matrix.cpu().numpy()), dtype=torch.float, device=xyz.device)
        else:
            raise ValueError("Please provide a rotation matrix")
        
        if torch.all(quaternions == 0.) or torch.all(quaternions == torch.tensor(
                                                                        [1., 0., 0., 0.], 
                                                                        dtype=quaternions.dtype, 
                                                                        device=quaternions.device)):
            return xyz, rotations, features

        # rotate xyz
        # xyz = torch.einsum("bi, bji -> bj", xyz, rotation_matrix)
        # for b in range(xyz.shape[0]):
        #     xyz[b] = torch.matmul(xyz[b], rotation_matrix[b].T)
        # rotate gaussian quaternions
        rotations = torch.nn.functional.normalize(cls.quat_multiply(
            rotations,
            quaternions,
        ))
        # TODO: Fix vectorized_transform_shs_2
        features = cls.vectorized_transform_shs_2(features, rotation_matrix)
        
        return xyz, rotations, features

    @staticmethod
    def quat_multiply(quaternion0, quaternion1):
        w0, x0, y0, z0 = torch.split(quaternion0, 1, dim=-1)
        w1, x1, y1, z1 = torch.split(quaternion1, 1, dim=-1)
        return torch.cat((
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
        ), dim=-1)

    @classmethod
    def rotate_by_matrix(cls, xyz, rotations, rotation_matrix):
        # rotate xyz
        xyz = torch.matmul(xyz, rotation_matrix.T)

        # rotate via quaternion
        rotations = torch.nn.functional.normalize(cls.quat_multiply(
            rotations,
            torch.tensor([rotmat2qvec(rotation_matrix.cpu().numpy())]).to(xyz),
        ))

        return xyz, rotations
    
# if __name__ == "__main__":
#     import scipy
#     gtu = GaussianTransformUtils()
#     feats = torch.rand(100, 16, 3).float()
#     r = torch.asarray(scipy.spatial.transform.Rotation.random(100).as_matrix()).float()
#     r0 = r[0]
#     feats1 = gtu.vectorized_transform_shs(feats, r)
#     print(feats[0][1], id(feats))
#     print(feats1[0][1], id(feats1))
    

'''
def transform_shs(features, rotation_matrix):
    try:
        from e3nn import o3
        import einops
        from einops import einsum
    except:
        print("Please run `pip install e3nn einops` to enable SHs rotation")
        return

    if features.shape[1] == 1:
        return features

    features = features.clone()

    shs_feat = features[:, 1:, :]

    ## rotate shs
    P = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=shs_feat.dtype, device=shs_feat.device)  # switch axes: yzx -> xyz
    inversed_P = torch.tensor([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
    ], dtype=shs_feat.dtype, device=shs_feat.device)
    permuted_rotation_matrix = inversed_P @ rotation_matrix @ P
    rot_angles = o3._rotation.matrix_to_angles(permuted_rotation_matrix.cpu())

    # Construction coefficient
    D_1 = o3.wigner_D(1, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device=shs_feat.device)
    D_2 = o3.wigner_D(2, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device=shs_feat.device)
    D_3 = o3.wigner_D(3, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device=shs_feat.device)

    # rotation of the shs features
    one_degree_shs = shs_feat[:, 0:3]
    one_degree_shs = einops.rearrange(one_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    one_degree_shs = einsum(
        D_1,
        one_degree_shs,
        "... i j, ... j -> ... i",
    )
    one_degree_shs = einops.rearrange(one_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 0:3] = one_degree_shs

    if shs_feat.shape[1] >= 4:
        two_degree_shs = shs_feat[:, 3:8]
        two_degree_shs = einops.rearrange(two_degree_shs, 'n shs_num rgb -> n rgb shs_num')
        two_degree_shs = einsum(
            D_2,
            two_degree_shs,
            "... i j, ... j -> ... i",
        )
        two_degree_shs = einops.rearrange(two_degree_shs, 'n rgb shs_num -> n shs_num rgb')
        shs_feat[:, 3:8] = two_degree_shs

        if shs_feat.shape[1] >= 9:
            three_degree_shs = shs_feat[:, 8:15]
            three_degree_shs = einops.rearrange(three_degree_shs, 'n shs_num rgb -> n rgb shs_num')
            three_degree_shs = einsum(
                D_3,
                three_degree_shs,
                "... i j, ... j -> ... i",
            )
            three_degree_shs = einops.rearrange(three_degree_shs, 'n rgb shs_num -> n shs_num rgb')
            shs_feat[:, 8:15] = three_degree_shs

    return features
    
    
def vectorized_transform_shs(features, rotation_matrix):
    """
    https://github.com/graphdeco-inria/gaussian-splatting/issues/176#issuecomment-2147223570
    """

    try:
        from e3nn import o3
        import einops
        from einops import einsum
    except:
        print("Please run `pip install e3nn einops` to enable SHs rotation")
        return

    if features.shape[1] == 1:
        return features

    features = features.clone()

    shs_feat = features[:, 1:, :]

    ## rotate shs
    P = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=shs_feat.dtype, device=shs_feat.device)  # switch axes: yzx -> xyz
    inversed_P = torch.tensor([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
    ], dtype=shs_feat.dtype, device=shs_feat.device)
    permuted_rotation_matrix = inversed_P @ rotation_matrix @ P
    rot_angles = o3._rotation.matrix_to_angles(permuted_rotation_matrix.cpu())

    # Construction coefficient
    D_1 = o3.wigner_D(1, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device=shs_feat.device)
    D_2 = o3.wigner_D(2, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device=shs_feat.device)
    D_3 = o3.wigner_D(3, rot_angles[0], - rot_angles[1], rot_angles[2]).to(device=shs_feat.device)

    # rotation of the shs features
    one_degree_shs = shs_feat[:, 0:3]
    one_degree_shs = einops.rearrange(one_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    one_degree_shs = einsum(
        D_1,
        one_degree_shs,
        "... i j, ... j -> ... i",
    )
    one_degree_shs = einops.rearrange(one_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 0:3] = one_degree_shs

    if shs_feat.shape[1] >= 4:
        two_degree_shs = shs_feat[:, 3:8]
        two_degree_shs = einops.rearrange(two_degree_shs, 'n shs_num rgb -> n rgb shs_num')
        two_degree_shs = einsum(
            D_2,
            two_degree_shs,
            "... i j, ... j -> ... i",
        )
        two_degree_shs = einops.rearrange(two_degree_shs, 'n rgb shs_num -> n shs_num rgb')
        shs_feat[:, 3:8] = two_degree_shs

        if shs_feat.shape[1] >= 9:
            three_degree_shs = shs_feat[:, 8:15]
            three_degree_shs = einops.rearrange(three_degree_shs, 'n shs_num rgb -> n rgb shs_num')
            three_degree_shs = einsum(
                D_3,
                three_degree_shs,
                "... i j, ... j -> ... i",
            )
            three_degree_shs = einops.rearrange(three_degree_shs, 'n rgb shs_num -> n shs_num rgb')
            shs_feat[:, 8:15] = three_degree_shs

    return features
'''

@torch.no_grad()
def remove_samples_from_gaussians(gaussians, mask):
    gaussians._xyz = gaussians._xyz[mask]
    gaussians.canonical_xyz = gaussians.canonical_xyz[mask]
    gaussians._features_dc = gaussians._features_dc[mask]
    gaussians._features_rest = gaussians._features_rest[mask]
    gaussians._scaling = gaussians._scaling[mask]
    gaussians._rotation = gaussians._rotation[mask]
    gaussians._opacity = gaussians._opacity[mask]
    
    gaussians.gauss2smpl = gaussians.gauss2smpl[mask]