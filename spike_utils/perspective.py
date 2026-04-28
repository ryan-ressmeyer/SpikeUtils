"""
Perspective-correct stimulus sampling for flat-screen visual experiments.

The problem
-----------
In most visual-neuroscience pipelines, each bin's stimulus is sampled at
``gaze_pix(t) + rf_offset_pix`` — i.e. the receptive field is assumed to
translate 1:1 with gaze in screen-pixel coordinates. This "translation
approximation" silently treats the screen as a sphere centered on the eye.

On a flat monitor it is only accurate for small eccentricities and small
gaze excursions from primary gaze. A direction fixed in eye coordinates
(the cell's RF) lands on *different* pixel locations as gaze rotates,
because the mapping from eye-angle to screen-pixel is a perspective
projection, not a translation. The error grows quickly with both RF
eccentricity and gaze distance from the screen center, and it distorts
both the RF *center* and the local RF *shape*.

This module provides the geometric primitives needed to replace the
translation approximation with an explicit eye → world-ray → screen-pixel
projection.

Geometry and conventions
------------------------
- **Coordinate system** — right-handed Cartesian ``(x, y, z)`` with:

    * ``+x`` = forward (from eye toward screen, primary gaze direction)
    * ``+y`` = left on the screen (from the observer's perspective)
    * ``+z`` = up on the screen

  The eye (observer) sits at the origin. The screen is a flat plane
  perpendicular to ``+x`` at some fixed distance. "Primary gaze" points
  at the screen center. The ``+y`` axis points to the observer's left
  so that ``+x × +y = +z`` (right-handed). Because screen columns
  increase to the right, this means a pixel at ``col_offset > 0`` has
  world ``y < 0``, and positive azimuth ≡ leftward gaze.

- **Screen pixels** use image convention ``(row, col)`` with rows
  increasing downward, so ``+z`` (up) corresponds to *decreasing* row.
  Helpers that convert between pixel and Cartesian coordinates
  (``pix2pos`` / ``pos2pix``) take ``(row, col)`` indexing and produce
  ``(x, y)`` positions where ``x`` is horizontal-on-screen and ``y``
  is vertical-on-screen (note: these are *2D screen* coordinates, not
  the 3D world frame above — they describe positions within the
  screen plane).

- **Spherical coordinates** follow the physics convention
  ``(r, θ, φ)``: ``θ`` is the polar angle from ``+z``, ``φ`` is the
  azimuthal angle in the ``xy`` plane measured from ``+x``.
  ``cartesian_to_horizontal`` re-expresses a direction as
  ``(azimuth, elevation)``, where azimuth ≡ φ and elevation ≡ π/2 − θ.

- **Rotations** (``rotate_x``, ``rotate_y``, ``rotate_z``,
  ``zyx_rotation_matrix``, ``listing_rotation_matrix``) are right-handed
  and apply on the *left* of a column vector: ``v_rotated = R @ v``.
  Inputs are in radians and may be scalars or batched tensors; returned
  matrices carry a matching leading batch shape and trailing ``(3, 3)``.

- **Listing's law** (``listing_rotation_matrix``) produces the eye
  orientation that brings the primary gaze direction ``+x`` to the
  requested ``(azimuth, elevation)`` while enforcing zero torsion
  relative to primary position. It is the physiologically correct
  rotation to use for fixations / slow eye movements; it encodes the
  fact that the eye does not freely roll about the line of sight.

Typical use
-----------
The canonical perspective-correction workflow is:

1. Start from an RF offset measured under the translation approximation,
   ``rf_offset_pix`` (row, col), at primary gaze (screen center).
2. Convert the RF's screen pixel to a direction in the eye-rest frame
   (note the sign on ``col_offset``: screen columns increase to the
   right, but ``+y`` is left)::

       cm_per_pix = D_cm * (pi / 180) / pix_per_deg
       D_pix      = D_cm / cm_per_pix
       ray_eye    = (D_pix, -col_offset, -row_offset)         # (x, y, z)
       v_eye      = ray_eye / ||ray_eye||

3. For each time bin, build the Listing's-law rotation from the current
   gaze (converted from pixels to azimuth/elevation the same way), and
   rotate the eye-frame RF direction into world coordinates::

       R       = listing_rotation_matrix(azimuth, elevation)
       v_world = R @ v_eye

4. Project the rotated ray onto the screen plane (``project_to_x_plane``
   or the inverse of step 2) to obtain the true per-bin RF screen
   location under perspective, then sample the stimulus around that
   location instead of around ``gaze_pix + rf_offset_pix``.

Convenience helpers:

- ``pix2pos`` / ``pos2pix`` convert between screen pixel indices and
  screen-plane cm positions (square-pixel aware).
- ``project_to_x_plane`` intersects a batch of rays with a plane
  perpendicular to ``+x`` at a given distance; useful for stepping
  directly from world-frame rays back onto the screen.
- ``cartesian_to_spherical`` / ``spherical_to_cartesian`` /
  ``cartesian_to_horizontal`` convert between the coordinate
  representations used above.

All functions are PyTorch-based and preserve batch dimensions, so they
compose into vectorized per-bin samplers that are autograd-friendly
(useful if the screen distance or calibration is being fit alongside
a model).
"""

import torch


def pix2pos(pix: torch.Tensor, resolution, width, height=None, center_pos=None) -> torch.Tensor:
    '''
        Convert from pixel coordinates (i,j) to 2D position (x,y).
        The origin is set as the center of the screen can be changed with center_pos.
        The 2D positions corespond to the center of the pixels.


        Parameters
        ----------
        pix : array-like
            Pixel coordinates (..., 2) -> (row, col)
        resolution : array-like
            Screen resolution (width, height)
        width : float
            Screen width in cm
        height : float, optional
            Screen height in cm, by default None
            If None, assumes square pixels
        center_pos : array-like, optional
            Center position in cm (x_pos, y_pos), by default (0, 0)
    '''
    if center_pos is None:
        center_pos = torch.zeros(2)

    if height is None:
        height = width * resolution[1] / resolution[0]

    pix = torch.as_tensor(pix)

    center_pix = (resolution - 1) / 2
    cm_per_pix_x = width / (resolution[0] + 1)
    cm_per_pix_y = height / (resolution[1] + 1)
    pos = torch.zeros_like(pix, dtype=torch.float32, device=pix.device)
    pos[..., 0] = (pix[...,1] - center_pix[0]) * cm_per_pix_x + center_pos[0]
    pos[..., 1] = (center_pix[1] - pix[...,0]) * cm_per_pix_y + center_pos[1]
    return pos

def pos2pix(pos: torch.Tensor, resolution, width, height=None, center_pos=None) -> torch.Tensor:
    '''
    Convert from 2D position (x,y) to pixel coordinates (i,j).
    The origin is set as the center of the screen can be changed with center_pos.
    The 2D positions correspond to the center of the pixels.
    
    Parameters
    ----------
    pos : array-like
        2D positions (..., 2) -> (x, y) in cm
    resolution : array-like
        Screen resolution (width, height)
    width : float
        Screen width in cm
    height : float, optional
        Screen height in cm, by default None
        If None, assumes square pixels
    center_pos : array-like, optional
        Center position in cm (x_pos, y_pos), by default (0, 0)
        
    Returns
    -------
    torch.Tensor
        Pixel coordinates (..., 2) -> (row, col)
    '''
    if center_pos is None:
        center_pos = torch.zeros(2)
    if height is None:
        height = width * resolution[1] / resolution[0]
        
    pos = torch.as_tensor(pos)
    center_pix = (resolution - 1) / 2
    cm_per_pix_x = width / (resolution[0] + 1)
    cm_per_pix_y = height / (resolution[1] + 1)
    
    pix = torch.zeros_like(pos, dtype=torch.float32, device=pos.device)
    
    pix[..., 0] = center_pix[1] - (pos[..., 1] - center_pos[1]) / cm_per_pix_y
    pix[..., 1] = (pos[..., 0] - center_pos[0]) / cm_per_pix_x + center_pix[0]
     
    return pix

def spherical_to_cartesian(spherical: torch.Tensor) -> torch.Tensor:
    """
    Convert spherical coordinates (r, theta, phi) to Cartesian coordinates (x, y, z).
    If input's last dimension is 2, assumes unit vectors (r=1).
    
    Parameters:
    spherical : torch.Tensor
        Input coordinates where the last dimension must be 2 or 3
        If last dim is 3: Contains (r, theta, phi)
        If last dim is 2: Contains (theta, phi), assumes r=1
        E.g., shapes (2,), (3,), (N,2), (N,3), (M,N,2), (M,N,3), etc. are valid
        
    Returns:
    torch.Tensor
        Cartesian coordinates with same shape as input except last dimension
        contains (x, y, z)
    """
    spherical = torch.as_tensor(spherical)
    # Store original shape and device
    original_shape = spherical.shape
    device = spherical.device
    
    # Check if we're dealing with unit vectors (last dim = 2) or full spherical coords (last dim = 3)
    if original_shape[-1] not in (2, 3):
        raise ValueError("Last dimension must be 2 (theta,phi) or 3 (r,theta,phi)")
    
    # Reshape to 2D tensor
    if len(original_shape) == 1:
        spherical_2d = spherical.reshape(1, -1)
    else:
        spherical_2d = spherical.reshape(-1, original_shape[-1])
    
    # Extract coordinates
    if spherical_2d.shape[1] == 3:
        r, theta, phi = spherical_2d[:, 0], spherical_2d[:, 1], spherical_2d[:, 2]
    else:  # shape[1] == 2
        theta, phi = spherical_2d[:, 0], spherical_2d[:, 1]
        r = torch.ones_like(theta, device=device)
    
    # Convert to Cartesian coordinates
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    
    # Stack results
    cartesian = torch.stack((x, y, z), dim=-1)
    
    # Reshape back to original dimensions
    if len(original_shape) == 1:
        # Handle 1D input case
        cartesian = cartesian.squeeze(0)
    else:
        # Reshape to match original dimensions with last dim = 3
        new_shape = original_shape[:-1] + (3,)
        cartesian = cartesian.reshape(new_shape)
    
    return cartesian

def cartesian_to_spherical(xyz: torch.Tensor) -> torch.Tensor:
    """
    Convert Cartesian coordinates to spherical coordinates.
    For zero vectors, returns (0, NaN, NaN).
    
    Parameters:
    xyz : torch.Tensor
        Input coordinates where the last dimension must be 3
        E.g., shapes (3,), (N, 3), (M, N, 3), etc. are valid
        
    Returns:
    torch.Tensor
        Spherical coordinates with same shape as input except last dimension
        contains (r, theta, phi). For zero vectors, returns (0, NaN, NaN)
    """
    # Store original shape and device
    original_shape = xyz.shape
    device = xyz.device
    
    if original_shape[-1] != 3:
        raise ValueError("Last dimension must be 3 (x,y,z)")
    
    # Reshape to 2D tensor
    xyz_2d = xyz.reshape(-1, 3)
    x, y, z = xyz_2d[:, 0], xyz_2d[:, 1], xyz_2d[:, 2]
    
    # Calculate r
    r = torch.sqrt(x**2 + y**2 + z**2)
    
    # Create tensors for theta and phi initialized with NaN
    theta = torch.full_like(r, float('nan'), device=device)
    phi = torch.full_like(r, float('nan'), device=device)
    
    # Create mask for nonzero elements
    nonzero = r != 0
    
    # Calculate angles only where r != 0
    # Use where to maintain gradient flow
    # For theta: arccos(z/r) where r != 0
    safe_r = torch.where(nonzero, r, torch.ones_like(r))  # Avoid division by zero
    cos_theta = z / safe_r
    theta = torch.where(nonzero, 
                       torch.acos(torch.clamp(cos_theta, -1.0, 1.0)),  # Clamp for numerical stability
                       theta)
    
    # For phi: arctan2(y, x) where r != 0
    phi = torch.where(nonzero,
                     torch.atan2(y, x),
                     phi)
    
    # Stack results
    spherical = torch.stack((r, theta, phi), dim=-1)
    
    # Reshape back to original dimensions
    if len(original_shape) == 1:
        # Handle 1D input case
        spherical = spherical.squeeze(0)
    else:
        # Reshape to match original dimensions with last dim = 3
        new_shape = original_shape[:-1] + (3,)
        spherical = spherical.reshape(new_shape)
    
    return spherical

def cartesian_to_horizontal(xyz: torch.Tensor) -> torch.Tensor:
    '''
    Convert Cartesian coordinates to horizontal coordinates (azimuth, elevation).

    Parameters:
    -----------
    xyz : torch.Tensor
        Input coordinates where the last dimension must be 3
        E.g., shapes (3,), (N, 3), (M, N, 3), etc. are valid

    Returns:
    --------
    torch.Tensor
        Horizontal coordinates with same shape as input except last dimension
        contains (azimuth, elevation)
    '''

    # Store original shape and device
    spherical = cartesian_to_spherical(xyz)
    azimuth = spherical[..., 2]
    elevation = torch.pi / 2 - spherical[..., 1]
    return torch.stack((azimuth, elevation), dim=-1)

def project_to_x_plane(rays, plane_x_distance, p_eye=None, n_screen=None):
    '''
    Project a set of rays to a plane perpendicular to the x-axis
    
    Parameters
    ----------
    rays : torch.Tensor
        Rays to project with shape (..., 3)
    plane_x_distance : float
        Distance of the plane from the origin in the x-direction
    p_eye : torch.Tensor, optional
        Eye position with shape (3,), by default None
    n_screen : torch.Tensor, optional
        Normal vector of the screen, by default None

    Returns
    -------
    torch.Tensor
        Intersection points with the plane with shape (..., 3)
    '''
    if p_eye is None:
        p_eye = torch.zeros(3, device=rays.device, dtype=rays.dtype)
    if n_screen is None:
        n_screen = torch.tensor([1., 0., 0.], device=rays.device, dtype=rays.dtype)
        
    p_screen = torch.tensor([plane_x_distance, 0., 0.], 
                           device=rays.device, dtype=rays.dtype)
    
    # Compute intersection parameter t
    # Using einsum for dot product to maintain gradients
    numerator = torch.einsum('...i,i->...', (p_screen - p_eye), n_screen)
    denominator = torch.einsum('...i,i->...', rays, n_screen)
    t = numerator / denominator
    
    # Compute intersection points
    # Unsqueeze t to match rays dimensions for broadcasting
    p_samp = p_eye + t.unsqueeze(-1) * rays
    
    return p_samp

def rotate_x(theta):
    '''
    Rotate theta radians around x-axis
    Applies on the right, i.e. R @ v.
    Args:
        theta: float or torch.Tensor - rotation angle in radians of any shape
    Returns:
        torch.Tensor - rotation matrix with shape [..., 3, 3] where ... is theta's shape
    '''
    # Handle input type and device
    dtype = theta.dtype if torch.is_tensor(theta) else torch.float32
    device = theta.device if torch.is_tensor(theta) else 'cpu'
    
    # Convert to tensor if not already
    if not torch.is_tensor(theta):
        theta = torch.tensor(theta, dtype=dtype, device=device)
    
    # Create rotation matrix elements with proper broadcasting
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    zeros = torch.zeros_like(theta)
    ones = torch.ones_like(theta)
    
    # Stack the elements into a 3x3 matrix
    # Add two dimensions at the end for the 3x3 matrix
    R = torch.stack([
        torch.stack([ones, zeros, zeros], dim=-1),
        torch.stack([zeros, cos_t, -sin_t], dim=-1),
        torch.stack([zeros, sin_t, cos_t], dim=-1)
    ], dim=-2)
    
    return R

def rotate_y(theta):
    '''
    Rotate theta radians around y-axis
    Applies on the right, i.e. R @ v.
    Args:
        theta: float or torch.Tensor - rotation angle in radians of any shape
    Returns:
        torch.Tensor - rotation matrix with shape [..., 3, 3] where ... is theta's shape
    '''
    dtype = theta.dtype if torch.is_tensor(theta) else torch.float32
    device = theta.device if torch.is_tensor(theta) else 'cpu'
    
    if not torch.is_tensor(theta):
        theta = torch.tensor(theta, dtype=dtype, device=device)
    
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    zeros = torch.zeros_like(theta)
    ones = torch.ones_like(theta)
    
    R = torch.stack([
        torch.stack([cos_t, zeros, sin_t], dim=-1),
        torch.stack([zeros, ones, zeros], dim=-1),
        torch.stack([-sin_t, zeros, cos_t], dim=-1)
    ], dim=-2)
    
    return R

def rotate_z(theta):
    '''
    Rotate theta radians around z-axis
    Applies on the right, i.e. R @ v.
    Args:
        theta: float or torch.Tensor - rotation angle in radians of any shape
    Returns:
        torch.Tensor - rotation matrix with shape [..., 3, 3] where ... is theta's shape
    '''
    dtype = theta.dtype if torch.is_tensor(theta) else torch.float32
    device = theta.device if torch.is_tensor(theta) else 'cpu'
    
    if not torch.is_tensor(theta):
        theta = torch.tensor(theta, dtype=dtype, device=device)
    
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    zeros = torch.zeros_like(theta)
    ones = torch.ones_like(theta)
    
    R = torch.stack([
        torch.stack([cos_t, -sin_t, zeros], dim=-1),
        torch.stack([sin_t, cos_t, zeros], dim=-1),
        torch.stack([zeros, zeros, ones], dim=-1)
    ], dim=-2)
    
    return R

def zyx_rotation_matrix(theat_z, theta_y, theta_x):
    '''
    Return rotation matrix for given angles around z, y, x axes.
    Applies on the right, i.e. R @ v.

    Parameters
    ----------
    theat_z : float
        Rotation angle around z-axis in radians
    theta_y : float
        Rotation angle around y-axis in radians
    theta_x : float
        Rotation angle around x-axis in radians

    Returns
    -------
    R : np.array
        Rotation matrix
    '''
    x = rotate_x(theta_x)
    y = rotate_y(theta_y)
    z = rotate_z(theat_z)
    zyx = torch.einsum('...ij,...jk,...kl->...il', (z, y, x))
    return zyx

def listing_rotation_matrix(azimuth, elevation):
    '''
    Return rotation matrix for given listing's law angles.
    Note: this only rotates from rest to the given angle, it does not rotate between angles.
    Applies on the right, i.e. R @ v.
    
    Parameters
    ----------
    azimuth : float or torch.Tensor
        Azimuth angle in radians
    elevation : float or torch.Tensor
        Elevation angle in radians
        
    Returns
    -------
    R : torch.Tensor
        Rotation matrix
    '''
    # Convert inputs to tensors if they aren't already
    if not torch.is_tensor(azimuth):
        azimuth = torch.tensor(azimuth)
    if not torch.is_tensor(elevation):
        elevation = torch.tensor(elevation)
        
    # Ensure both tensors are on same device and have same dtype
    elevation = elevation.to(device=azimuth.device, dtype=azimuth.dtype)
    
    # Calculate rotation angle and degree
    angle = torch.atan2(azimuth, -elevation)
    degree = torch.sqrt(elevation**2 + azimuth**2)
    
    # Compose rotations using matrix multiplication
    return rotate_x(angle) @ rotate_y(degree) @ rotate_x(-angle)


