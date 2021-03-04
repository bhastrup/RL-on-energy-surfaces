import numpy as np


def mirror(pos, B, n_surf):

    pos_rel = pos - pos[-1]

    # First we need orthogonal basis for the plane spanned by n_surf and B
    # (using relative position, so plane now goes through agent)

    # Gram Schmidt orthogonalization:
    vec1 = n_surf
    vec2 = B - np.dot(B, n_surf)/np.dot(n_surf, n_surf) * n_surf

    # Project onto plane 
    proj_rel = np.expand_dims((np.dot(pos_rel, vec1)/np.dot(vec1, vec1)), axis=1) * vec1 \
        + np.expand_dims((np.dot(pos_rel, vec2)/np.dot(vec2, vec2)), axis=1) * vec2

    # Mirror position in relative coordinates
    mirror_pos_rel = pos_rel + 2*(proj_rel - pos_rel)

    # Mirror pos in original coordinates
    mirror_pos = mirror_pos_rel + pos[-1]

    return mirror_pos


