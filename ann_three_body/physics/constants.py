G = 6.67408e-11  # N-m2/kg2


#
#   Normalize the constants such that m, r, t and v are of the order 10^1.
#


def get_normalization_constants_alphacentauri():
    # Normalize the masses to the mass of our sun
    m_nd = 1.989e+30  # kg

    # Normalize distances to the distance between Alpha Centauri A and Alpha Centauri B
    r_nd = 5.326e+12  # m

    # Normalize velocities to the velocity of earth around the sun
    v_nd = 30000  # m/s

    # Normalize time to the orbital period of Alpha Centauri A and B
    t_nd = 79.91 * 365 * 24 * 3600 * 0.51  # s

    return m_nd, r_nd, v_nd, t_nd


def get_normalization_constants_earthsun():
    # Normalize the masses to the mass of our sun
    m_nd = 1.989e+30  # kg

    # Normalize distances to the distance between Earth and Sun
    r_nd = 149.47e9  # m

    # Normalize velocities to the velocity of earth around the sun
    v_nd = 29.78e3  # m/s

    # Normalize time to the orbital period of Earth and Sun
    t_nd = 365 * 24 * 3600 * 0.51  # s

    return m_nd, r_nd, v_nd, t_nd


def normalization_constants_none():
    return 1, 1, 1, 1


m_nd, r_nd, v_nd, t_nd = get_normalization_constants_alphacentauri()
