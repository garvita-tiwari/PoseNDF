import quaternionic
import spherical
import ipdb

ell_max = 16  # Use the largest ℓ value you expect to need
wigner = spherical.Wigner(ell_max)
R = quaternionic.array([1, 2, 3, 4]).normalized
ipdb.set_trace()
wigner.rotate(modes, R)
wigner.evaluate(modes, R)
D = wigner.D(R)
D[wigner.Dindex(ell, mp, m)]
Y = wigner.sYlm(s, R)
Y[wigner.Yindex(ell, m)]
