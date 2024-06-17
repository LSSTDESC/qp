import qp
import qp.projectors as proj
import numpy as np


def make_qp_ens(file):
    zs = file['zs']
    nzs = file['pzs']
    dz = np.mean(np.diff(zs))
    zs_edges = np.append(zs - dz/2, zs[-1] + dz/2)
    q = qp.Ensemble(qp.hist, data={"bins":zs_edges, "pdfs":nzs})
    return q

def test_base():
    file = np.load('tests/qp/dummy.npz')
    ens = make_qp_ens(file)
    projector = proj.ProjectorBase(ens)
    m, n = projector.nzs.shape
    k, = projector.z.shape
    nzs = file['pzs']
    nzs /= np.sum(nzs, axis=1)[:, None]
    assert n == k
    assert np.allclose(projector.nz_mean, np.mean(nzs, axis=0))
