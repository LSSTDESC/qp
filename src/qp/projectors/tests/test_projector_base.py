import qp
import numpy as np
import rail_projector.projectors as rp


def make_qp_ens(file):
    zs = file['zs']
    pzs = file['pzs']
    dz = np.mean(np.diff(zs))
    zs_edges = np.append(zs - dz/2, zs[-1] + dz/2)
    q = qp.Ensemble(qp.hist, data={"bins":zs_edges, "pdfs":pzs})
    return q

def test_base_from_qp():
    file = np.load('rail_projector/tests/dummy.npz')
    ens = make_qp_ens(file)
    projector = rp.ProjectorBase(ens)
    m, n = projector.pzs.shape
    k, = projector.z.shape
    pzs = file['pzs']
    pzs /= np.sum(pzs, axis=1)[:, None]
    assert n == k
    assert np.allclose(projector.pz_mean, np.mean(pzs, axis=0))

def test_base_from_arrs():
    file = np.load('rail_projector/tests/dummy.npz')
    zs = file['zs']
    pzs = file['pzs']
    projector = rp.ProjectorBase(zs, pzs)
    m, n = projector.pzs.shape
    k, = projector.z.shape
    pzs = file['pzs']
    pzs /= np.sum(pzs, axis=1)[:, None]
    assert n == k
    assert np.allclose(projector.pz_mean, np.mean(pzs, axis=0))