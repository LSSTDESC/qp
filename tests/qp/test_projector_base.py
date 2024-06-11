import qp
import qp.projectors as proj
import numpy as np


def make_qp_ens(file):
    zs = file['zs']
    pzs = file['pzs']
    dz = np.mean(np.diff(zs))
    zs_edges = np.append(zs - dz/2, zs[-1] + dz/2)
    q = qp.Ensemble(qp.hist, data={"bins":zs_edges, "pdfs":pzs})
    return q

def test_base_from_qp():
    file = np.load('tests/qp/dummy.npz')
    ens = make_qp_ens(file)
    projector = proj.ProjectorBase.ProjectorBase(ens)
    m, n = projector.pzs.shape
    k, = projector.z.shape
    pzs = file['pzs']
    pzs /= np.sum(pzs, axis=1)[:, None]
    assert n == k
    assert np.allclose(projector.pz_mean, np.mean(pzs, axis=0))

def test_base_from_arrs():
    file = np.load('tests/qp/dummy.npz')
    zs = file['zs']
    pzs = file['pzs']
    projector = proj.ProjectorBase.ProjectorBase(zs, pzs)
    m, n = projector.pzs.shape
    k, = projector.z.shape
    pzs = file['pzs']
    pzs /= np.sum(pzs, axis=1)[:, None]
    assert n == k
    assert np.allclose(projector.pz_mean, np.mean(pzs, axis=0))