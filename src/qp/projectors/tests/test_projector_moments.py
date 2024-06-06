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


def make_projector():
    file = np.load('rail_projector/tests/dummy.npz')
    ens = make_qp_ens(file)
    return rp.ProjectorMoments(ens)


def test_prior():
    projector = make_projector()
    prior = projector.get_prior()
    assert prior is not None


def test_sample_prior():
    projector = make_projector()
    pz = projector.sample_prior()
    assert len(pz) == len(projector.pz_mean)


def test_model():
    projector = make_projector()
    shift = projector.sample_prior()
    input = np.array([projector.z, projector.pz_mean])
    output = projector.evaluate_model(input, shift)
    assert (projector.z == output[0]).all()
    assert len(output[1]) == len(projector.pz_mean)
