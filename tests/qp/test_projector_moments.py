import qp
import numpy as np
import qp.projectors as proj


def make_qp_ens(file):
    zs = file['zs']
    nzs = file['pzs']
    dz = np.mean(np.diff(zs))
    zs_edges = np.append(zs - dz/2, zs[-1] + dz/2)
    q = qp.Ensemble(qp.hist, data={"bins":zs_edges, "pdfs":nzs})
    return q


def make_projector():
    file = np.load('tests/qp/dummy.npz')
    ens = make_qp_ens(file)
    return proj.ProjectorMoments(ens)


def test_prior():
    projector = make_projector()
    prior = projector.get_prior()
    assert prior is not None


def test_sample_prior():
    projector = make_projector()
    nz = projector.sample_prior()
    assert len(nz) == len(projector.nz_mean)


def test_model():
    projector = make_projector()
    shift = projector.sample_prior()
    input = np.array([projector.z, projector.nz_mean])
    output = projector.evaluate_model(input, shift)
    assert (projector.z == output[0]).all()
    assert len(output[1]) == len(projector.nz_mean)
