def do_case(i):
    print(cases[i])
    (dirname, n_pdfs, n_params) = cases[i]

    (z_low, z_high) = datasets[dirname]['z_ends']
    z = np.arange(z_low, z_high, 0.01, dtype='float')
    z_range = z_high - z_low
    delta_z = z_range / len(z)

    with open(datasets[dirname]['filename'], 'rb') as data_file:
        lines = (line.split(None) for line in data_file)
        lines.next()
        all_pdfs = np.array([[float(line[k]) for k in range(1,len(line))] for line in lines])

    n_pdfs_tot = len(all_pdfs)
    full_pdf_range = range(n_pdfs_tot)
    subset = np.random.choice(full_pdf_range, n_pdfs)
    pdfs = all_pdfs[subset]

    nz_stats = {}
    nz_stats['KLD'] = {}
    nz_stats['RMS'] = {}

    pr = cProfile.Profile()
    pr.enable()

    E0 = qp.Ensemble(n_pdfs, gridded=(z, pdfs), vb=False)
    samparr = E0.sample(n_samps, vb=False)
    Ei = qp.Ensemble(n_pdfs, samples=samparr, vb=False)
    fits = Ei.mix_mod_fit(comps=datasets[dirname]['n_comps'], using='gridded', vb=False)
    Ef = qp.Ensemble(n_pdfs, truth=fits, vb=False)

    P = qp.PDF(gridded=Ef.stack(z, using='mix_mod', vb=False)['mix_mod'], vb=False)

    # pr = cProfile.Profile()
    # pr.enable()

    Eq = qp.Ensemble(n_pdfs, quantiles=Ef.quantize(N=n_params))
    Q = qp.PDF(gridded=Eq.stack(z, using='quantiles', vb=False)['quantiles'], vb=False)
    nz_stats['KLD']['quantiles'] = qp.utils.calculate_kl_divergence(P, Q)
    nz_stats['RMS']['quantiles'] = qp.utils.calculate_rmse(P, Q)

    Eh = qp.Ensemble(n_pdfs, histogram=Ef.histogramize(N=n_params, binrange=datasets[dirname]['z_ends']))
    Q = qp.PDF(gridded=Eh.stack(z, using='histogram', vb=False)['histogram'], vb=False)
    nz_stats['KLD']['histogram'] = qp.utils.calculate_kl_divergence(P, Q)
    nz_stats['RMS']['histogram'] = qp.utils.calculate_rmse(P, Q)

    Es = qp.Ensemble(n_pdfs, samples=Ef.sample(samps=n_params))
    Q = qp.PDF(gridded=Es.stack(z, using='samples', vb=False)['samples'], vb=False)
    nz_stats['KLD']['samples'] = qp.utils.calculate_kl_divergence(P, Q)
    nz_stats['RMS']['samples'] = qp.utils.calculate_rmse(P, Q)

    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumtime'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    with open(logfilename, 'wb') as logfile:
        logfile.write('ran case '+str(cases[i])+' with '+str(s.getvalue())+'\n')
    print(n_pdfs, n_params, s.getvalue())

    all_stats[str(i)] = nz_stats

    outpath = os.path.join(dirname, str(n_pdfs)+dirname+str(n_params)+'.hkl')
    with open(outpath, 'w') as outfile:
        hkl.dump(nz_stats, outfile)

    return

if __name__ == "__main__":

    import os
    import itertools
    import numpy as np
    import cProfile
    import pstats
    import StringIO
    import hickle as hkl
    # import pathos
    import matplotlib.pyplot as plt
    import qp

    logfilename = 'progress.txt'

    datasets = {}
    datasets['mg'] = {}
    datasets['mg']['filename'] = 'bpz_euclid_test_10_2.probs'
    datasets['mg']['z_ends'] = (0.01, 3.51)
    datasets['mg']['n_comps'] = 3
    datasets['ss'] = {}
    datasets['ss']['filename'] = 'test_magscat_trainingfile_probs.out'
    datasets['ss']['z_ends'] = (0.005, 2.11)
    datasets['ss']['n_comps'] = 5
    for dirname in datasets.keys():
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    n_samps = 1000
    n_gals = [10000]
    n_params_to_test = [3, 10, 30, 100]
    params_to_test = ['samples', 'histogram', 'quantiles']
    cases = list(itertools.product(datasets.keys(), n_gals, n_params_to_test))
    case_range = range(len(cases))

    all_stats = {}

    # nps = psutil.cpu_count()
    # pool = Pool(nps)
    # final = pool.map(do_case, case_range)

    for i in case_range:
        do_case(i)
