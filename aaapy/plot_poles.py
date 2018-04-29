import numpy as np

def plot_poles(r, plt, size=True, color=False, trunc=5, small=0.2, scale=10):
    """Plot poles of a rational function with marker size reflecting residue
    magnitude. 
    
    :param r: BaryFun object whose poles will be plotted.

    :param plt: matplotlib pyplot to plot on.

    :param size: if `True` vary marker sizes with residues. Default `size =
    True`.

    :param color: if `True` vary marker colors with residues. Default `color =
    False`.

    :param trunc: residues with absolute value less than `10**(-trunc)`
        will be assigned marker size `small`. Default `trunc = 5`.

    :param small: float from 0 to 1 that specifies what fraction of `scale` is
        the smallest possible marker size. Default `small = 0.2`.

    :param scale: specifies scale of marker sizes. Default `scale = 10`.
    """

    pol = r.pol()
    res = abs(r.res())
    sz = scale * np.ones(len(pol))
    col = np.chararray(len(pol), itemsize=10)
    col[:] = 'red'

    if size:
        alf = (trunc + np.log10(res)) / trunc
        alf = small + (1-small) * np.minimum(1, np.maximum(alf, 0))
        sz = scale * alf

    if color:
        col[:] = 'pink'
        col[res > 1e-12] = 'lightgreen'
        col[res > 1e-9] = 'green'
        col[res > 1e-6] = 'cyan'
        col[res > 1e-3] = 'blue'

    for k in range(len(pol)):
        plt.plot(np.real(pol[k]), np.imag(pol[k]), '.', markersize=sz[k],
                c=col[k].decode('utf-8'))
