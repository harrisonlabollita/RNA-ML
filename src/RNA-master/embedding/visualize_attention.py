import os
import numpy as np
import pylab
import matplotlib.pyplot as plt


def PDF_SAVECHOP(f, PNG=False):
    from os import system
    if PNG == False:
        pylab.savefig(f + '.pdf', pad_inches=0, transparent=False)
        system('pdfcrop %s.pdf' % f)
        system('mv %s-crop.pdf %s.pdf' % (f, f))
    if PNG:
        pylab.savefig(f + '.jpg', pad_inches=0)
        system('convert -trim %s.jpg %s.jpg' % (f, f))

def plot_attention_weights(attention_weights, input_seq, decoded_seq):
    plt.rc(
        'font', **{
            'family': 'serif',
            'serif': ['Computer Modern Roman'],
            'monospace': ['Computer Modern Typewriter']
        })

    params = {
        'backend': 'ps',
        'axes.labelsize': 20,
        'font.size': 20,
        #'legend.fontsize': 10,
        #'legend.handlelength': 1,
        #'legend.columnspacing': 1,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        #'axes.linewidth': 0.1,
        #'axes.labelpad': 1,
        'text.usetex': True,
        'text.latex.preamble':
        [r"\usepackage{amstext}", r"\usepackage{mathpazo}"],
        #'xtick.major.pad': 10,
        #'ytick.major.pad': 10
    }
    pylab.rcParams.update(params)

    assert_msg ='Your attention weights was empty. Please check if the decoder produced  a proper translation'
    assert len(attention_weights) != 0, assert_msg

    mats = []
    dec_inputs = []
    for dec_ind, attn in attention_weights:
        mats.append(attn.reshape(-1)[:len(input_seq)])
        dec_inputs.append(dec_ind)
    attention_mat = np.transpose(np.array(mats))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(attention_mat, origin = 'lower')

    ax.set_xticks(np.arange(attention_mat.shape[1]))
    ax.set_yticks(np.arange(attention_mat.shape[0]))

    ax.set_xticklabels([x for x in decoded_seq])
    ax.set_yticklabels([x for x in input_seq])

    #ax.tick_params(labelsize=48)
    #ax.tick_params(axis='x', labelrotation=0)

    ax.set_xlabel(r"$\text{Decoded sequence (3' - 5')}$")
    ax.set_ylabel(r"$\text{Encoded sequence (5' - 3')}$")

    fig.tight_layout()

    if not os.path.exists(os.path.join('.', 'results')):
        os.mkdir(os.path.join('.', 'results'))
    
    PDF_SAVECHOP('results/{}'.format(input_seq))
    #plt.savefig(os.path.join('.', 'results', '{}.png'.format(input_seq)))
    plt.close()