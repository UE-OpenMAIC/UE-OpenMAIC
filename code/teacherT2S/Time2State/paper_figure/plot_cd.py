                                                          

                                                      

                                                

                                                        

                                                          

               

import numpy as np

import pandas as pd

import matplotlib

matplotlib.use('agg')

import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'sans-serif'

matplotlib.rcParams['font.sans-serif'] = 'Arial'

import operator

import math

from scipy.stats import wilcoxon

from scipy.stats import friedmanchisquare

import networkx

import os 

script_path = os.path.dirname(__file__)

result_path = os.path.join(script_path, '../results')

                                                        

result_file_path = os.path.join(result_path, 'CD_methods.csv')

out_file_name = 'CD.eps'

                                                                                                        

def graph_ranks(avranks, names, p_values, cd=None, cdmethod=None, lowv=None, highv=None,

                width=6, textspace=1, reverse=False, filename=None, labels=False, **kwargs):

    

       

    try:

        import matplotlib

        import matplotlib.pyplot as plt

        from matplotlib.backends.backend_agg import FigureCanvasAgg

    except ImportError:

        raise ImportError("Function graph_ranks requires matplotlib.")

    width = float(width)

    textspace = float(textspace)

    def nth(l, n):

        

           

        n = lloc(l, n)

        return [a[n] for a in l]

    def lloc(l, n):

        

           

        if n < 0:

            return len(l[0]) + n

        else:

            return n

    def mxrange(lr):

        

           

        if not len(lr):

            yield ()

        else:

                                             

            index = lr[0]

            if isinstance(index, int):

                index = [index]

            for a in range(*index):

                for b in mxrange(lr[1:]):

                    yield tuple([a] + list(b))

    def print_figure(fig, *args, **kwargs):

        canvas = FigureCanvasAgg(fig)

        canvas.print_figure(*args, **kwargs)

    sums = avranks

    nnames = names

    ssums = sums

    if lowv is None:

        lowv = min(1, int(math.floor(min(ssums))))

    if highv is None:

        highv = max(len(avranks), int(math.ceil(max(ssums))))

    cline = 0.4

    k = len(sums)

    lines = None

    linesblank = 0

    scalewidth = width - 2 * textspace

    def rankpos(rank):

        if not reverse:

            a = rank - lowv

        else:

            a = highv - rank

        return textspace + scalewidth / (highv - lowv) * a

    distanceh = 0.25

    cline += distanceh

                                                

    minnotsignificant = max(2 * 0.2, linesblank)

    height = cline + ((k + 1) / 2) * 0.2 + minnotsignificant

    fig = plt.figure(figsize=(width, height))

    fig.set_facecolor('white')

    ax = fig.add_axes([0, 0, 1, 1])                  

    ax.set_axis_off()

    hf = 1. / height                 

    wf = 1. / width

    def hfl(l):

        return [a * hf for a in l]

    def wfl(l):

        return [a * wf for a in l]

                                 

    ax.plot([0, 1], [0, 1], c="w")

    ax.set_xlim(0, 1)

    ax.set_ylim(1, 0)

    def line(l, color='k', **kwargs):

        

           

        ax.plot(wfl(nth(l, 0)), hfl(nth(l, 1)), color=color, **kwargs)

    def text(x, y, s, *args, **kwargs):

        ax.text(wf * x, hf * y, s, *args, **kwargs)

    line([(textspace, cline), (width - textspace, cline)], linewidth=2)

    bigtick = 0.3

    smalltick = 0.15

    linewidth = 2.0

    linewidth_sign = 4.0

    tick = None

    for a in list(np.arange(lowv, highv, 0.5)) + [highv]:

        tick = smalltick

        if a == int(a):

            tick = bigtick

        line([(rankpos(a), cline - tick / 2),

              (rankpos(a), cline)],

             linewidth=2)

    for a in range(lowv, highv + 1):

        text(rankpos(a), cline - tick / 2 - 0.05, str(a),

             ha="center", va="bottom", size=16)

    k = len(ssums)

    def filter_names(name):

        return name

    space_between_names = 0.24

    for i in range(math.ceil(k / 2)):

        chei = cline + minnotsignificant + i * space_between_names

        line([(rankpos(ssums[i]), cline),

              (rankpos(ssums[i]), chei),

              (textspace - 0.1, chei)],

             linewidth=linewidth)

        if labels:

            text(textspace + 0.3, chei - 0.075, format(ssums[i], '.4f'), ha="right", va="center", size=10)

        text(textspace - 0.2, chei, filter_names(nnames[i]), ha="right", va="center", size=16)

    for i in range(math.ceil(k / 2), k):

        chei = cline + minnotsignificant + (k - i - 1) * space_between_names

        line([(rankpos(ssums[i]), cline),

              (rankpos(ssums[i]), chei),

              (textspace + scalewidth + 0.1, chei)],

             linewidth=linewidth)

        if labels:

            text(textspace + scalewidth - 0.3, chei - 0.075, format(ssums[i], '.4f'), ha="left", va="center", size=10)

        text(textspace + scalewidth + 0.2, chei, filter_names(nnames[i]),

             ha="left", va="center", size=16)

                           

    def draw_lines(lines, side=0.05, height=0.1):

        start = cline + 0.2

        for l, r in lines:

            line([(rankpos(ssums[l]) - side, start),

                  (rankpos(ssums[r]) + side, start)],

                 linewidth=linewidth_sign)

            start += height

            print('drawing: ', l, r)

                       

    start = cline + 0.2

    side = -0.02

    height = 0.1

                               

                     

    cliques = form_cliques(p_values, nnames)

    i = 1

    achieved_half = False

    print(nnames)

    for clq in cliques:

        if len(clq) == 1:

            continue

        print(clq)

        min_idx = np.array(clq).min()

        max_idx = np.array(clq).max()

        if min_idx >= len(nnames) / 2 and achieved_half == False:

            start = cline + 0.25

            achieved_half = True

        line([(rankpos(ssums[min_idx]) - side, start),

              (rankpos(ssums[max_idx]) + side, start)],

             linewidth=linewidth_sign)

        start += height

def form_cliques(p_values, nnames):

    

       

                                      

    m = len(nnames)

    g_data = np.zeros((m, m), dtype=np.int64)

    for p in p_values:

        if p[3] == False:

            i = np.where(nnames == p[0])[0][0]

            j = np.where(nnames == p[1])[0][0]

            min_i = min(i, j)

            max_j = max(i, j)

            g_data[min_i, max_j] = 1

    g = networkx.Graph(g_data)

    return networkx.find_cliques(g)

def draw_cd_diagram(df_perf=None, alpha=0.05, title=None, labels=False):

    

       

    p_values, average_ranks, _ = wilcoxon_holm(df_perf=df_perf, alpha=alpha)

    print(average_ranks)

    for p in p_values:

        print(p)

    graph_ranks(average_ranks.values, average_ranks.keys(), p_values,

                cd=None, reverse=True, width=9, textspace=1.5, labels=labels)

    font = {'family': 'sans-serif',

        'color':  'black',

        'weight': 'normal',

        'size': 22,

        }

    if title:

        plt.title(title,fontdict=font, y=0.9, x=0.5)

    plt.savefig(os.path.join(result_path, out_file_name), bbox_inches='tight')

def wilcoxon_holm(alpha=0.05, df_perf=None):

    

       

    print(pd.unique(df_perf['classifier_name']))

                                                        

    df_counts = pd.DataFrame({'count': df_perf.groupby(

        ['classifier_name']).size()}).reset_index()

                                               

    max_nb_datasets = df_counts['count'].max()

                                                                         

    classifiers = list(df_counts.loc[df_counts['count'] == max_nb_datasets]

                       ['classifier_name'])

                                                                              

    friedman_p_value = friedmanchisquare(*(

        np.array(df_perf.loc[df_perf['classifier_name'] == c]['accuracy'])

        for c in classifiers))[1]

    if friedman_p_value >= alpha:

                                                                                 

        print('the null hypothesis over the entire classifiers cannot be rejected')

        exit()

                                   

    m = len(classifiers)

                                                                                       

    p_values = []

                                                     

    for i in range(m - 1):

                                        

        classifier_1 = classifiers[i]

                                               

        perf_1 = np.array(df_perf.loc[df_perf['classifier_name'] == classifier_1]['accuracy']

                          , dtype=np.float64)

        for j in range(i + 1, m):

                                                   

            classifier_2 = classifiers[j]

                                                   

            perf_2 = np.array(df_perf.loc[df_perf['classifier_name'] == classifier_2]

                              ['accuracy'], dtype=np.float64)

                                   

            p_value = wilcoxon(perf_1, perf_2, zero_method='pratt')[1]

                               

            p_values.append((classifier_1, classifier_2, p_value, False))

                                  

    k = len(p_values)

                                                  

    p_values.sort(key=operator.itemgetter(2))

                                 

    for i in range(k):

                                 

        new_alpha = float(alpha / (k - i))

                                                              

        if p_values[i][2] <= new_alpha:

            p_values[i] = (p_values[i][0], p_values[i][1], p_values[i][2], True)

        else:

                  

            break

                                                                                  

                                        

    sorted_df_perf = (
        df_perf.loc[df_perf['classifier_name'].isin(classifiers)]
        .sort_values(['classifier_name', 'dataset_name'])
    )

                       

    rank_data = np.array(sorted_df_perf['accuracy']).reshape(m, max_nb_datasets)

                                                   

    df_ranks = pd.DataFrame(data=rank_data, index=np.sort(classifiers), columns=

    np.unique(sorted_df_perf['dataset_name']))

                    

    dfff = df_ranks.rank(ascending=False)

    print(dfff[dfff == 1.0].sum(axis=1))

                       

    average_ranks = df_ranks.rank(ascending=False).mean(axis=1).sort_values(ascending=False)

                                               

    return p_values, average_ranks, max_nb_datasets

df_perf = pd.read_csv(result_file_path, index_col=False)                    

draw_cd_diagram(df_perf=df_perf, title='Rank', labels=True)
