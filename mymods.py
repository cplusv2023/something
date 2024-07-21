import matplotlib as mpl
import matplotlib.ticker as ticker
import matplotlib.colorbar as colorbar
import os, re, sys, datetime, io
from subprocess import Popen,PIPE
from subprocess import run as Run
import numpy as np
import random
import matplotlib.pyplot as plt

def flow(source, cmd, read=False):
    out = io.BytesIO()
    tmpname = "notmp"
    if source == None:
        source = "/dev/null"
        read = True
    if not read:
        source.seek(0)
        try:
            source.fileno()
        except:
            alphabets = 'abcdefghijklmnopqrstuvwxyz1234567890'
            tmpname = "%s/%s.rsf" % (os.environ.get("DATAPATH"), ''.join(random.sample(alphabets, 7)))
            with open(tmpname, "wb") as fsrc:
                fsrc.write(source.read())
            source = tmpname
            read = True
    if read:
        with open(source, "rb") as fsrc:
            run = Run(cmd, stdin=fsrc, stdout=PIPE, shell=True, check=True)
    else:
        run = Run(cmd, stdin=source, stdout=PIPE, shell=True, check=True)
    out.write(run.stdout)
    out.seek(0)
    if not tmpname == "notmp": os.system("rm -rf %s" % tmpname)
    return out


def readrsf(fp):
    forms = {"native_int": np.int32,
             "native_float": np.float32,
             "native_complex": np.complex64}

    fstream = fp.read()
    fheadend = fstream.find(b"\x0c\x0c\x04")
    fhead = str(fstream[:fheadend].decode())
    pattern = re.compile(r"\S+?=\S+")
    fhead = pattern.findall(fhead)
    headdict = {}
    for record in fhead:
        key0, val0 = record.split("=")
        if len(val0) > 1 and val0[0] == "\"" and val0[-1] == "\"": val0 = val0[1:-1]
        headdict[key0] = val0
    dim = 1
    dims = [int(headdict["n1"])]
    for idim in range(2, 10):
        if "n%d" % idim in headdict.keys():
            dim += 1
            dims.append(int(headdict["n%d" % idim]))
    headdict["dim"] = dim
    while dim > 1 and dims[-1] == 1:
        dim -= 1
        dims.pop()
    if "data_format" in headdict.keys():
        format0 = headdict["data_format"]
    else:
        format0 = "native_float"

    fdata = np.frombuffer(fstream[fheadend + 3:], dtype=forms[format0])
    dims.reverse()
    fdata = fdata.reshape(dims)
    fdata = fdata.transpose()
    return fdata, headdict


def strfun(str1):
    return str1.encode("utf-8")


def writersf(data, filename=None, dict=None):
    RSFHSPLITER = b"\x0c\x0c\x04"
    dims = []
    for idim in range(len(data.shape)):
        dims.append(data.shape[idim])

    if not filename == None:
        fout = open(filename, "wb")
    else:
        fout = io.BytesIO()
    fout.write(strfun("This file is created through Python %s\n" % (sys.version)))
    fout.write(strfun("%s\n" % (datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S'))))
    for idim in range(len(dims)): fout.write(strfun("n%d=%d " % (idim + 1, dims[idim])))
    fout.write(strfun("o1=0 d1=1 o2=0 d2=1\n"))
    fout.write(strfun("data_format=native_float in=stdin\n"))
    fout.write(strfun("%s\n" % (dict)))
    fout.write(RSFHSPLITER)
    fout.write(data.transpose().tobytes())
    if filename == None:
        fout.seek(0)
        return fout
    else:
        fout.close()
        return None


def getaxis(headdict):
    axes = []
    for idim in range(1, 10):
        if "n%d" % idim in headdict.keys():
            n = int(headdict["n%d" % idim])
            if "d%d" % idim in headdict.keys():
                d = float(headdict["d%d" % idim])
            else:
                d = 1
            if "o%d" % idim in headdict.keys():
                o = float(headdict["o%d" % idim])
            else:
                o = 0
            axes.append(np.linspace(o, o + n * d, n, False))
    return axes

def grey(fdata,fhead=None,vmin=None, vmax=None, figsize=[3, 4],facecolor="white",cmap="seismic",
          newfig=True,dpi=100):
    inputshape = fdata.shape
    maxelem = np.max(np.abs(fdata.ravel()))
    if vmin == None and vmax==None:
        vmin = -maxelem
        vmax = maxelem
    elif vmin==None: vmin = -vmax
    elif vmax==None: vmax = -vmin
    fhead0 = {
            "label1":"Time","label2":"Distance",
            "unit1":"s","unit2":"km",
            "n1":inputshape[0],
            "n2":inputshape[1],
            "d1":0.002,"o1":0.,
            "d2":1,"o2":0.
        }
    if fhead == None:
        fhead= fhead0
    else:
        for ikeys in fhead0.keys():
            if ikeys not in fhead:
                fhead[ikeys] = fhead0[ikeys]
    axes1 = getaxis(fhead)
    axist = axes1[0]
    axisx = axes1[1]
    if newfig:
        fig = plt.figure(facecolor=facecolor, figsize=figsize,dpi=dpi)
        ax = fig.subplots(1, 1)
    else:
        fig = plt.gcf()
        ax = plt.gca()
    ax.imshow(fdata, aspect="auto", cmap=cmap,
              vmin=vmin, vmax=vmax, extent=[axisx[0], axisx[-1], axist[-1], axist[0]])

    if fhead["unit1"]=="": ylabel = fhead["label1"]
    else: ylabel = "%s (%s)" % (fhead["label1"], fhead["unit1"])
    if fhead["unit2"]=="": xlabel = fhead["label2"]
    else: xlabel = "%s (%s)" % (fhead["label2"], fhead["unit2"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def grey3(frame1, frame2, frame3, sfaxes, title="", iframe1=0, iframe2=0, iframe3=0, point1=0.75, point2=0.5, vmin=None, vmax=None,
          color="jet", fontsize=15, wanttitle=False, wantbar=False, bartitle="", label1="", label2="", label3="", barpos=None,
          figsize=[5, 5],
          newfig=True):
    if vmin == None or vmax == None:
        maxes = []
        maxes.append(np.max(np.abs(frame1.ravel())))
        maxes.append(np.max(np.abs(frame2.ravel())))
        maxes.append(np.max(np.abs(frame3.ravel())))
        # mins = []
        # mins.append(np.min(frame1.ravel()))
        # mins.append(np.min(frame2.ravel()))
        # mins.append(np.min(frame3.ravel()))
        vmax1 = np.max(maxes)
        # vmin = np.min(mins)
        vmin1 = -vmax1
            # vmin = vmin * cmax / 100.
    if vmin == None: vmin = vmin1
    if vmax == None: vmax = vmax1

    linecol = 'blue'
    textcol = 'blue'
    if color == 'gray':
        linecol = 'white'
        textcol = 'black'
    if color == 'jet':
        linecol = 'black'
        textcol = 'black'

    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-1, 3))
    current_map = color

    if newfig:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.gca()
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['font.size'] = fontsize

    # width = fig.get_figwidth()
    # height = fig.get_figheight()

    btm_margin = 0.125  # for labels and axis
    top_margin = 0.15  # for titles
    left_margin = 0.15  # for labels and axis
    right_margin = 0.25  # for labels and axis
    if newfig: ax0 = fig.add_axes([0, 0, 1, 1], facecolor='w')  # The base axes
    else: ax0 = fig.inset_axes([0, 0, 1, 1], facecolor='w')  # The base axes
    ax0.set_frame_on(False)
    ax0.get_xaxis().set_visible(False)
    ax0.get_yaxis().set_visible(False)

    if wanttitle:
        title_pos = [0, 1 - top_margin, 1, top_margin]
        if newfig: ax_title = fig.add_axes(title_pos, facecolor='w')
        else: ax_title = fig.inset_axes(title_pos, facecolor='w')
        ax_title.set_frame_on(False)
        ax_title.get_xaxis().set_visible(False)
        ax_title.get_yaxis().set_visible(False)
        ax_title.set_xlim([0, 1])
        ax_title.set_ylim([0, 1])
        ax_title.text(0.5, 0.5, title, va='center', ha='center', fontsize=25)
        ax0.add_child_axes(ax_title)
    else:
        top_margin = 0.075

    if wantbar:
        # bar_pos = [1-right_margin,btm_margin,right_margin,1]
        if newfig:
            bar_pos1 = [1 - right_margin + 0.1, btm_margin, right_margin - 0.2, 0.9 * (1 - top_margin - btm_margin)]
            ax_bar = fig.add_axes(bar_pos1, facecolor='none')
        else:
            if barpos == None: bar_pos1 = [0.9,btm_margin,right_margin - 0.2, 0.9 * (1 - top_margin - btm_margin)]
            else :bar_pos1 = barpos
            ax_bar = plt.gcf().add_axes(bar_pos1, facecolor='none')
        ax_bar.set_title(bartitle)
        if not newfig: right_margin = 0.1
    else:
        right_margin = 0.1

    h_ax1 = 1 - top_margin - btm_margin
    w_ax1 = 1 - left_margin - right_margin

    if newfig: ax1 = fig.add_axes([left_margin, btm_margin, w_ax1 * point2, h_ax1 * point1])
    else: ax1 = fig.inset_axes([left_margin, btm_margin, w_ax1 * point2, h_ax1 * point1])
    ax0.add_child_axes(ax1)
    im1 = ax1.imshow(frame1, aspect='auto', cmap=current_map
                     , vmin=vmin, vmax=vmax
                     , extent=[sfaxes[1][0], sfaxes[1][-1], sfaxes[0][-1], sfaxes[0][0]])
    ax1.set_xlabel(label2)
    ax1.set_ylabel(label1)
    ax1.set_yticklabels(ax1.get_yticklabels(), va='top')

    if newfig: ax2 = fig.add_axes([left_margin + w_ax1 * point2, btm_margin, w_ax1 * (1 - point2), h_ax1 * point1])
    else: ax2 = fig.inset_axes([left_margin + w_ax1 * point2, btm_margin, w_ax1 * (1 - point2), h_ax1 * point1])
    ax0.add_child_axes(ax2)
    ax2.imshow(frame2.T, aspect='auto', cmap=current_map
               , vmin=vmin, vmax=vmax
               , extent=[sfaxes[2][0], sfaxes[2][-1], sfaxes[0][0], sfaxes[0][-1]])
    ax2.set_xlabel(label3)
    ax2.set_xticklabels(ax2.get_xticklabels(), va='top', ha='center')
    ax2.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_label_position('bottom')
    ax2.get_xaxis().set_ticks_position('bottom')

    if newfig: ax3 = fig.add_axes([left_margin, btm_margin + h_ax1 * point1, w_ax1 * point2, h_ax1 * (1 - point1)])
    else: ax3 = fig.inset_axes([left_margin, btm_margin + h_ax1 * point1, w_ax1 * point2, h_ax1 * (1 - point1)])
    ax0.add_child_axes(ax3)
    ax3.imshow(frame3, aspect='auto', cmap=current_map, origin="lower"
               , vmin=vmin, vmax=vmax
               , extent=[sfaxes[1][0], sfaxes[1][-1], sfaxes[2][0], sfaxes[2][-1]])
    ax3.get_xaxis().set_visible(False)
    # ax3.invert_yaxis()
    ax3.set_ylabel(label3)
    ax3.set_yticklabels(ax3.get_yticklabels(), va='bottom')

    if wantbar:
        cbar = colorbar.Colorbar(ax_bar, im1, cmap=current_map, format=fmt)
        cbar.ax.get_yaxis().set_label_position('right')

    if newfig: ax_01 = fig.add_axes([left_margin, btm_margin, w_ax1, h_ax1])
    else: ax_01 = fig.inset_axes([left_margin, btm_margin, w_ax1, h_ax1])
    ax_01.set_frame_on(False)
    ax_01.get_xaxis().set_visible(False)
    ax_01.get_yaxis().set_visible(False)
    ax_01.set_xlim([0, 1])
    ax_01.set_ylim([0, 1])

    # hline for iframe1
    lframe1 = (len(sfaxes[0]) - iframe1) / len(sfaxes[0]) * point1
    ax_01.axhline(lframe1, linestyle='--', linewidth=1, color=linecol)
    ax_01.text(1.01, lframe1, "%3.2f" % (sfaxes[0][iframe1]), va='center', ha='left', color=textcol)
    # vline for iframe2
    lframe2 = iframe2 / len(sfaxes[1]) * point2
    ax_01.axvline(lframe2, linestyle='--', linewidth=1, color=linecol)
    ax_01.text(lframe2, 1.01, "%3.2f" % (sfaxes[1][iframe2]), va='bottom', ha='center', color=textcol)
    # hline for iframe3
    lframe3 = point1 + iframe3 / len(sfaxes[2]) * (1 - point1)
    ax_01.plot([0, point2], [lframe3, lframe3], linestyle='--', linewidth=1, color=linecol)
    ax_01.text(point2 + 0.01, lframe3, "%3.2f" % (sfaxes[2][iframe3]), va='bottom', ha='left', color=textcol)
    # vline for iframe3
    lframe3 = point2 + iframe3 / len(sfaxes[2]) * (1 - point2)
    ax_01.plot([lframe3, lframe3], [0, point1], linestyle='--', linewidth=1, color=linecol)
    if iframe3 / len(sfaxes[2]) >= 0.3:
        ax_01.text(lframe3, point1 + 0.01, "%3.2f" % (sfaxes[2][iframe3]), va='bottom', ha='left', color=textcol)
