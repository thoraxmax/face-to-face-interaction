__author__ = 'Max Thorsson'
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns
import matplotlib as mpl
import cv2
#to estimate the angle between three points, used in the AOI outline
def angle_btw(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return angle
#plotting function for AOIs and facial landmarks
def plot_aoi(dst_pts,r,ax):
    mpl.rcParams['font.family'] = 'Times New Roman'
    plt.set_cmap("Set2")
    c2 = np.sqrt(
        abs(dst_pts[0][1] - dst_pts[2][1])**2 +
        abs(dst_pts[0][0] - dst_pts[2][0])**2)
    c = abs(dst_pts[0][0] - dst_pts[1][0])
    angle = angle_btw(dst_pts[0], dst_pts[2], dst_pts[1]) / 2
    angle2 = angle_btw(dst_pts[0], dst_pts[1], dst_pts[2])
    alpha = np.arccos((r**2 + c**2 - r**2) / (2 * r * c))
    alpha2 = np.arccos((r**2 + c2**2 - r**2) / (2 * r * c2)) + angle
    alpha3 = np.arccos((r**2 + c2**2 - r**2) / (2 * r * c2)) + angle2
    angs = [np.linspace(alpha, 2 * np.pi - alpha3, 300), np.linspace(-np.pi + alpha3, np.pi - alpha, 300),np.linspace(np.pi / 2 + alpha2, np.pi * 2 + np.pi / 2 - alpha2, 300)]
    for p,ang in zip(dst_pts,angs):
        plt.plot(np.cos(ang)*r+p[0],np.sin(ang)*r+p[1],lw=1,ls=':',c='k',alpha=0.6)
    plt.plot(np.linspace(0,0,300)+150,np.linspace(150,np.sin(alpha)*r+dst_pts[0][1],300),lw=1,ls='-',c='k')
    plt.plot(np.linspace(150,np.cos(2*np.pi-alpha3)*r+dst_pts[0][0],300),np.linspace(150,np.sin(2*np.pi-alpha3)*r+dst_pts[0][1],300),lw=1,ls='-',c='k')
    plt.plot(np.linspace(150,np.cos(-np.pi+alpha3)*r+dst_pts[1][0],300),np.linspace(150,np.sin(2*np.pi-alpha3)*r+dst_pts[0][1],300),lw=1,ls='-',c='k')
    l1 = list(range(0, 305, 50))
    l1.reverse()
    l1 = [-l for l in l1]
    l2 = list(range(50, 305, 50))
    l1.extend(l2)
    ax.set_xticks(np.arange(0,301,25))
    ax.set_xticklabels(l1)
    ax.set_yticks(np.arange(0,301,25))
    ax.set_yticklabels(l1)
    ax.set_xlim(0,300)
    ax.set_ylim(50,250)
    ax.set_ylabel('Y face center offset  (cm)',fontsize=7)
    ax.set_xlabel('X face center offset (cm)',fontsize=7)
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.tick_params(axis='both',which='major',length=6,width=1,labelsize=7)
    plt.subplots_adjust(bottom=0.2)
    ax.legend(loc=0,fontsize=7)
    plt.tight_layout()
    return ax
