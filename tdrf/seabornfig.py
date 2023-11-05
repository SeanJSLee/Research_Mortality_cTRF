print('Seaborn Fig - Loaded')

import pandas as pd
import numpy as np

# %matplotlib ipympl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns


class sb_fig : 
    def __init__(self, data, x, y_1=None, y_2=None, col=None, hue='year'):
        # initialize
        self.data           = data
        self.col            = col
        self.x              = x
        self.y_1            = y_1
        self.y_2            = y_2
        # self.add_stat       = add_stat
        self.hue            = hue
        # self.kind           = kind
        # self.palette_1      = palette_1
        # self.palette_2      = palette_2
        self.style          = 'dark'
        # self.col_wrap       = col_wrap
        # self.height         = height
        # self.aspect         = aspect
        # self.legend_2       = legend_2
        # self.set_ylabel_1   = set_ylabel_1
        # self.set_ylim_1     = set_ylim_1  
        pass
    # 
    def test(self) :
        print('ttttt')
        pass
    # 

    # add additional stats
    def _add_plot_stats(self, i_col_val, add_stat=dict, decimal=1) :
        add_stat_text = ''
        for key in add_stat.keys():
            i_stat = round(self.data[self.data[self.col]==i_col_val][key].mean(),
                           ndigits=decimal)
            add_stat_text += f'{add_stat[key]}: {i_stat}\n'
            # may need np.nanmean()
        return add_stat_text
        

    # rel plot #1 death to month
    def rel_plot_1 (self, 
                    add_stat = {'age_mean_y_c':'Age mean:', 'inc_adj05_y_c':'Income_mean'}, 
                    kind='scatter', palette_1='crest', palette_2='flare', 
                    col_wrap=3, height=3, aspect=1.2, legend_2=False, 
                    set_ylabel_1 = '', set_ylim_ax1=[0,200],
                    set_ylim_ax2 = [0,180],
                    text_x = -1, text_y = -80):
        sns.set_theme(style=self.style)
        g = sns.relplot(data=self.data, x=self.x, y=self.y_1, col=self.col, 
                        hue=self.hue, kind=kind, palette=palette_1,
                        col_wrap=col_wrap, height=height, aspect=aspect,  legend=False)
        # 
        for i_column, ax in g.axes_dict.items():
            ax.set_ylim(set_ylim_ax1)
            ax1 = ax.twinx()
            sns.scatterplot(data=self.data[self.data[self.col] == i_column], 
                    x=self.x, y=self.y_2, hue=self.hue, palette=palette_2,
                    legend=legend_2, ax=ax1)
            ax1.set_ylabel(set_ylabel_1)
            ax1.set_ylim(set_ylim_ax2)
            if not len(add_stat) == 0 :
                ax1.text(x=text_x, y=text_y, s=sb_fig._add_plot_stats(self, i_column, add_stat))

        plt.show()
        pass
    # 

    # rel plot #2 death to temp
    def rel_plot_2( self,
                    add_stat = {'age_mean_y_c':'Age mean:', 'inc_adj05_y_c':'Income_mean'},
                    kind='scatter', palette='crest',  
                    set_ylabel_1 = '', set_ylim_ax1 = [0,180],
                    set_ylim_ax2=[0,0.5],
                    stat_ax2='density', binwidth_ax2=1, binrange_ax2=(0,35),
                    col_wrap=3, height=3, aspect=1.2,  legend=False,
                    text_x = -1, text_y = -80):
        sns.set_theme(style=self.style)
        g = sns.relplot(data=self.data, x=self.y_1, y=self.y_2,col=self.col, 
                        hue=self.hue, kind=kind, palette=palette,
                        col_wrap=col_wrap, height=height, aspect=aspect,  legend=False)


        for i_column, ax in g.axes_dict.items():
            ax.set_ylim(set_ylim_ax1)    
            ax1 = ax.twinx()
            sns.histplot(data=self.data[self.data[self.col] == i_column], 
                    x=self.y_1, stat=stat_ax2,   binwidth=binwidth_ax2, binrange=binrange_ax2, #hue=self.hue,palette='flare', 
                    legend=legend,ax=ax1)
            ax1.set_ylabel(set_ylabel_1)
            ax1.set_ylim(set_ylim_ax2)
            ax1.text(x=text_x, y=text_y, s=sb_fig._add_plot_stats(self, i_column, add_stat))
            pass































# def sb_rel_plot_1 (data, col, x, y_1, y_2, add_stat = {'age_mean_y_c':'Age mean:', 'inc_adj05_y_c':'Income_mean'}, hue=self.hue, kind='scatter', palette_1='crest', palette_2='flare', style='dark',col_wrap=3, height=3, aspect=1.2, legend_2=False, set_ylabel_1 = '', set_ylim_1=[0,200]) :
#     sns.set_theme(style=style)
#     g = sns.relplot(data=data, x=x, y=y_1, col=col, 
#                     hue=self.hue, kind=kind, palette=palette_1,
#                     col_wrap=col_wrap, height=height, aspect=aspect,  legend=False)

#     for i_column, ax in g.axes_dict.items():
#         ax1 = ax.twinx()
#         sns.scatterplot(data=data[data[col] == i_column], 
#                 x=x, y=y_2, hue=self.hue, palette=palette_2,
#                 legend=legend_2, ax=ax1)
#         ax1.set_ylabel(set_ylabel_1)
#         ax1.set_ylim(set_ylim_1)
#         if not len(add_stat) == 0 :
#             add_text = ''
#             for key in add_stat.keys():
#                 add_text += f'{add_stat[key]}: {round(data[data[col]==i_column][key].mean() )}\n'
#                 # may need np.nanmean()
#         ax1.text(x=0, y=-40, s=add_text)

#     plt.show()