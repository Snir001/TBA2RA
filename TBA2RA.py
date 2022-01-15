# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3.9.5 64-bit
#     language: python
#     name: python3
# ---

# # Times Buchi Automaton To Region Automaton

# ### Definitions for timed transition table (TTA) $\mathscr{A}$: 
#
# * $\Sigma$ -  alphabet
# * $S$ - set of states
# * $S_0 \subseteq S$ - set of start states
# * $C$ - set of clock
# * $E \subseteq S \times S \times \Sigma \times 2^C \times \Phi (C)$ - set of transitions.
#
# An edge $<s,s',a,\lambda,\delta> $ represents a transition from state $s$ to state $s'$ on input symbol $a$.
#
# the set $\lambda \subseteq C$ are the clock to be reset with this transition, and $\lambda$ is a clock constraint over $C$ 
#
# Example: $<s_0, s_1, a, (x,y) , (x<10,z>2) >$ - transition from $s_0$ to $s_1$ on char $a$ if clock x under 10 and clock z above 2, and clock x and y are reset on this transition

class edge:
    source=""
    target=""
    input=""
    clk_init=""
    condition=""
    def __init__(self,source,target,input,clk_init,condition):
        # set values to vars
        # think how to save 'condition' so we can later evaluate them
        pass
    
    def __str__(self):
        pass



# ## Timed buchi automaton (TBA) 
# $< \Sigma ,S ,S_0 ,C, E ,F > $ wehere $< \Sigma  ,S ,S_0 ,C, E> $ is $\mathscr{A}$ and $F \subseteq S$ are the accepting states

# +
class TBA:
    alfbet=[]
    states=[]
    start=[]
    clks=[]
    edges=[] # use edge class
    accepting=[]

    '''
    those arguments.
    '''
    def __init__(self,alfbet,states,start,clks,edges,accepting):
        pass

    '''
    print function
    '''
    def __str__(self):
        pass
    

# -

# ### Definition for Region Automaton (RA) of times transition table  $R(\mathscr{A})$:
# * states of $R(\mathscr{A})$ of the form $(s,\alpha)$ where $s \in S$ and is a clock region
# * initial states are of the form $(s_0, [v_0])$ where $ss_0 \in S_0$ and $v_0(x)=0$ for all $x \in C$
# * $R(\mathscr{A})$ has an edge $<(s,\alpha),(s',\alpha'),a>$ iff there is an edge $<s,s',a,\lambda,\delta>\in E$ and a region $\alpha''$ such that
#     * $\alpha''$ is a time-successor of $\alpha$
#     * $\alpha''$ satisfies $\delta$
#     * $\alpha=[\lambda \mapsto 0] \alpha''$ (i dont know what it means...)
#
#
#
#

class RA:

    
    def calculate_clks_max_value():
        pass

    def calaculate_time_successor(self,clk_region):
        pass

# some code:

# +
# ast for  ast.literal_eval()
# example:
# >>> ast.literal_eval('[1,"b",3,[3,4]]')
# [1, 'b', 3, [3, 4]]

import ast

'''
read file and return list of lines,
while ignoring lines begin with #
and continuing lines end with /
'''
def read_file(file_path) -> list[str]:
    pass



'''
read TAB object from file and return TAB
the file should contain 6 rows, stantds for the ffollowing variables:
    alfbet= <list of characters>    ; ['a','b','c',...]
    states= <list of states>        ; ['s1','s2','s3',...]
    start= <list of states>         ; ['s2',...]
    clks= <list of clocks names>    ; ['c1','c2',...]
    edges= <list of edges>          ; [('s2','s3','a',['c2',..],['c1 < c2','c2 > 3',...])]
    accepting= <list of states>     ; ['s3',...]
'''
def  read_TAB_from_file(path: str) -> TBA :
    lines=read_file(path)
    TAB_attributes = [ast.literal_eval(v) for v in lines]
    alfbet,states,start,clks,edges,accepting=TAB_attributes
    tba=TBA(alfbet,states,start,clks,edges,accepting)
    
    return tba


# read TBA from file and print it:


