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


from copy import copy, deepcopy
import PySimpleAutomata
from PySimpleAutomata import automata_IO, NFA
import visual_automata
from visual_automata.fa.dfa import VisualDFA
import graphviz

g_id=-1
def get_id():
    global g_id
    g_id=g_id+1 
    return g_id


class Edge:
    source = ""
    target = ""
    input = ""
    clk_init = []
    conditions = []

    def __init__(self, source, target, input, clk_init, conditions):
        self.id=get_id()
        self.source = source
        self.target = target
        self.input = input
        self.clk_init = clk_init
        self.conditions = conditions
        # set values to vars
        # think how to save 'condition' so we can later evaluate them
        pass

    def __str__(self):
        string = "\t\tSource = %s\n" % self.source
        string += "\t\tTarget = %s\n" % self.target
        string += "\t\tInput = %s\n" % self.input
        string += "\t\tclk_init = %s\n" % self.clk_init
        string += "\t\tConditions = %s\n" % str(self.conditions)
        return string


# ## Timed buchi automaton (TBA)
# $< \Sigma ,S ,S_0 ,C, E ,F > $ wehere $< \Sigma  ,S ,S_0 ,C, E> $ is $\mathscr{A}$ and $F \subseteq S$ are the accepting states


class TBA:
    alphabet = []
    states = []
    start = []
    clks = []
    edges = []  # use edge class
    accepting = []

    '''
    those arguments.
    '''

    def __init__(self, alphabet, states, start, clks, edges, accepting):
        self.id=get_id()
        self.alphabet = alphabet
        self.states = states
        self.start = start
        self.clks = clks
        for new_edge in edges:
            new_edge_vals = new_edge.split(":")
            source = new_edge_vals[0]
            target = new_edge_vals[1]
            input = new_edge_vals[2]
            clks_init = new_edge_vals[3].split("&")
            conditions = new_edge_vals[4].split("&")
            edge = Edge(source, target, input, clks_init, conditions)
            self.edges.append(edge)
        self.accepting = accepting
        pass

    '''
    print function
    '''

    def __str__(self):
        string = "Alphabet = %s\n" % self.alphabet
        string += "States = %s\n" % self.states
        string += "Start = %s\n" % self.start
        string += "Clocks = %s\n" % self.clks
        string += "Edges =\n"
        for i, an_edge in enumerate(self.edges):
            string += "\tEdge %s:\n" % str(i + 1)
            string += self.edges[i].__str__()
        string += "\nAccepting = %s\n" % self.accepting
        return string


class Clk_status:
    """
    this class is the state of each clock and the relation between them
    """
    # integral -  ["<",<int>] ||  ["=",<int>] ||  ["<<",<small int>,<big int>]
    integral = []

    # *_fract = ["clk1_name","clk2_name",...]
    bigger_fract = []
    equal_fract = []
    smaller_fract = []

    def __init__(self, integral, bigger_fract, equal_fract, smaller_fract):
        self.id=get_id()
        self.integral = integral
        self.bigger_fract = bigger_fract
        self.equal_fract = equal_fract
        self.smaller_fract = smaller_fract

    def __eq__(self, value):
        if value.integral == self.integral and value.bigger_fract == self.bigger_fract \
                and value.equal_fract == self.equal_fract and value.smaller_fract == self.smaller_fract:
            return True
        return False

    def __key(self):
        return (self.integral, self.bigger_fract, self.equal_fract, self.smaller_fract)

    def __hash__(self):
        return hash(self.__key())

    def __str__(self):
        string = f"{self.integral}, <{self.bigger_fract} , ={self.equal_fract} , >{self.smaller_fract}"
        return string


'''
contains list o clocks statuses
    its better to use a dict to represent this by dict:
    {
        <clk_name1> : <clk_status1>
        <clk_name2> : <clk_status2>
        ....
        "x" : <clk_status>
    }
'''


class Clk_region(dict):
    # def __str__(self):
    #     return str(self)

    def __init__(self, *args, **kwargs ):
        self.id=get_id()
        dict.__init__(self, *args, **kwargs )


    def __str__(self):
        string=""
        for clk, value in self.items():
            string= string+str(clk)+":"+str(value)+","
        return string

    def satisfy(self, conditions: list):
        """
        gets self and list of conditions (from the edges)
        return true if self satesfy all the conditions
        """

        for single_condition in conditions:  # a>b / a<b / a=b
            if single_condition == '':
                continue
            else:
                clock_name, clock_cond_val = re.split("[<>=]{1,2}", single_condition)
                clock_integral = self.get(clock_name).integral
                if clock_integral[0] == '=':
                    z = clock_integral[1]
                elif clock_integral[0] == '>':
                    z = clock_integral[1] + 1
                elif clock_integral[0] == '<<':
                    lower = clock_integral[1]
                    upper = clock_integral[2]
                    z = (lower + upper) / 2
                else:
                    print("Oh, Fatal Error")

                sign = re.search("[<>=]{1,2}", single_condition).group(0)
                if sign == "=":
                    sign= "=="

                eval_exp = str(z) + sign + clock_cond_val
                if not eval(eval_exp):
                    return False

        return True

    def init_clocks(self, clks: list):
        """
        gets list of clocks and initiate them:
        1) make them equal to zero
        2) change bigger, equal, smaller fract to be right
        """
        # make copy of self:
        new_clk_region= deepcopy(self)


        for clk in new_clk_region:
            if clk in clks:
                new_clk_region[clk].integral = ['=', 0]
                new_clk_region[clk].smaller_fract = []
                new_clk_region[clk].bigger_fract = []
                new_clk_region[clk].equal_fract = []

                for other_clk in new_clk_region:
                    if clk == other_clk:  # ignores himself
                        continue
                    elif other_clk in clks:  # if the other clock also initiated
                        new_clk_region[clk].equal_fract.append(other_clk)
                    else:  # if the other clock was not initiated
                        new_clk_region[clk].bigger_fract.append(other_clk)  # other clock is bigger than this clock
                        # TODO: check about clock that was initiated in the past
                        if clk in new_clk_region[other_clk].bigger_fract:  # if this clock listed as bigger than other
                            new_clk_region[other_clk].bigger_fract.remove(clk)  # remove initiated clock from bigger list
                        elif clk in new_clk_region[other_clk].equal_fract:  # if this clock listed as equal to other
                            new_clk_region[other_clk].equal_fract.remove(clk)  # remove initiated clock from equal list

                        if clk not in new_clk_region[other_clk].smaller_fract:  # if clock is not in the smaller list of other
                            new_clk_region[other_clk].smaller_fract.append(clk)  # add it to list
        return new_clk_region


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


class Extended_state:
    state = ""
    id=None

    # dict of Clk_status
    clk_region = None

    def __init__(self, state, clk_region):
        self.id=get_id()
        self.state = state
        self.clk_region = clk_region

    def __eq__(self, value):
        if self.state == value.state and self.clk_region == value.clk_region:
            return True

        return False

    def __str__(self):
        return f"{self.state}: {self.clk_region}"


# regular expression to easily seperate condition arguments

import re


# print pages 19 to 25

class RA:
    # contain the max number that a clock is compared to
    # structure:
    # <clock_name>: <max value:int>
    max_clock = {}
    tba = None
    alphabet = []
    ex_states = []
    ex_start = []
    # ex_start = {}
    clks = []
    edges = []  # use edge class
    accepting = []
    graph = None

    def __init__(self, tba: TBA) -> None:
        self.id=get_id()
        self.tba = tba
        self.alphabet = tba.alphabet
        self.clks = tba.clks

        self.calculate_clks_max_value()
        self.calculate_start_states()
        succ = self.calculate_time_successor(self.ex_start[0].clk_region)
        for time_reg in succ:
            if succ.count(time_reg) > 1:
                succ.remove(time_reg)


        #check this basterd:
        # x:['<<', 0, 1], <[] , =[] , >['y'],y:['=', 0], <['x'] , =[] , >[],
        test={  "x":Clk_status(['<<', 0, 1], [], [], ['y']) ,  
                "y":Clk_status(['=', 0], ['x'] , [] , [])
            }

        succ = self.calculate_time_successor(test)

        self.graph = self.BFS(self.ex_start[0])
        print("done")

        # go through the edges of start, and build their neighbors ex_states.
        # while there are new states, go through them and go though their edges.

        pass

    def calculate_clks_max_value(self):
        """
        calculate max number of condition of each clock
        """

        for edge in self.tba.edges:
            for condition in edge.conditions:
                if condition == "":
                    continue
                # seperate condition args
                condition_args = re.split("[<>=]{1,2}", condition)

                # check where is the number:
                if condition_args[0].isdigit():
                    number = int(condition_args[0])
                    clock_name = condition_args[1]
                elif condition_args[1].isdigit():
                    number = int(condition_args[1])
                    clock_name = condition_args[0]
                else:
                    # just names. continue
                    continue

                # get current max value (set to 0 if not defined)
                current_max = self.max_clock.get(clock_name, 0)

                if number > current_max:
                    current_max = number

                self.max_clock[clock_name] = current_max
            # TODO?: after we ffill the numbers, we need to figure out what to do with the clock with no number.

    def calculate_start_states(self):
        """
        calculate_start_states - create the start clock region (all clock zero and start state < s0,[v0] >)
        """
        # TODO: fill fract parts
        start_states = self.tba.start
        start_clk_regions = []
        # start_clk_regions = {}

        for s in start_states:
            clk_constraints_dict = Clk_region()
            # build constraint:
            for clk in self.tba.clks:
                all_clks = self.tba.clks
                all_but_himself = [x for x in all_clks if x != clk]
                clk_constraint = Clk_status(["=", 0], [], all_but_himself, [])
                clk_constraints_dict[clk] = clk_constraint

            # create start Clk_region:
            start_clk_regions.append(Extended_state(s, clk_constraints_dict))
            # start_clk_regions[s] = clk_constraints_dict

        self.ex_start = start_clk_regions

    def calculate_time_successor(self, clk_region: Clk_region):
        max_clocks_values = self.max_clock
        new_clock_region = Clk_region()

        # if region is c>cx for all clocks, return itself.
        all_bigger_than_max = True
        for clk_name, clk_status in clk_region.items():
            # for clk_status in clk_region.clk_status_list:
            if clk_status.integral[0] != '>':  #cc
                all_bigger_than_max = False
                break

        # recurtion break
        if all_bigger_than_max:
            return [clk_region]

        # case some of clocks are equal to integer
        some_equal_to_number = False
        for clk_name, clk_status in clk_region.items():
            if clk_status.integral[0] == '=':
                some_equal_to_number = True
                break

        if some_equal_to_number:
            # build clock_status for clocks:
            for clk_name, clk_status in clk_region.items():
                if clk_status.integral[0] == '=':
                    # if equal to max value of clock
                    if max_clocks_values[clk_name] == clk_status.integral[1]:
                        new_clock_region[clk_name] = Clk_status([">", clk_status.integral[1]],  #cc
                                                                clk_status.bigger_fract, clk_status.equal_fract,
                                                                clk_status.smaller_fract)
                    if max_clocks_values[clk_name] > clk_status.integral[1]:
                        # clock is compared to less then his max value
                        new_clock_region[clk_name] = Clk_status(["<<", clk_status.integral[1],
                                                                 clk_status.integral[1] + 1],
                                                                clk_status.bigger_fract, clk_status.equal_fract,
                                                                clk_status.smaller_fract)
                if clk_status.integral[0] != '=':
                    new_clock_region[clk_name] = clk_status

            return_regions = [new_clock_region] + self.calculate_time_successor(new_clock_region)
            for time_reg in return_regions:
                if return_regions.count(time_reg) > 1:
                    return_regions.remove(time_reg)
            return return_regions

        # case no one equal and not all above max
        # in this case' the time succesor incuding the clk_region itself, his time succesor, and his time succesor
        else:
            maximal_fract_clocks=[]
            for clk_name, clk_status in clk_region.items():
                if clk_status.integral[0] != '>':  #cc
                    # not bigger than max value
                    # check if has maximal fractional part between all the clock
                    # that smaller than thier max value. so we make sure all the clocks in the bogger_fract are above their max value:
                    have_maximal_fract = True
                    for bigger_fract_clk in clk_status.bigger_fract:
                        # check that all the clocks with bigger fract are bigger than their max value
                        if clk_region[bigger_fract_clk].integral[0] != '>':  #cc
                            have_maximal_fract = False
                            break
                    if have_maximal_fract:
                        maximal_fract_clocks.append(clk_name)

            # here we need to adjust the fract part!!
            # for each clock with maximal fract, make it equal to nnext integer and 
            # and change his fract to be smallest, and other fratcs smaller to incllude him:
            new_clock_region= clk_region.init_clocks(maximal_fract_clocks)
            for clk_name in maximal_fract_clocks:
                new_clock_region[clk_name].integral=['=',clk_region[clk_name].integral[2]]

            # in this case' the time succesor incuding the clk_region itself, his time succesor, and his time succesor



            return_regions= [clk_region, new_clock_region] + self.calculate_time_successor(new_clock_region)
            for time_reg in return_regions:
                if return_regions.count(time_reg) > 1:
                    return_regions.remove(time_reg)
            return return_regions

    '''
    input: tba, extended state,
    output:the next extended.
    '''

    def next_states(self, start_ex_state: Extended_state):
        # R(A) has an edge <<s,c>, <s',c'>, a> iff there is an edge 
        # <s,s',a,lambda,delta> in tba.E and a region c'' such that:
        #   1) c'' is a time successor of c
        #   2) c'' satisfies delta
        #   3) c' = [lambda->0]c''

        next_ex_states = []
        ex_edges = []
        # get state name and edges going out of it:
        s0 = start_ex_state.state
        s0_edges = [e for e in self.tba.edges if e.source == s0]

        # get time successors of start_ex_state
        time_suc_list = self.calculate_time_successor(start_ex_state.clk_region)

        for e in s0_edges:
            # check time_suc satisfies condition:
            for time_suc in time_suc_list:
                if time_suc.satisfy(e.conditions):
                    # edit time_suc to fit lambda (zero the clocks need to be zeroed)

                    ex_state = Extended_state(e.target, time_suc.init_clocks(e.clk_init))
                    # ex_state = time_suc.init_clocks(e.clk_init)
                    next_ex_states.append(ex_state)

                    ex_edges.append((start_ex_state, ex_state, e.input))

        return (next_ex_states, ex_edges)

    def BFS(self, start:Extended_state):

        # Mark all the vertices as not visited

        # visited = [False] * (max(self.graph) + 1)

        # Create a queue for BFS
        queue = []
        visited = []
        edge_list = []

        # Mark the source node as visited and enqueue it
        queue.append(start)
        visited.append(start)
        self.ex_states.append(start)

        while queue:

            # Dequeue a vertex from queue and print it
            s = queue.pop(0)
            print(s)

            # Get all adjacent vertices of the dequeued vertex s.
            adjacents, edges = self.next_states(s)
            # print(s)

            # need to inset edge only if her node is in.
            edge_list.extend(edges)
            # If a adjacent has not been visited, then mark it
            # visited and enqueue it
            for a in adjacents:
                # add edge to list
                if a not in visited:
                    queue.append(a)
                    visited.append(a)
                    if a not in self.ex_states:
                        # id_counter=id_counter+1
                        # a.id=id_counter
                        self.ex_states.append(a)

        return (visited, edge_list)

    def build_all(self):
        pass

    def __str__(self):
        pass


def ex_state_to_full_state(ex_state):
    state = ex_state.state
    region = ex_state.clk_region
    # return ex_state.state + "\n" + str(ex_state.clk_region)
    data = ""
    for clk in region:
        integral = region[clk].integral
        if integral[0] == '=':
            data += clk + '_eq_' + str(integral[1])
        elif integral[0] == '>':
            data += clk + '_less_' + str(integral[1])
        elif integral[0] == '<<':
            data += clk + '_from_' + str(integral[1]) + "_to_" +str(integral[2])

    return f'{state}_{data}', f'{state}_{data}'


def ex_edge_to_full_edge(ex_edge):
    start = ex_state_to_full_state(ex_edge[0])[0]
    char = ex_edge[2]
    target = ex_state_to_full_state(ex_edge[1])[0]
    return start, char, target


def print_automata(ra):
    ex_states = ra.graph[0]
    ex_edges = ra.graph[1]
    initial_state = ex_state_to_full_state(ra.ex_start[0])
    full_states = []
    for ex_state in ex_states:
        full_state = ex_state_to_full_state(ex_state)
        full_states.append(full_state)

    full_edges = []
    for ex_edge in ex_edges:
        start, char, target = ex_edge_to_full_edge(ex_edge)
        full_edges.append([start, target, char])
    ####generate NFA########
    nfa_str = "digraph{\n\tfake [style=invisible]\n"
    nfa_str += 'node [margin=0 fontcolor=blue fontsize=32 width=0.5 shape=box style=filled]'
    nfa_str += f"\t{initial_state[0]} [root=true label={initial_state[1]}]\n"
    nfa_str += f"\tfake -> {initial_state[0]} [style=bold]\n\n"

    for state in full_states:
        nfa_str += f'\t{state[0]} [label={state[1]}]\n'
    nfa_str += "\n"

    for edge in full_edges:
        nfa_str += f"\t{edge[0]} -> {edge[1]} [label=\"{edge[2]}\"]\n"
    nfa_str += "}"
    print(nfa_str)
    file = open('nfa.txt', 'w')
    file.write(nfa_str)
    file.close()


    #######################

    nfa = automata_IO.nfa_dot_importer('nfa.txt')
    automata_IO.nfa_to_dot(nfa, 'dot_file')



def print_automata_old(ra):
    ex_states = ra.graph[0]
    ex_edges = ra.graph[1]
    full_states = set()
    full_edges = {}
    for ex_state in ex_states:
        full_state = ex_state_to_full_state(ex_state)
        full_states.add(full_state)

    for ex_edge in ex_edges:
        key, char, value = ex_edge_to_full_edge(ex_edge)
        if key in full_edges:
            full_edges[key].update({char: value})
        else:
            full_edges[key] = {char: value}

    input_symbols = set(ra.alphabet)
    initial_state = ex_state_to_full_state(ra.ex_start[0])

    dfa = VisualDFA(
        states=full_states,
        input_symbols=input_symbols,
        transitions=full_edges,
        initial_state=initial_state,
        final_states=set()
    )

    dfa = VisualDFA(dfa)
    dfa.show_diagram(view=True)
    print("hi")


'''
read file and return list of lines,
while ignoring lines begin with #
and continuing lines end with /
return list of strings
-> list[str]
'''


def read_file(file_path):
    file = open(file_path)
    lines = file.readlines()
    return lines


automaton1 = """\
alphabet = a, b, c, d
states = s0, s1, s2, s3
start = s0
clks = x, y
edges=s0:s1:a:y:, s1:s2:b::y=1, s1:s3:c::x<1, s2:s3:c::x<1, s3:s3:d::x>1, s3:s1:a:y:y<1
accepting=s3\
"""

automaton2 = """\
alphabet = a, b, c, d
states = s0, s1, s2, s3
start = s0
clks = x, y
edges=s0:s1:a:y:, s1:s2:b::y=1, s1:s3:c::x<1, s2:s3:c::x<1, s3:s3:d::x>1, s3:s1:a:y:y<1
accepting=s3\
"""


def read_TBA_from_text(text: str) -> TBA:
    """
    read TAB object from file and return TAB
    the file should contain 6 rows, stantds for the ffollowing variables:
        alfbet= <list of characters>    ; a, b, c, ...
        states= <list of states>        ; s1, s2, s3, ...
        start= <list of states>         ; s2, ...
        clks= <list of clocks names>    ; c1, c2, ...
        edges= s:s':a:lambda:delta      ; s2:s3:a:c2:c1<c2&c2>3, s1:s3:b:c1:c2<c1&c3>3, ...
        accepting= <list of states>     ; s3, ...
    """
    alphabet, states, start, clks, edges, accepting = "", "", "", "", "", ""
    # read TBA from file and print it:
    lines = text.split('\n')
    for i, line in enumerate(lines):
        line = line.replace("\n", "")  # remove \n
        line = line.replace(" ", "")  # remove spaces
        line = line.split("#")[0]  # ignore comments
        lines[i] = line  # update lines list

        variable, value = line.split("=", 1)
        if i == 0 and variable == "alphabet":
            alphabet = value.split(",")
        elif i == 1 and variable == "states":
            states = value.split(",")
        elif i == 2 and variable == "start":
            start = value.split(",")
        elif i == 3 and variable == "clks":
            clks = value.split(",")
        elif i == 4 and variable == "edges":
            edges = value.split(",")
        elif i == 5 and variable == "accepting":
            accepting = value.split(",")
        else:
            print("Error at variable: %s " % variable)
            exit(-1)

    tba = TBA(alphabet, states, start, clks, edges, accepting)
    print(tba)
    return tba


# read TBA from file and print it:
tba = read_TBA_from_text(automaton1)
ra = RA(tba)
print_automata(ra)
print("hi")
# print(ra)
