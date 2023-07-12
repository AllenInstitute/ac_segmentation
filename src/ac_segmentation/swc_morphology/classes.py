# Allen Institute Software License - This software license is the 2-clause BSD
# license plus a third clause that prohibits redistribution for commercial
# purposes without further permission.
#
# Copyright 2017. Allen Institute. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Redistributions for commercial purposes are not permitted without the
# Allen Institute's written permission.
# For purposes of this license, commercial purposes is the incorporation of the
# Allen Institute's software into anything for which you will charge fees or
# other compensation. Contact terms@alleninstitute.org for commercial licensing
# opportunities.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#


from typing import Sequence, Dict
from statistics import mean
import functools
from six import iteritems
import numpy as np
import copy
import queue
import math

#for simpletree class
import operator as op
from collections import defaultdict


#define constants
SOMA = 1
AXON = 2
BASAL_DENDRITE = 3
APICAL_DENDRITE = 4
CUT_DENDRITE = 10
NO_RECONSTRUCTION = 20
TYPE_30 = 30
SPACING = [.1144, .1144, .28]


class SimpleTree( object ):
    def __init__(self, nodes, 
                 node_id_cb, 
                 parent_id_cb):
        '''A tree structure
        
        Parameters
        ----------
        nodes : list of dict
            Each dict is a node in the tree. The keys of the dict name the 
            properties of the node and should be consistent across nodes.
        node_id_cb : function | node dict -> node id
            Calling node_id_cb on a node dictionary ought to produce a unique 
            identifier for that node (we call this the node's id). The type 
            of the node id is up to you, but ought to be consistent across 
            nodes and must be hashable.
        parent_id_cb : function | node_dict => parent node's id
            As node_id_cb, but returns the id of the node's parent.
            
        Notes
        -----
        It is easy to pass a pandas DataFrame as the nodes. Just use the 
        to_dict method of the dataframe like so:
            list_of_dict = your_dataframe.to_dict('record')
            your_tree = SimpleTree(list_of_dict, ...)
        Converting a list of dictionaries to a pandas DataFrame is also very 
        easy. The DataFrame constructor does it for you:
            your_dataframe = pandas.DataFrame(list_of_dict)
             
        '''

        self._nodes = { node_id_cb(n):n for n in nodes }
        self._parent_ids = { nid:parent_id_cb(n) for nid,n in iteritems(self._nodes) }
        self._child_ids = { nid:[] for nid in self._nodes }

        for nid in self._parent_ids:
            pid = self._parent_ids[nid]
            if pid is not None:
                self._child_ids[pid].append(nid)

        self.node_id_cb = node_id_cb
        self.parent_id_cb = parent_id_cb


    def filter_nodes(self, criterion):
        '''Obtain a list of nodes filtered by some criterion
        
        Parameters
        ----------
        criterion : function | node dict => bool
            Only nodes for which criterion returns true will be returned.
            
        Returns
        -------
        list of dict :
            Items are node dictionaries that passed the filter.
        
        '''
    
        return list(filter(criterion, self._nodes.values()))

        
    def value_map(self, from_fn, to_fn):
        '''Obtain a look-up table relating a pair of node properties across 
        nodes
        
        Parameters
        ----------
        from_fn : function | node dict => hashable value
            The keys of the output dictionary will be obtained by calling 
            from_fn on each node. Should be unique.
        to_fn : function | node_dict => value
            The values of the output function will be obtained by calling 
            to_fn on each node.
            
        Returns
        -------
        dict :
            Maps the node property defined by from_fn to the node property 
            defined by to_fn across nodes.

        '''
        
        vm = {}
        for node in self._nodes.values():
            key = from_fn(node)
            value = to_fn(node)
            
            if key in vm:
                raise RuntimeError('from_fn is not unique across nodes. '
                                   'Collision between {0} and {1}.'.format(value, vm[key]))    
            vm[key] = value
  
        return vm


    def nodes_by_property(self, key, values, to_fn=None):
        '''Get nodes by a specified property

        Parameters
        ----------
        key : hashable or function
            The property used for lookup. Should be unique. If a function, will 
            be invoked on each node.
        values : list
            Select matching elements from the lookup.
        to_fn : function, optional
            Defines the outputs, on a per-node basis. Defaults to returning 
            the whole node.
  
        Returns
        -------
        list : 
            outputs, 1 for each input value.

        '''

        if to_fn is None:
            to_fn = lambda x: x

        if not callable( key ):
            from_fn = lambda x: x[key]
        else:
            from_fn = key

        value_map = self.value_map( from_fn, to_fn )
        return [ value_map[vv] for vv in values ]


    def node_ids(self):
        '''Obtain the node ids of each node in the tree
        
        Returns
        -------
        list :
            elements are node ids 
        
        '''
    
        return list(self._nodes)

    
    def parent_id(self, node_ids):
        return self.parent_ids(node_ids)

    
    def parent_ids(self, node_ids):
        '''Obtain the ids of one or more nodes' parents
        
        Parameters
        ----------
        node_ids : list of hashable
            Items are ids of nodes whose parents you wish to find.
        
        Returns
        -------
        list of hashable : 
            Items are ids of input nodes' parents in order.
        
        '''
        
        return [ self._parent_ids[nid] for nid in node_ids ]

   
    def child_ids(self, node_ids):
        '''Obtain the ids of one or more nodes' children
        
        Parameters
        ----------
        node_ids : list of hashable
            Items are ids of nodes whose children you wish to find.
            
        Returns
        -------
        list of list of hashable : 
            Items are lists of input nodes' children's ids.
            
        '''
    
        return [ self._child_ids[nid] for nid in node_ids ]


    def ancestor_ids(self, node_ids):
        '''Obtain the ids of one or more nodes' ancestors
        
        Parameters
        ----------
        node_ids : list of hashable
            Items are ids of nodes whose ancestors you wish to find.
        
        Returns
        -------
        list of list of hashable : 
            Items are lists of input nodes' ancestors' ids.
        
        Notes
        -----
        Given the tree:
        A -> B -> C
         `-> D
          
        The ancestors of C are [C, B, A]. The ancestors of A are [A]. The 
        ancestors of D are [D, A]
        
        '''
    
        out = []
        for nid in node_ids:
        
            current = [nid]
            while current[-1] is not None:
                current.extend(self.parent_ids([current[-1]]))
            out.append(current[:-1])
                
        return out
            
    
    def descendant_ids(self, node_ids):
        '''Obtain the ids of one or more nodes' descendants
        
        Parameters
        ----------
        node_ids : list of hashable
            Items are ids of nodes whose descendants you wish to find.
        
        Returns
        -------
        list of list of hashable : 
            Items are lists of input nodes' descendants' ids.
        
        Notes
        -----
        Given the tree:
        A -> B -> C
         `-> D
          
        The descendants of A are [B, C, D]. The descendants of C are [].
        
        '''
    
        out = []
        for ii, nid in enumerate(node_ids):
            
            current = [nid]
            children = self.child_ids([nid])[0]
            
            if children:
                current.extend(functools.reduce(op.add, map(list, 
                               self.descendant_ids(children))))
                               
            out.append(current)
        return out

    
    def node(self, node_ids=None):
        return self.nodes(node_ids)

    
    def nodes(self, node_ids=None):
        '''Get one or more nodes' full dictionaries from their ids.
        
        Parameters
        ----------
        node_ids : list of hashable
            Items are ids of nodes to be returned. Default is all.
            
        Returns
        -------
        list of dict : 
            Items are nodes corresponding to argued ids.
        '''
    
        if node_ids is None:
            node_ids = self.node_ids()
    
        return [ self._nodes[nid] if nid in self._nodes else None for nid in node_ids]


    def parent(self, node_ids):
        return self.parents(node_ids)


    def parents(self, node_ids):
        '''Get one or mode nodes' parent nodes
        
        Parameters
        ----------
        node_ids : list of hashable
            Items are ids of nodes whose parents will be found.
            
        Returns
        -------
        list of dict : 
            Items are parents of nodes corresponding to argued ids.
    
        '''
        
        return self.nodes([self._parent_ids[nid] for nid in node_ids])


    def children(self, node_ids):
        '''Get one or mode nodes' child nodes
        
        Parameters
        ----------
        node_ids : list of hashable
            Items are ids of nodes whose children will be found.
            
        Returns
        -------
        list of list of dict : 
            Items are lists of child nodes corresponding to argued ids.
        
        '''
    
        return list(map(self.nodes, self.child_ids(node_ids)))


    def descendants(self, node_ids):
        '''Get one or mode nodes' descendant nodes
        
        Parameters
        ----------
        node_ids : list of hashable
            Items are ids of nodes whose descendants will be found.
            
        Returns
        -------
        list of list of dict : 
            Items are lists of descendant nodes corresponding to argued ids.
        
        '''
        
        return list(map(self.nodes, self.descendant_ids(node_ids)))

    
    def ancestors(self, node_ids):
        '''Get one or mode nodes' ancestor nodes
        
        Parameters
        ----------
        node_ids : list of hashable
            Items are ids of nodes whose ancestors will be found.
            
        Returns
        -------
        list of list of dict : 
            Items are lists of ancestor nodes corresponding to argued ids.
        
        '''
    
        return list(map(self.nodes, self.ancestor_ids(node_ids)))



class Morphology(SimpleTree):

    def __init__(self, nodes, node_id_cb, parent_id_cb):

        self._nodes = {node_id_cb(n): n for n in nodes}
        self._parent_id_cb = lambda node: parent_id_cb(node) if parent_id_cb(node) in self._nodes else None
        self._parent_ids = {nid: self._parent_id_cb(n) for nid, n in iteritems(self._nodes)}
        self._child_ids = {nid: [] for nid in self._nodes}
        self.compartments_for_nodes = {}

        for nid in self._parent_ids:
            pid = self._parent_ids[nid]
            if pid is not None:
                self._child_ids[pid].append(nid)

        self.node_id_cb = node_id_cb
        self.parent_id_cb = self._parent_id_cb
        self.nodes_by_types = {}
        self._create_compartment_dictionary()
        self.compartments = self.get_compartments()

    def get_tree_list(self):
        tree_list = []
        tree_roots = self.get_roots()
        for tree_root in tree_roots:
            tree = []
            tree_queue = queue.Queue()
            tree_queue.put(tree_root)
            while not tree_queue.empty():
                root = tree_queue.get()
                tree.append(root)
                children = self.children_of(root)
                if children:
                    for child in children:
                        tree_queue.put(child)
            tree_list.append(tree)
        return tree_list


    def get_children_of_node_by_types(self, node, node_types):
        children = self.children_of(node)
        children_by_types = []
        for child in children:
            if child['type'] in node_types:
                children_by_types.append(child)
        return children_by_types

    def get_children(self, node, node_types=None):
        if node_types:
            return self.get_children_of_node_by_types(node, node_types)
        else:
            return self.children_of(node)


    def get_branching_nodes(self, node_types=None):
        if not node_types:
            nodes = self.get_non_soma_nodes()
        else:
            nodes = self.get_node_by_types(node_types)

        branching_nodes = []
        for node in nodes:
            if len(self.get_children(node)) > 1:
                branching_nodes.append(node)

        return branching_nodes


    def children_of(self, node):
        if node:
            children = self.children([node['id']])
            if children:
                return children[0]
        return None

    def node_by_id(self, node_id):
        return self._nodes[node_id]


    def get_roots(self):
        return self.filter_nodes(lambda node: self.parent_id_cb(node) is None)

    def get_root_id(self):
        return self.node_id_cb(self.get_root())

    def get_roots_for_nodes(self, nodes):
        tree_roots = []
        for node in nodes:
            if self.parent_of(node) not in nodes:
                tree_roots.append(node)
        return tree_roots


    def parent_of(self, node):
        if node:
            parent = self.parents([node['id']])
            if parent:
                return parent[0]
        return None


    def get_leaf_nodes(self, node_types=None):
        if not node_types:
            nodes = self.get_non_soma_nodes()
        else:
            nodes = self.get_node_by_types(node_types)

        leaf_nodes = []
        for node in nodes:
            if not self.get_children(node):
                leaf_nodes.append(node)

    def get_non_soma_nodes(self):
        return self.filter_nodes(lambda node: node['type'] != SOMA)


    def get_node_by_types(self, node_types=None):
        if node_types:
            node_by_types = []
            for node_type in node_types:
                if node_type not in self.nodes_by_types:
                    self.nodes_by_types[node_type] = self.filter_nodes(lambda node: node['type'] == node_type)
                node_by_types += self.nodes_by_types[node_type]
            return node_by_types
        else:
            return self.nodes()

    def _create_compartment_dictionary(self):

        nodes = self.nodes()
        for node in nodes:
            parent = self.parent_of(node)
            if not parent:
                continue
            compartment = [parent, node]
            self.compartments_for_nodes[node['id']] = compartment


    def get_compartments(self, nodes=None, node_types=None):

        if not nodes:
            nodes = self.nodes()
        compartments = []
        for node in nodes:
            node_id = node['id']
            if node_id in self.compartments_for_nodes:
                compartment = self.compartments_for_nodes[node_id]
                if node_types:
                    if compartment[0]['type'] in node_types:
                        compartments.append(compartment)
                else:
                    compartments.append(compartment)
        return compartments





