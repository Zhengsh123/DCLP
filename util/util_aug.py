import copy
import random
import numpy as np
from nasbench import api


class DagAug():
    def __init__(self,aug,search_space,nasbench,optional=1,mode='use'):
        super(DagAug).__init__()
        self.aug_name=aug
        self.optional=optional
        self.nasbench=nasbench
        self.search_space=search_space
        self.mode=mode

    def aug(self,module_adjacency, module_operations):
        if self.aug_name=='edit_node':
            return self.edit_node(module_adjacency,module_operations)
        elif self.aug_name=='edit_edge':
            return self.edit_edge(module_adjacency,module_operations)
        elif self.aug_name=='empty':
            return self.empty(module_adjacency,module_operations)

    def edit_node(self,module_adjacency, module_operations):
        adj = copy.deepcopy(module_adjacency)
        ops = copy.deepcopy(module_operations)
        ops_num=len(ops)
        edit_num = min(self.optional,ops_num-2)
        if ops_num<=3:
            return adj,ops
        change_ops_index= np.random.randint(1, len(ops) - 2, size=edit_num)
        all_ops_list = list(self.search_space.get_ops_dict())
        for remove_obj in['input','output','null','input1','input2']:
            if remove_obj in all_ops_list:
                all_ops_list.remove(remove_obj)
        select_ops_index = np.random.randint(0, len(all_ops_list), size=edit_num)
        select_ops=[all_ops_list[i] for i in select_ops_index]
        for index in range(len(change_ops_index)):
            ops[change_ops_index[index]] = select_ops[index]
        return adj,ops

    def edit_edge(self,module_adjacency, module_operations):
        adj = copy.deepcopy(module_adjacency)
        ops = copy.deepcopy(module_operations)
        edit_num = self.optional
        ops_num=len(ops)
        if ops_num<=3:
            return adj,ops
        for i in range(edit_num):
            flag = False
            while not flag:
                row = random.randint(1, ops_num - 3)
                col = random.randint(row+1, ops_num - 2)
                adj[row][col] = 1 - adj[row][col]
                if self.mode=='edit':
                    break
                model_spec = api.ModelSpec(matrix=adj, ops=ops)
                flag = self.nasbench.is_valid(model_spec)
        return adj,ops

    def empty(self,module_adjacency, module_operations):
        adj = copy.deepcopy(module_adjacency)
        ops = copy.deepcopy(module_operations)
        return adj,ops


