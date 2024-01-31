from search.GAon101.evolve import Evolution, query_fitness, query_fitness_for_indi
from search.GAon101.population import Population
import numpy as np
import copy
from search.GAon101.utils import operations2onehot, GP_log, population_log, NULL
from util.util_data import nasbench2graph_101,generate_101_base_info
from encoder.gin import EncoderGIN
import torch
from nasbench import api
from search.GAon101.individual import Individual
from torch_geometric.data import Data
from torch_geometric.data import Batch


def genotype2phenotype(pops: Population) -> Population:
    genotype_population = pops
    phenotype_population = Population(0)
    for indi in genotype_population.pops:
        matrix = indi.indi['matrix']
        op_list = indi.indi['op_list']
        model_spec = api.ModelSpec(matrix, op_list)
        pruned_matrix = model_spec.matrix
        pruned_op_list = model_spec.ops
        len_operations = len(pruned_op_list)
        assert len_operations == len(pruned_matrix)
        padding_matrix = copy.deepcopy(pruned_matrix)
        if len_operations != 7:
            for j in range(len_operations, 7):
                pruned_op_list.insert(j - 1, NULL)

            padding_matrix = np.insert(pruned_matrix, len_operations - 1,
                                       np.zeros([7 - len_operations, len_operations]), axis=0)
            padding_matrix = np.insert(padding_matrix, [len_operations - 1], np.zeros([7, 7 - len_operations]), axis=1)
        padding_matrix=pruned_matrix
        phenotype_individual = Individual()
        phenotype_individual.create_an_individual(padding_matrix, pruned_op_list)
        phenotype_individual.mean_acc = indi.mean_acc
        phenotype_individual.mean_training_time = indi.mean_training_time
        phenotype_population.add_individual(phenotype_individual)
    return phenotype_population


def get_input_X(pops: Population,ops_dict):
    X = []
    for indi in pops.pops:
        input_metrix = []
        matrix = indi.indi['matrix']
        op_list = indi.indi['op_list']
        edge_idx, node_feature = nasbench2graph_101([matrix, op_list],ops_dict)
        data = Data(x=node_feature, edge_index=edge_idx, dtype=torch.long)
        X.append(data)
    return X


class RF_evolution(Evolution):
    def __init__(self, pretrain_path,search_space,num_estimators, m_prob=0.2, m_num_matrix=1, m_num_op_list=1, x_prob=0.9,
                 population_size=100):
        super(RF_evolution, self).__init__(m_prob, m_num_matrix, m_num_op_list, x_prob, population_size)
        self.num_estimators = num_estimators
        self.archive = Population(0)
        self.pretrain_path=pretrain_path
        self.search_space=search_space

    def load_gin_model(self):
        self.model = EncoderGIN(output_dim=1, re_train=True)
        loaded_paras = torch.load(self.pretrain_path)
        self.model.load_state_dict(loaded_paras)

    def predict_by_predictor(self, pred_pops: Population, phenotype=False):
        if phenotype:
            pred_pops_new = genotype2phenotype(pred_pops)
            X = get_input_X(pred_pops_new,self.search_space.get_ops_dict())
        else:
            X = get_input_X(pred_pops,self.search_space.get_ops_dict())
        self.model.eval()
        with torch.no_grad():
            data=Batch.from_data_list(X)
            pred_y = self.model(data).reshape(-1).numpy()
        for i, indi in enumerate(pred_pops.pops):
            pred_pops.pops[i].mean_acc = pred_y[i]
            pred_pops.pops[i].mean_training_time = 0


if __name__ == '__main__':
    m_prob = 0.2
    m_num_matrix = 1
    m_num_op_list = 1
    x_prob = 0.8
    population_size = 100
    num_generation = 20
    num_resample = 0
    archive_num = 1000
    num_estimators=20
    phenotype = True
    surrogate = True

    total_training_time = 0
    final_one_acc = []
    pretrain_path='../../res/predictor_1000.pt'
    search_space=generate_101_base_info()
    RF_Evolution = RF_evolution(pretrain_path,search_space,num_estimators, m_prob, m_num_matrix, m_num_op_list, x_prob, population_size)
    RF_Evolution.initialize_popualtion(type='RF')
    RF_Evolution.load_gin_model()
    gen_no = 0
    if not surrogate:
        query_fitness(gen_no, RF_Evolution.pops)
    else:
        RF_Evolution.predict_by_predictor(RF_Evolution.pops, phenotype=phenotype)
    while True:
        gen_no += 1
        if gen_no > num_generation:
            break
        offspring = RF_Evolution.recombinate(population_size)
        if not surrogate:
            query_fitness(gen_no, offspring)
        else:
            RF_Evolution.predict_by_predictor(offspring, phenotype=phenotype)

        RF_Evolution.environmental_selection(gen_no, offspring)
    last_resample_num = 1
    sorted_acc_index = RF_Evolution.pops.get_sorted_index_order_by_acc()
    last_population = Population(0)
    for i in sorted_acc_index[:last_resample_num]:
        last_population.add_individual(RF_Evolution.pops.get_individual_at(i))

    gen_no = 'final'
    query_fitness(gen_no, last_population)
    population_log(gen_no, last_population)
    final_one_acc.append(last_population.pops[0].mean_acc)

    save_path = r'pops_log\total_training_time.txt'
    with open(save_path, 'w') as file:
        file.write('Total_training_time: ' + str(total_training_time) + '\n')
        file.write('Total_training_num: ' + str(population_size + num_generation * num_resample) + '\n')
        file.write(
            'm_prob: {}, m_num_matrix: {}, m_num_op_list: {}, x_prob: {}\n'.format(m_prob, m_num_matrix, m_num_op_list,
                                                                                   x_prob))
        file.write('RF_surrogate: True, num_resample: {}, phenotype: {}'.format(num_resample, phenotype))

    print('The ACC of the best individual found is: {}'.format(final_one_acc))
