from genotypes import Genotype,PRIMITIVES
import numpy as np
import pickle


def sample_darts_rnn_arch(available_ops):
    geno = []
    geno.extend([(np.random.choice(available_ops, 1)[0],0)])
    for i in range(1,8):
        ops_normal = np.random.choice(available_ops, 1)
        nodes_in_normal = np.random.choice(range(1,i + 1), 1)
        geno.extend([(ops_normal[0], nodes_in_normal[0])])
    genotype = Genotype(recurrent=geno, concat=range(1, 9))
    return genotype


if __name__=="__main__":
    temp=[]
    for i in range(1,16):
        file_name='../torch_rnn/rnn_data/'+str(i)+'.pkl'
        print(file_name)
        file = open(file_name, 'rb')
        data = pickle.load(file)
        temp.extend(data)

    res=[]
    while len(res)!=300:
        cell=sample_darts_rnn_arch(PRIMITIVES[1:])
        if cell in res or cell in temp:
            continue
        else:
            res.append(cell)
    # file_name = '../torch_rnn/rnn_data/test.pkl'
    # with open(file_name,'wb')as f:
    #     pickle.dump(res,f)
    for i in range(16):
        file_name='../torch_rnn/rnn_data/'+str(i)+'.pkl'
        with open(file_name, "wb") as f:
            pickle.dump(res[i*16:(i+1)*16], f)
