import json
from graphviz import Digraph
from collections import namedtuple

# Bring in your genotype conversion function:
def genotype_from_config(config):
    # [Insert the genotype_from_config code from above]
    Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
    normal = [("avg_pool_3x3", 0), ("avg_pool_3x3", 0), ("avg_pool_3x3", 2)]
    normal_concat = [3, 4, 5]
    reduce = [("sep_conv_3x3", 0), ("dil_conv_3x3", 0), ("skip_connect", 0)]
    reduce_concat = [3, 4, 5]
    return Genotype(normal=normal, normal_concat=normal_concat,
                    reduce=reduce, reduce_concat=reduce_concat)

def plot(genotype, filename):
    g = Digraph(
        format='pdf',
        edge_attr=dict(fontsize='20', fontname="times"),
        node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
        engine='dot')
    g.body.extend(['rankdir=LR'])

    g.node("c_{k-2}", fillcolor='darkseagreen2')
    g.node("c_{k-1}", fillcolor='darkseagreen2')
    # assert len(genotype) % 2 == 0
    steps = len(genotype) // 2

    for i in range(steps):
        g.node(str(i), fillcolor='lightblue')

    for i in range(steps):
        for k in [2*i, 2*i + 1]:
            op, j = genotype[k]
            if j == 0:
                u = "c_{k-2}"
            elif j == 1:
                u = "c_{k-1}"
            else:
                u = str(j-2)
            v = str(i)
            g.edge(u, v, label=op, fillcolor="gray")

    g.node("c_{k}", fillcolor='palegoldenrod')
    for i in range(steps):
        g.edge(str(i), "c_{k}", fillcolor="gray")

    g.render(filename, view=True)

if __name__ == '__main__':
    # Load your JSON config (you can paste your JSON into a file "sample.json")
    with open("/shared/jingchl6/TG-NAS/home/siemsj/projects/nasbench_201_2/analysis/nb_301_v13/lowpar_training_data/results_0.json", "r") as f:
        data = json.load(f)
    config = data["optimized_hyperparamater_config"]
    genotype = genotype_from_config(config)

    # Visualize the normal cell
    plot(genotype.normal, "result_visualize/normal")
    # Visualize the reduction cell
    plot(genotype.reduce, "result_visualize/reduction")