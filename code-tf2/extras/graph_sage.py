import stellargraph
import pandas as pd
from stellargraph import StellarGraph
from stellargraph.layer import GraphSAGE, MeanAggregator
from stellargraph.mapper import GraphSAGELinkGenerator

"""
Takes in a graph G, depth K, weight matrices W, non-linear function and aggregration
function.
Outputs a representation of G after applying the GraphSAGE algorithm on it.
"""


class GraphSageEmbedder:
    def __init__(self, triplets, batch_size=20, num_samples_per_hop=None, layer_sizes=None):
        if layer_sizes is None:
            layer_sizes = [20, 20]
        if num_samples_per_hop is None:
            num_samples_per_hop = [20, 10]

        # stellargraph representation of node connections
        self.stellar_graph_embeddings = self._getSGEmbeddings(triplets)

        # minibatch size used when computing pairs of nodes to input of graphSAGE
        self.batch_size = batch_size
        # number of samples per hop used in graphSAGE. The length of this array is the number of
        # layers/iterations in the algorithm
        self.num_samples_per_hop = num_samples_per_hop
        self.layers = len(self.num_samples_per_hop)
        # graphsage hidden layer size. The length of this should be equal to the number of layers
        self.layer_sizes = layer_sizes
        assert len(self.layer_sizes) == len(self.num_samples_per_hop)

    def _getSGEmbeddings(self, triplets):
        # https: // stellargraph.readthedocs.io / en / stable / demos / basics / loading - numpy.html  # Non-sequential-graph-structure
        edges = pd.DataFrame(
            {
                "sources": triplets[ :, 0], # Get the first column
                "targets": triplets[ :, 2] # get the thrid column
            }
        )
        return StellarGraph(edges)

    def generate_feature_embeddings(self):
        '''
        for k in range(k):
            for v in vertices:
                h = mean(previous_h, all neigbours of v)
                hv = non_linearity(W.concat(previous_h, h))
            for each v, h_v = hv / l2_norm(h_v)

        return all h_v's
        '''
        # Generate given to the graphSAGE algorithm
        gsage_links = GraphSAGELinkGenerator(self.stellar_graph_embeddings,
                                             self.batch_size, self.num_samples_per_hop)
        gs = GraphSAGE(layer_sizes=self.layer_sizes, generator=gsage_links, bias=False,
                       activations=["relu","softmax"], aggregator=MeanAggregator)
        g_input, g_output = gs.in_out_tensors()
        import tensorflow as tf
        tf.print(g_output[0])
        return g_output
