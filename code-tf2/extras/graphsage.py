from stellargraph.data import UnsupervisedSampler
from tensorflow import keras
import pandas as pd
from model import Model
from stellargraph import StellarGraph, IndexedArray
from stellargraph.layer import GraphSAGE, MeanAggregator, link_classification
from stellargraph.mapper import GraphSAGELinkGenerator, GraphSAGENodeGenerator
import numpy as np
"""
Takes in a graph G, depth K, weight matrices W, non-linear function and aggregration
function.
Outputs a representation of G after applying the GraphSAGE algorithm on it.
"""


class GraphSageEmbedding(Model):
    embedding_width = None

    W = None
    b = None
    shape = None

    stored_W = None

    def __init__(self, entities, shape, settings,
                    next_component,
                    batch_size=20,
                    num_samples_per_hop=None,
                    layer_sizes=None):
        Model.__init__(self, next_component, settings)
        self.entities = entities
        self.shape = shape
        if layer_sizes is None:
            layer_sizes = [self.shape[1], self.shape[1]]
        if num_samples_per_hop is None:
            num_samples_per_hop = [20, 10]

        # minibatch size used when computing pairs of nodes to input of graphSAGE
        self.batch_size = batch_size

        # number of samples per hop used in graphSAGE. The length of this array is the number of
        # layers/iterations in the algorithm
        self.num_samples_per_hop = num_samples_per_hop
        self.layers = len(self.num_samples_per_hop)

        # graphsage hidden layer size. The length of this should be equal to the number of layers
        self.layer_sizes = layer_sizes
        assert len(self.layer_sizes) == len(self.num_samples_per_hop)

    def _get_one_hot_encoded(self, triplets):
        """
        Returns OH encoding for each node in the graph.
        For each node, len(onehot_vector) = number of edges
        If a node has an edge, that edge's one hot value is set to 1.
        """
        edges = triplets[:, 1]      # Get all unique edges that exist between two nodes
        number_of_edges = len(set(edges))

        # Create a 2D array of (number_of_nodes, number_of_edges) to hold OH encoding for each node
        # result = np.zeros([len(self.entites), number_of_edges])
        # Create a dict for each node in the dataset
        result = {key: [0] * number_of_edges for key in range(0, len(self.entities))}

        # For each node, a 1 is added to the one hot vector if it has an edge
        for n1, e, n2 in triplets:
            result[n1][e] = 1
            result[n2][e] = 1

        return result

    def _get_stellargraph_embeddings(self, triplets):
        # https: // stellargraph.readthedocs.io / en / stable / demos / basics / loading - numpy.html  # Non-sequential-graph-structure
        edges = pd.DataFrame(
            {
                "source": triplets[ :, 0 ], # get the first column (source nodes)
                "target": triplets[ :, 2 ], # get the third column  (destination nodes)
            }
        )

        # One hot node feature vector
        one_hot_node_features_dict = self._get_one_hot_encoded(triplets)
        oh_dataframe = pd.DataFrame(one_hot_node_features_dict).transpose()

        return StellarGraph(oh_dataframe, edges)

    def generate_feature_embeddings(self):
        '''
        for k in range(k):
            for v in vertices:
                h = mean(previous_h, all neigbours of v)
                hv = non_linearity(W.concat(previous_h, h))
            for each v, h_v = hv / l2_norm(h_v)

        return all h_v's
        '''
        # stellargraph representation of node connections
        stellar_graph_data = self._get_stellargraph_embeddings(self.next_component.triples)
        print(f'Stellargraph data: {stellar_graph_data.info()}')

        generator = GraphSAGELinkGenerator(stellar_graph_data,
                                             self.batch_size, self.num_samples_per_hop)

        gs = GraphSAGE(layer_sizes=self.layer_sizes, generator=generator, bias=False,
                       activations=["relu", "softmax"], aggregator=MeanAggregator)

        x_inp, x_out = gs.in_out_tensors()

        # nodes = list(stellar_graph_data.nodes())
        # number_of_walks = 1
        # length = 5
        # unsupervised_samples = UnsupervisedSampler(
        #     stellar_graph_data, nodes=nodes, length=length,
        #     number_of_walks=number_of_walks)

        # train_gen = generator.flow(unsupervised_samples)
        # prediction = link_classification(output_dim=1, output_act="sigmoid",
        #                                  edge_embedding_method="ip")(x_out)
        #
        # model = keras.Model(inputs=x_inp, outputs=prediction)
        # model.compile(
        #     optimizer=keras.optimizers.Adam(lr=1e-3),
        #     loss=keras.losses.binary_crossentropy,
        #     metrics=[keras.metrics.binary_accuracy],
        # )
        #
        # history = model.fit(
        #     train_gen,
        #     epochs=1,
        #     verbose=1,
        #     use_multiprocessing=False,
        #     workers=4,
        #     shuffle=True,
        # )


        # Get list of odd elements of x_inp
        x_inp_src = x_inp[0::2]
        x_out_src = x_out[0]
        embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)
        # node_ids = self.next_component.triples[ :, 0 ]      # source ids
        node_ids = np.array(list(range(len(self.entities))))

        batch_size = 50
        node_gen = GraphSAGENodeGenerator(stellar_graph_data, batch_size, self.num_samples_per_hop).flow(node_ids)
        node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=1)

        # This produces the embeddings for both the nodes, 
        # which then is passed to the message passing algorithms
        return node_embeddings

    def get_all_codes(self, mode='train'):
        # This function is a function of the class Model that is used to get the weights
        # in the training pipeline. 
        # We overload this function here, and plug in the weights of the edges.
        g_output = self.generate_feature_embeddings()
        return g_output, None, g_output
        # should return (14541,500) shape