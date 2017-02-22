from model import Model
import tensorflow as tf
from common.shared_functions import dot_or_lookup, glorot_variance, make_tf_variable, make_tf_bias

class GatedMessageGcn(Model):

    onehot_input = True
    use_nonlinearity = True
    vertex_embedding_function = {'train': None, 'test': None}

    relation_dim = 100

    def __init__(self, shape, settings, next_component=None, onehot_input=False, use_nonlinearity=True):
        Model.__init__(self, next_component, settings)
        self.onehot_input = onehot_input
        self.use_nonlinearity = use_nonlinearity
        self.shape = shape

        self.relation_dim = settings['RelationDimension']

    def needs_graph(self):
        return True

    def local_initialize_train(self):
        relation_tensor_shape = (self.relation_count, self.relation_dim)
        relation_var = glorot_variance(relation_tensor_shape)

        self.G_sender = make_tf_variable(0, relation_var, relation_tensor_shape)
        self.G_receiver = make_tf_variable(0, relation_var, relation_tensor_shape)

    def local_get_weights(self):
        return [self.G_sender, self.G_receiver]

    def calculate_message_to_message_matrix(self):
        gate_energies = tf.matmul(self.G_sender, tf.transpose(self.G_receiver))

        message_index_pairs = self.get_graph().get_message_pairs()
        relation_indices = self.get_graph().get_message_type_pairs()

        flat_energies = tf.reshape(gate_energies, [-1, 1])
        energies

