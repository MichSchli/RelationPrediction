from encoders.relation_embedding import RelationEmbedding
from encoders.affine_transform import AffineTransform
from decoders.bilinear_diag import BilinearDiag
from extras.graph_representations import Representation

from decoders.nonlinear_transform import NonlinearTransform

from encoders.bipartite_gcn import BipartiteGcn

from encoders.message_gcns.gcn_diag import DiagGcn

from encoders.message_gcns.gcn_basis import BasisGcn



def build_encoder(encoder_settings, triples):
    if encoder_settings['Name'] == "embedding":
        return RelationEmbedding(Embedding(encoder_settings), encoder_settings)
    elif encoder_settings['Name'] == "gcn":
        graph = Representation(triples, encoder_settings)

        first_layer = GCN(encoder_settings, graph, next_component=graph, onehot_input=True)
        second_layer = GCN(encoder_settings, graph, next_component=first_layer)
        transform = LinearTransform(second_layer, encoder_settings)
        return RelationEmbedding(transform, encoder_settings)
    elif encoder_settings['Name'] == "gcn_diag":
        graph = Representation(triples, encoder_settings)

        input_shape = [int(encoder_settings['EntityCount']),
                       int(encoder_settings['InternalEncoderDimension'])]

        embedding = AffineTransform(input_shape,
                                    encoder_settings,
                                    next_component=graph,
                                    onehot_input=True,
                                    use_bias=True,
                                    use_nonlinearity=True)

        first_layer = DiagGcn(encoder_settings, graph, next_component=embedding, onehot_input=False, use_nonlinearity=False)
        #second_layer = BasisGcn(encoder_settings, graph, next_component=first_layer, use_nonlinearity=False)
        #transform = LinearTransform(first_layer, encoder_settings)
        return RelationEmbedding(first_layer, encoder_settings)

    elif encoder_settings['Name'] == "gcn_basis":
        graph = Representation(triples, encoder_settings)

        #embedding = Embedding(encoder_settings, next_component=graph)
        first_layer = BasisGcn(encoder_settings, graph, next_component=graph, onehot_input=True, use_nonlinearity=False)
        #second_layer = BasisGcn(encoder_settings, graph, next_component=first_layer, use_nonlinearity=False)
        #transform = LinearTransform(first_layer, encoder_settings)
        return RelationEmbedding(first_layer, encoder_settings)

    elif encoder_settings['Name'] == "gcn_diag_sigmoid_gate":
        graph = Representation(triples, encoder_settings)

        first_layer = DiagGcnSigmoidGate(encoder_settings, graph, next_component=graph, onehot_input=True)
        #second_layer = DiagGcnSigmoidGate(encoder_settings, graph, next_component=first_layer)
        #transform = LinearTransform(second_layer, encoder_settings)
        return RelationEmbedding(first_layer, encoder_settings)

    elif encoder_settings['Name'] == "gated_basis_gcn":
        graph = Representation(triples, encoder_settings)

        first_layer = GatedBasisGcn(encoder_settings, graph)
        #second_layer = GatedBasisGcn(encoder_settings, graph, next_component=first_layer)
        #transform = LinearTransform(second_layer, encoder_settings)
        return RelationEmbedding(first_layer, encoder_settings)

    elif encoder_settings['Name'] == "simple_gated_basis_gcn":
        graph = Representation(triples, encoder_settings)

        first_layer = SimpleGatedBasisGcn(encoder_settings, graph)
        second_layer = SimpleGatedBasisGcn(encoder_settings, graph, next_component=first_layer)

        transform = LinearTransform(second_layer, encoder_settings)

        return RelationEmbedding(transform, encoder_settings)
    elif encoder_settings['Name'] == "bipartite_gcn":
        graph = Representation(triples, encoder_settings, bipartite=True)

        first_layer = BipartiteGcn(encoder_settings, graph)
        second_layer = BipartiteGcn(encoder_settings, graph, next_component=first_layer)
        third_layer = BipartiteGcn(encoder_settings, graph, next_component=second_layer)
        fourth_layer = BipartiteGcn(encoder_settings, graph, next_component=third_layer)

        transform = LinearTransform(fourth_layer, encoder_settings)

        return RelationEmbedding(transform, encoder_settings)

    else:
        return None

def build_decoder(encoder, decoder_settings):
    if decoder_settings['Name'] == "bilinear-diag":
        return BilinearDiag(encoder, decoder_settings)
    elif decoder_settings['Name'] == "nonlinear-transform":
        return NonlinearTransform(encoder, decoder_settings)
    else:
        return None