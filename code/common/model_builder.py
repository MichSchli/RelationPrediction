from encoders.relation_embedding import RelationEmbedding
from encoders.affine_transform import AffineTransform
from decoders.bilinear_diag import BilinearDiag
from extras.graph_representations import Representation

from decoders.nonlinear_transform import NonlinearTransform

from encoders.bipartite_gcn import BipartiteGcn
from encoders.message_gcns.gcn_diag import DiagGcn
from encoders.message_gcns.gcn_basis import BasisGcn
from encoders.message_gcns.gcn_basis_plus_diag import BasisGcnWithDiag
from encoders.message_gcns.gcn_basis_times_diag import BasisGcnTimesDiag



def build_encoder(encoder_settings, triples):
    if encoder_settings['Name'] == "embedding":
        input_shape = [int(encoder_settings['EntityCount']),
                       int(encoder_settings['CodeDimension'])]

        embedding = AffineTransform(input_shape,
                                    encoder_settings,
                                    onehot_input=True,
                                    use_bias=False,
                                    use_nonlinearity=False)

        full_encoder = RelationEmbedding(input_shape,
                                         encoder_settings,
                                         next_component=embedding)

        return full_encoder

    elif encoder_settings['Name'] == "gcn_diag":
        # Define graph representation:
        graph = Representation(triples, encoder_settings)

        # Define shapes:
        input_shape = [int(encoder_settings['EntityCount']),
                       int(encoder_settings['InternalEncoderDimension'])]
        internal_shape = [int(encoder_settings['InternalEncoderDimension']),
                            int(encoder_settings['InternalEncoderDimension'])]
        projection_shape = [int(encoder_settings['InternalEncoderDimension']),
                            int(encoder_settings['CodeDimension'])]

        relation_shape = [int(encoder_settings['EntityCount']),
                          int(encoder_settings['CodeDimension'])]

        layers = int(encoder_settings['NumberOfLayers'])

        # Initial embedding:
        encoding = AffineTransform(input_shape,
                                    encoder_settings,
                                    next_component=graph,
                                    onehot_input=True,
                                    use_bias=True,
                                    use_nonlinearity=True)

        # Hidden layers:
        for layer in range(layers):
            use_nonlinearity = layer < layers - 1
            encoding = DiagGcn(internal_shape,
                               encoder_settings,
                               next_component=encoding,
                               onehot_input=False,
                               use_nonlinearity=use_nonlinearity)

        # Output transform if chosen:
        if encoder_settings['UseOutputTransform'] == "Yes":
            encoding = AffineTransform(projection_shape,
                                       encoder_settings,
                                       next_component=encoding,
                                       onehot_input=False,
                                       use_nonlinearity=False,
                                       use_bias=True)

        # Encode relations:
        full_encoder = RelationEmbedding(relation_shape,
                                         encoder_settings,
                                         next_component=encoding)

        return full_encoder

    elif encoder_settings['Name'] == "gcn_basis":
        graph = Representation(triples, encoder_settings)

        # Define graph representation:
        graph = Representation(triples, encoder_settings)

        # Define shapes:
        input_shape = [int(encoder_settings['EntityCount']),
                       int(encoder_settings['InternalEncoderDimension'])]
        internal_shape = [int(encoder_settings['InternalEncoderDimension']),
                          int(encoder_settings['InternalEncoderDimension'])]
        projection_shape = [int(encoder_settings['InternalEncoderDimension']),
                            int(encoder_settings['CodeDimension'])]

        relation_shape = [int(encoder_settings['EntityCount']),
                          int(encoder_settings['CodeDimension'])]

        layers = int(encoder_settings['NumberOfLayers'])

        # Initial embedding:
        if encoder_settings['UseInputTransform'] == "Yes":
            encoding = AffineTransform(input_shape,
                                       encoder_settings,
                                       next_component=graph,
                                       onehot_input=True,
                                       use_bias=True,
                                       use_nonlinearity=True)
        else:
            encoding = graph

        # Hidden layers:
        for layer in range(layers):
            use_nonlinearity = layer < layers - 1

            if layer == 0 and encoder_settings['UseInputTransform'] == "No":
                onehot_input=True
            else:
                onehot_input=False

            if encoder_settings['AddDiagonal'] == "Yes":
                model = BasisGcnWithDiag
            elif encoder_settings['DiagonalCoefficients'] == "Yes":
                model = BasisGcnTimesDiag
            else:
                model = BasisGcn

            encoding = model(internal_shape,
                               encoder_settings,
                               next_component=encoding,
                               onehot_input=onehot_input,
                               use_nonlinearity=use_nonlinearity)


        # Output transform if chosen:
        if encoder_settings['UseOutputTransform'] == "Yes":
            encoding = AffineTransform(projection_shape,
                                       encoder_settings,
                                       next_component=encoding,
                                       onehot_input=False,
                                       use_nonlinearity=False,
                                       use_bias=True)

        # Encode relations:
        full_encoder = RelationEmbedding(relation_shape,
                                         encoder_settings,
                                         next_component=encoding)

        return full_encoder

    else:
        '''
        elif encoder_settings['Name'] == "bipartite_gcn":
            graph = Representation(triples, encoder_settings, bipartite=True)

            first_layer = BipartiteGcn(encoder_settings, graph)
            second_layer = BipartiteGcn(encoder_settings, graph, next_component=first_layer)
            third_layer = BipartiteGcn(encoder_settings, graph, next_component=second_layer)
            fourth_layer = BipartiteGcn(encoder_settings, graph, next_component=third_layer)

            transform = AffineTransform(fourth_layer, encoder_settings)

            return RelationEmbedding(transform, encoder_settings)
        '''
        return None


def build_decoder(encoder, decoder_settings):
    if decoder_settings['Name'] == "bilinear-diag":
        return BilinearDiag(encoder, decoder_settings)
    elif decoder_settings['Name'] == "nonlinear-transform":
        return NonlinearTransform(encoder, decoder_settings)
    else:
        return None