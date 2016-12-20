from encoders.embedding import Embedding
from encoders.gated_basis_gcn import GatedBasisGcn
from encoders.relation_embedding import RelationEmbedding
from encoders.linear_transform import LinearTransform
from decoders.bilinear_diag import BilinearDiag
from common.graph_representations import Representation

def build_encoder(encoder_settings, triples):
    if encoder_settings['Name'] == "embedding":
        return RelationEmbedding(Embedding(encoder_settings), encoder_settings)
    if encoder_settings['Name'] == "gated_basis_gcn":
        graph = Representation(triples, encoder_settings)

        first_layer = GatedBasisGcn(encoder_settings, graph)
        second_layer = GatedBasisGcn(encoder_settings, graph, next_component=first_layer)
        transform = LinearTransform(second_layer, encoder_settings)
        return RelationEmbedding(transform, encoder_settings)
    else:
        return None

def build_decoder(encoder, decoder_settings):
    if decoder_settings['Name'] == "bilinear-diag":
        return BilinearDiag(encoder, decoder_settings)
    else:
        return None