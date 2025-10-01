import dgl
from dgl.nn.pytorch import RelGraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F


# reduce dimensions by Autoencoder
class TextEmbeddingAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, dropout_rate=0.2):
        super(TextEmbeddingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim * 2),
            nn.BatchNorm1d(encoding_dim * 2),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(encoding_dim * 2, encoding_dim),
            nn.BatchNorm1d(encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim * 2),
            nn.BatchNorm1d(encoding_dim * 2),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(encoding_dim * 2, input_dim),
            nn.ReLU(True)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class BaseRGCN(nn.Module):
    """
    Base class for Relational Graph Convolutional Network (R-GCN) model.
    This class initializes the model and defines the base layers.
    """

    def __init__(
        self,
        num_nodes,
        hidden_dim,
        output_dim,
        num_relations,
        num_bases=-1,
        num_hidden_layers=1,
        dropout=0.0,
        use_self_loop=False,
        use_cuda=False,
        pretrained_text_embeddings=None,
        pretrained_domain_embeddings=None,
        pretrained_global_embeddings=None,
        freeze=False,
        w_text=0.5, w_domain=0.5, w_global=0.0
    ):
        super(BaseRGCN, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_relations = num_relations
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.use_cuda = use_cuda

        # pretrained embeddings
        self.pretrained_text_embeddings = pretrained_text_embeddings
        self.pretrained_domain_embeddings = pretrained_domain_embeddings
        self.pretrained_global_embeddings = pretrained_global_embeddings

        self.freeze = freeze

        # weights (sum to 1 in EmbeddingLayer at runtime)
        self.w_text = w_text
        self.w_domain = w_domain
        self.w_global = w_global

        # Create RGCN layers
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        # Input to hidden layer
        input_layer = self.build_input_layer()
        if input_layer is not None:
            self.layers.append(input_layer)
        # Hidden to hidden layers
        for idx in range(self.num_hidden_layers):
            hidden_layer = self.build_hidden_layer(idx)
            self.layers.append(hidden_layer)
        # Hidden to output layer (if necessary)
        output_layer = self.build_output_layer()
        if output_layer is not None:
            self.layers.append(output_layer)

    def build_input_layer(self):
        # Override in subclass
        return None

    def build_hidden_layer(self, idx):
        # Override in subclass
        raise NotImplementedError

    def build_output_layer(self):
        # Override in subclass
        return None

    def forward(self, graph, node_ids, rel_ids, norm):
        """
        Forward pass through the RGCN layers.
        """
        x = node_ids
        for layer in self.layers:
            x = layer(graph, x, rel_ids, norm)
        return x


class EmbeddingLayer(nn.Module):
    """
    Embedding layer to initialize node features with up to three pretrained embeddings
    (text / domain / global). Each is normalized then projected to hidden, and fused by weights.
    """

    def __init__(
        self,
        num_nodes,
        hidden_dim,
        pretrained_text_embeddings,
        pretrained_domain_embeddings,
        pretrained_global_embeddings=None,
        freeze=False,
        w_text=0.5, w_domain=0.5, w_global=0.0
    ):
        super(EmbeddingLayer, self).__init__()
        self.w_text = w_text
        self.w_domain = w_domain
        self.w_global = w_global

        eps = 1e-8

        # ----- Domain (graph) embeddings -----
        if pretrained_domain_embeddings is not None:
            domain_np = torch.from_numpy(pretrained_domain_embeddings).float()
            # Min-max normalize
            denom = (domain_np.max() - domain_np.min()) + eps
            norm_domain = (domain_np - domain_np.min()) / denom
            self.norm_domain_embeddings = nn.Embedding.from_pretrained(norm_domain, freeze=freeze)
            self.poincare_to_euclidean = nn.Linear(pretrained_domain_embeddings.shape[1], hidden_dim)
            print(f"Loaded pretrained domain embeddings, freeze={freeze}.")
        else:
            self.norm_domain_embeddings = nn.Embedding(num_nodes, hidden_dim)
            self.poincare_to_euclidean = nn.Linear(hidden_dim, hidden_dim)
            print("Initialized random domain embeddings.")

        # ----- Text embeddings (Autoencoder) -----
        if pretrained_text_embeddings is not None:
            text_np = torch.from_numpy(pretrained_text_embeddings).float()
            denom = (text_np.max() - text_np.min()) + eps
            norm_text = (text_np - text_np.min()) / denom
            self.norm_text_embeddings = nn.Embedding.from_pretrained(norm_text, freeze=freeze)
            self.autoencoder = TextEmbeddingAutoencoder(pretrained_text_embeddings.shape[1], hidden_dim)
            print(f"Loaded pretrained text embeddings, freeze={freeze}.")
        else:
            self.norm_text_embeddings = nn.Embedding(num_nodes, hidden_dim)
            self.autoencoder = TextEmbeddingAutoencoder(hidden_dim, hidden_dim)
            print("Initialized random text embeddings.")

        # ----- Global embeddings (community/global) -----
        if pretrained_global_embeddings is not None:
            global_np = torch.from_numpy(pretrained_global_embeddings).float()
            denom = (global_np.max() - global_np.min()) + eps
            norm_global = (global_np - global_np.min()) / denom
            self.norm_global_embeddings = nn.Embedding.from_pretrained(norm_global, freeze=freeze)
            self.global_to_hidden = nn.Linear(pretrained_global_embeddings.shape[1], hidden_dim)
            print(f"Loaded pretrained global embeddings, freeze={freeze}.")
        else:
            self.norm_global_embeddings = None
            self.global_to_hidden = None
            print("No pretrained global embeddings provided.")

    def forward(self, graph, node_ids, rel_ids, norm):
        # Text → hidden via autoencoder
        text_raw = self.norm_text_embeddings(node_ids.squeeze())
        transformed_text, _ = self.autoencoder(text_raw)

        # Domain → hidden via linear (Poincaré → Euclidean)
        domain_raw = self.norm_domain_embeddings(node_ids.squeeze())
        transformed_domain = self.poincare_to_euclidean(domain_raw)

        # Global → hidden via linear if available; else zeros
        if (self.norm_global_embeddings is not None) and (self.global_to_hidden is not None):
            global_raw = self.norm_global_embeddings(node_ids.squeeze())
            transformed_global = self.global_to_hidden(global_raw)
        else:
            transformed_global = torch.zeros_like(transformed_domain)

        # Normalize weights to sum 1
        s = self.w_text + self.w_domain + self.w_global
        if s <= 0:
            wt, wd, wg = 1.0, 0.0, 0.0
        else:
            wt, wd, wg = self.w_text / s, self.w_domain / s, self.w_global / s

        combined = wd * transformed_domain + wt * transformed_text + wg * transformed_global
        return combined


class RGCN(BaseRGCN):
    """
    Implementation of R-GCN with support for link prediction.
    """

    def build_input_layer(self):
        # Initialize node features with embedding layer
        return EmbeddingLayer(
            self.num_nodes,
            self.hidden_dim,
            self.pretrained_text_embeddings,
            self.pretrained_domain_embeddings,
            pretrained_global_embeddings=self.pretrained_global_embeddings,
            freeze=self.freeze,
            w_text=self.w_text, w_domain=self.w_domain, w_global=self.w_global
        )

    def build_hidden_layer(self, idx):
        # Activation function for all but the last layer
        activation = F.relu if idx < self.num_hidden_layers - 1 else None
        return RelGraphConv(
            in_feat=self.hidden_dim,
            out_feat=self.hidden_dim,
            num_rels=self.num_relations,
            regularizer='bdd',
            num_bases=self.num_bases,
            activation=activation,
            self_loop=self.use_self_loop,
            dropout=self.dropout
        )


class LinkPredict(nn.Module):
    """
    Link prediction model using R-GCN.
    """

    def __init__(
        self, input_dim, hidden_dim, num_relations, num_bases=-1,
        num_hidden_layers=1, dropout=0.0, use_cuda=False, regularization_param=0.0,
        pretrained_text_embeddings=None, pretrained_domain_embeddings=None,
        pretrained_relation_embeddings=None, pretrained_global_embeddings=None,
        freeze=False, w_text=0.5, w_domain=0.5, w_global=0.0
    ):
        super(LinkPredict, self).__init__()
        self.rgcn = RGCN(
            input_dim, hidden_dim, hidden_dim, num_relations * 2, num_bases,
            num_hidden_layers, dropout, use_cuda,
            pretrained_text_embeddings=pretrained_text_embeddings,
            pretrained_domain_embeddings=pretrained_domain_embeddings,
            pretrained_global_embeddings=pretrained_global_embeddings,
            freeze=freeze, w_text=w_text, w_domain=w_domain, w_global=w_global
        )
        self.regularization_param = regularization_param

        if pretrained_relation_embeddings is not None:
            self.relation_weights = nn.Parameter(torch.Tensor(pretrained_relation_embeddings))
            normalized_relations = (self.relation_weights - self.relation_weights.min()) / (
                (self.relation_weights.max() - self.relation_weights.min()) + 1e-8
            )
            self.relation_weights.data.copy_(normalized_relations)
            print("Loaded pretrained relation embeddings.")
        else:
            self.relation_weights = nn.Parameter(torch.Tensor(num_relations, hidden_dim))
            nn.init.xavier_uniform_(self.relation_weights, gain=nn.init.calculate_gain('relu'))
            print("Initialized random relation embeddings.")

    def calculate_score(self, embeddings, triplets):
        """
        Calculate the score for triplets using DistMult.
        """
        subject_embeddings = embeddings[triplets[:, 0]]
        relation_embeddings = self.relation_weights[triplets[:, 1]]
        object_embeddings = embeddings[triplets[:, 2]]
        score = torch.sum(subject_embeddings * relation_embeddings * object_embeddings, dim=1)
        return score

    def forward(self, graph, node_ids, rel_ids, norm):
        return self.rgcn(graph, node_ids, rel_ids, norm)

    def regularization_loss(self, embeddings):
        """
        Compute regularization loss for embeddings and relation weights.
        """
        return torch.mean(embeddings.pow(2)) + torch.mean(self.relation_weights.pow(2))

    def get_loss(self, graph, embeddings, triplets, labels):
        """
        Compute loss for link prediction, including regularization loss.
        """
        score = self.calculate_score(embeddings, triplets)
        prediction_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embeddings)
        return prediction_loss + self.regularization_param * reg_loss

    def set_fusion_weights(self, w_text, w_domain, w_global):
        self.w_text = float(w_text)
        self.w_domain = float(w_domain)
        self.w_global = float(w_global)