import torch.nn as nn
import torch

repo = 'epic-kitchens/action-models'
class_counts = (125, 352)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


def get_tsn_model(base_model="resnet50", segment_count=8, tune_model=True):
    tsn = torch.hub.load(repo, 'TSN', class_counts, segment_count, 'RGB',
                     base_model=base_model,
                     pretrained='epic-kitchens', force_reload=False)
    if tune_model:
        ### transfer learning.. freezing weights and adding a new FC layer
        for param in tsn.parameters():
            param.requires_grade = False
        in_features = tsn.fc_verb.in_features
        tsn.fc_verb = nn.Linear(in_features, out_features=3, bias=True)
        tsn.fc_noun = Identity()
        # sanity check ensuring newly FC is trainable
        for param in tsn.fc_verb.parameters():
            param.requires_grad = True

        """
        for name, param in tsn.named_parameters():
            print(name, ":", param.requires_grad)

        """
        return tsn
    else:
        return tsn


def get_trn_model(base_model="resnet50", segment_count=8, tune_model=False):
    trn = torch.hub.load(repo, 'TRN', class_counts, segment_count, 'RGB',
                     base_model=base_model,
                     pretrained='epic-kitchens')
    if tune_model:
        raise Exception("fine tuning not supported on trn model yet")
    return trn



