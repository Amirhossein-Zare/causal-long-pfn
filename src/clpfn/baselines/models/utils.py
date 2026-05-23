import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F


def grad_reverse(x, scale=1.0):

    class ReverseGrad(Function):
        """
        Gradient reversal layer.
        """

        @staticmethod
        def forward(ctx, x):
            return x

        @staticmethod
        def backward(ctx, grad_output):
            return scale * grad_output.neg()

    return ReverseGrad.apply(x)


def bce(treatment_pred, current_treatments, mode, weights=None):
    if mode == "multiclass":
        return F.cross_entropy(
            treatment_pred.permute(0, 2, 1),
            current_treatments.permute(0, 2, 1),
            reduction="none",
            weight=weights,
        )

    if mode == "multilabel":
        return F.binary_cross_entropy_with_logits(
            treatment_pred,
            current_treatments,
            reduction="none",
            weight=weights,
        ).mean(dim=-1)

    raise NotImplementedError()


class OutcomeHead(nn.Module):
    """Used by GT."""

    def __init__(self, seq_hidden_units, hr_size, fc_hidden_units, dim_treatments, dim_outcome):
        super().__init__()

        self.seq_hidden_units = seq_hidden_units
        self.hr_size = hr_size
        self.fc_hidden_units = fc_hidden_units
        self.dim_treatments = dim_treatments
        self.dim_outcome = dim_outcome

        self.linear1 = nn.Linear(self.hr_size + self.dim_treatments, self.fc_hidden_units)
        self.elu = nn.ELU()
        self.linear2 = nn.Linear(self.fc_hidden_units, self.dim_outcome)
        self.trainable_params = ["linear1", "linear2"]

    def build_outcome(self, hr, current_treatment):
        x = torch.cat((hr, current_treatment), dim=-1)
        x = self.elu(self.linear1(x))
        outcome = self.linear2(x)
        return outcome


class BRTreatmentOutcomeHead(nn.Module):
    """Used by CRN, EDCT, MultiInputTransformer."""

    def __init__(
        self,
        seq_hidden_units,
        br_size,
        fc_hidden_units,
        dim_treatments,
        dim_outcome,
        alpha=0.0,
        update_alpha=True,
        balancing="grad_reverse",
    ):
        super().__init__()

        self.seq_hidden_units = seq_hidden_units
        self.br_size = br_size
        self.fc_hidden_units = fc_hidden_units
        self.dim_treatments = dim_treatments
        self.dim_outcome = dim_outcome
        self.alpha = alpha if not update_alpha else 0.0
        self.alpha_max = alpha
        self.balancing = balancing

        self.linear1 = nn.Linear(self.seq_hidden_units, self.br_size)
        self.elu1 = nn.ELU()

        self.linear2 = nn.Linear(self.br_size, self.fc_hidden_units)
        self.elu2 = nn.ELU()
        self.linear3 = nn.Linear(self.fc_hidden_units, self.dim_treatments)

        self.linear4 = nn.Linear(self.br_size + self.dim_treatments, self.fc_hidden_units)
        self.elu3 = nn.ELU()
        self.linear5 = nn.Linear(self.fc_hidden_units, self.dim_outcome)

        self.treatment_head_params = ["linear2", "linear3"]

    def build_treatment(self, br, detached=False):
        if detached:
            br = br.detach()

        if self.balancing == "grad_reverse":
            br = grad_reverse(br, self.alpha)

        br = self.elu2(self.linear2(br))
        treatment = self.linear3(br)
        return treatment

    def build_outcome(self, br, current_treatment):
        x = torch.cat((br, current_treatment), dim=-1)
        x = self.elu3(self.linear4(x))
        outcome = self.linear5(x)
        return outcome

    def build_br(self, seq_output):
        br = self.elu1(self.linear1(seq_output))
        return br


class ROutcomeVitalsHead(nn.Module):
    """Used by G-Net."""

    def __init__(self, seq_hidden_units, r_size, fc_hidden_units, dim_outcome, dim_vitals, num_comp, comp_sizes):
        super().__init__()

        self.seq_hidden_units = seq_hidden_units
        self.r_size = r_size
        self.fc_hidden_units = fc_hidden_units
        self.dim_outcome = dim_outcome
        self.dim_vitals = dim_vitals
        self.num_comp = num_comp
        self.comp_sizes = comp_sizes

        self.linear1 = nn.Linear(self.seq_hidden_units, self.r_size)
        self.elu1 = nn.ELU()

        self.cond_nets = []
        add_input_dim = 0

        for comp in range(self.num_comp):
            linear2 = nn.Linear(self.r_size + add_input_dim, self.fc_hidden_units)
            elu2 = nn.ELU()
            linear3 = nn.Linear(self.fc_hidden_units, self.comp_sizes[comp])
            self.cond_nets.append(nn.Sequential(linear2, elu2, linear3))

            add_input_dim += self.comp_sizes[comp]

        self.cond_nets = nn.ModuleList(self.cond_nets)

    def build_r(self, seq_output):
        r = self.elu1(self.linear1(seq_output))
        return r

    def build_outcome_vitals(self, r):
        vitals_outcome_pred = []

        for cond_net in self.cond_nets:
            out = cond_net(r)
            r = torch.cat((out, r), dim=-1)
            vitals_outcome_pred.append(out)

        return torch.cat(vitals_outcome_pred, dim=-1)

