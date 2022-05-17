import torch.nn as nn
import torch
import torch.nn.functional as F

def kl_divergence(p, q, temperature=1.0, probs=False):
    if probs is False:
        p = F.softmax(p / temperature, dim=-1)
        q = F.softmax(q / temperature, dim=-1)
    return torch.sum(p * (torch.log(p+1e-8) - torch.log(q+1e-8)), dim=-1).mean()

class ClassificationLoss(nn.Module):
    def __init__(self):
        super(ClassificationLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, output, target):
        if len(output.shape) == 2:
            output = output.unsqueeze(1)
        loss = 0.0
        N = output.shape[1]
        for i in range(N):
            loss += self.ce(output[:, i, :], target)
        return loss / N
    
class ClassificationStudentTeacherLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = ClassificationLoss()
        
    def forward(self, teacher_output, student_output, target, temperature_mean=3.0, temperature_individual=1.0):
        if len(student_output.shape) == 2:
            student_output = student_output.unsqueeze(1)

        T = teacher_output.shape[1]
        S = student_output.shape[1]
        M = max(T, S)

        student_loss = self.ce(student_output, target)
        teacher_loss_individual = 0.0
        for i in range(M):
            if T==S:
                teacher_output_i = teacher_output[:, i, :]
                student_output_i = student_output[:, i, :]
            elif T>S:
                teacher_output_i = teacher_output[:, i, :]
                student_output_i = student_output[:, i%S, :]
            else:
                teacher_output_i = teacher_output[:, i%T, :]
                student_output_i = student_output[:, i, :]
            # Compute the kl divergence between the teacher and the student
            teacher_loss_individual += kl_divergence(teacher_output_i, student_output_i, temperature_individual, probs=False)
        teacher_loss_individual*=(temperature_individual**2)/M

        teacher_loss_mean = 0.0
        student_mean = F.softmax(student_output / temperature_mean, dim=2).mean(dim=1)
        teacher_mean = F.softmax(teacher_output / temperature_mean, dim=2).mean(dim=1)
        teacher_loss_mean = kl_divergence(teacher_mean, student_mean, temperature_mean, probs=True)
        teacher_loss_mean*=temperature_mean**2
        
        return student_loss, teacher_loss_mean, teacher_loss_individual

# Taken and adapted from here: https://github.com/lennelov/endd-reproduce
class ClassificationEnDDLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, teacher_output, student_output, target, temperature_mean=3.0, temperature_individual=1.0):
        # Note that target has actually never been used 
        student_output = student_output.squeeze(1)
        alphas = torch.exp(student_output / temperature_individual)
        precision = torch.sum(alphas, dim=1)  #sum over classes
        teacher_output = F.softmax(teacher_output/temperature_individual, dim=2)

        log_ensemble_probs_geo_mean = torch.mean(torch.log(teacher_output + 1e-8), dim=1)
        target_independent_term = torch.sum(torch.lgamma(alphas + 1e-8), dim=1) - torch.lgamma(precision + 1e-8)  
        target_dependent_term = -torch.sum((alphas - 1.) * log_ensemble_probs_geo_mean, dim=1)

        cost = target_dependent_term + target_independent_term
        return torch.mean(cost,dim=0) * (temperature_individual**2), 0.0, 0.0
    
class RegressionLoss(nn.Module):
    def __init__(self):
        super(RegressionLoss, self).__init__()
        self.loss = nn.GaussianNLLLoss()

    def forward(self, output, target):
        if len(output.shape) == 2:
            output = output.unsqueeze(1)
        N = output.shape[1]
        loss = 0.0
        for i in range(N):
            mean, var = output[:, i, 0], output[:, i, 1].exp()
            loss += self.loss(mean, target.squeeze(), var)
        return loss / N

class RegressionStudentTeacherLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = RegressionLoss()
        
    def forward(self, teacher_output, student_output, target, temperature_mean=1.0, temperature_individual=1.0):
        # Note that the temperature is actually not used here 
        if len(student_output.shape) == 2:
            student_output = student_output.unsqueeze(1)

        T = teacher_output.shape[1]
        S = student_output.shape[1]
        M = max(T, S)

        student_loss = self.loss(student_output, target)
        teacher_loss_individual = 0.0
        for i in range(M):
            if T==S:
                teacher_output_i = teacher_output[:, i, :]
                student_output_i = student_output[:, i, :]
            elif T>S:
                teacher_output_i = teacher_output[:, i, :]
                student_output_i = student_output[:, i%S, :]
            else:
                teacher_output_i = teacher_output[:, i%T, :]
                student_output_i = student_output[:, i, :]

            teacher_mean_i = teacher_output_i[:, 0]
            teacher_var_i = teacher_output_i[:, 1].exp()+1e-8
            student_mean_i = student_output_i[:, 0]
            student_var_i = student_output_i[:, 1].exp()+1e-8
            # Compute the kl divergence between the teacher and the student
            teacher_loss_individual += (0.5*((teacher_var_i + (teacher_mean_i - student_mean_i)**2) / (student_var_i + 1e-8) + torch.log(student_var_i + 1e-8))).mean()
        teacher_loss_individual/=M

        teacher_loss_mean = 0.0
        student_mean = student_output[:,:,0].mean(dim=1)
        student_var = student_output[:,:,1].exp().mean(dim=1)+student_output[:,:,0].var(dim=1)+1e-8 
        teacher_mean = teacher_output[:,:,0].mean(dim=1)
        teacher_var = teacher_output[:,:,1].exp().mean(dim=1)+teacher_output[:,:,0].var(dim=1)+1e-8
        teacher_loss_mean = (0.5*((teacher_var + (teacher_mean - student_mean)**2) / (student_var + 1e-8) + torch.log(student_var + 1e-8))).mean()

        return student_loss, teacher_loss_mean, teacher_loss_individual

def losses_factory(task, method):
    if task == 'classification':
        if method == "ensemble":
            return ClassificationLoss()
        elif method in ["hydra+", "drop", "gauss", "hydra"]:
            return ClassificationStudentTeacherLoss()
        elif method == "endd":
            return ClassificationEnDDLoss()
        else:
            return ClassificationLoss()
    elif task == 'regression':
        if method == "ensemble":
            return RegressionLoss()
        elif method in ["hydra+", "drop", "gauss", "hydra"]:
            return RegressionStudentTeacherLoss()
        else:
            return RegressionLoss()
    else:
        raise ValueError('Unknown task {}'.format(task))