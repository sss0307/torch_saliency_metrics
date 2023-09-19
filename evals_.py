'''
Evaluation metrics

Saliency metrics for evaluating saliency prediction:
1. Normalized Scanpath Saliency(NSS), 
2. Pearson's Correlation Coefficient(PCC), 
3. KL-Divergence(KL), 
4. Information Ganin(IG),
5. Similarity Measure(SIM)
'''

# Requirements
import torch
import os

class Evaluation_metrics():

    def __init__(self, smap_val_dataloader, gen_smap_val_dataloader):
        self.smap_val_dataloader = smap_val_dataloader
        self.gen_smap_val_dataloader = gen_smap_val_dataloader
        self.image_size = 256
        self.eps = 2.22e-16
        self.num_of_images = len(self.smap_val_dataloader.dataset) 
        
        # Center Bias Baseline Saliency Map
        larger_matrix = torch.zeros(256,256)
        smaller_matrix = torch.ones(64,64)*0.9
        row_start = (larger_matrix.size(0) - smaller_matrix.size(0)) // 2 # Specify the position to add the smaller matrix in the center of the larger matrix
        col_start = (larger_matrix.size(1) - smaller_matrix.size(1)) // 2
        larger_matrix[row_start:row_start+smaller_matrix.size(0), col_start:col_start+smaller_matrix.size(1)] += smaller_matrix # Add the smaller matrix to the center of the larger matrix
        self.baseline_smap = larger_matrix.unsqueeze(dim=0)
    
    def normalize_image(self, image, method='gaussian'):
        smap = image.detach().clone()

        if method=='gaussian':
            mean = smap.mean(dim=(1, 2), keepdim=True)
            std = smap.std(dim=(1, 2), keepdim=True)

            normalized_smap = (smap - mean) / std

        if method=='minmax':
            normalized_smap = (smap - smap.min()) / (smap.max() - smap.min())

        if method=='unity':
            normalized_smap = smap / smap.sum()
        
        return normalized_smap

    def pearson_correlation(self, tensor1, tensor2):
        cov = ( (tensor1 - tensor1.mean()) * (tensor2 - tensor2.mean()) ).mean()    # Covariance   
        correlation = cov / ( tensor1.std() * tensor2.std() )   # Correlation Coefficient
        return correlation

    def nss(self):
        sum_value = 0

        for i,(smap, gen_smap) in enumerate(zip(self.smap_val_dataloader, self.gen_smap_val_dataloader)):
            discrete_smap = torch.tensor(smap > 0.5, dtype=torch.float)
            normalized_gen_smap = self.normalize_image(gen_smap, method='gaussian')

            value = discrete_smap * normalized_gen_smap
            sum_value += value.sum() / discrete_smap.sum()

        nss = sum_value / self.num_of_images
        return nss

    def pcc(self):
        sum_value = 0

        for i,(smap, gen_smap) in enumerate(zip(self.smap_val_dataloader, self.gen_smap_val_dataloader)):
            sum_value += self.pearson_correlation(smap, gen_smap).item()
        
        pcc = sum_value / self.num_of_images
        return pcc

    def ig(self):
        sum_value = 0

        for i,(smap, gen_smap) in enumerate(zip(self.smap_val_dataloader, self.gen_smap_val_dataloader)):
            discrete_smap = torch.tensor(smap > 0.5, dtype=torch.float)

            value = (torch.log2(self.eps + gen_smap) - torch.log2(self.eps + self.baseline_smap)) * discrete_smap
            sum_value += value.sum() / discrete_smap.sum()

        ig = sum_value / self.num_of_images
        return ig

    def sim(self):
        sum_value = 0

        for i,(smap, gen_smap) in enumerate(zip(self.smap_val_dataloader, self.gen_smap_val_dataloader)):
            smap_unity = self.normalize_image(smap, method='unity')
            gen_smap_unity = self.normalize_image(gen_smap, method='unity')

            value = torch.min(smap_unity, gen_smap_unity)
            sum_value += value.sum()

        sim = sum_value / self.num_of_images
        return sim
    
    def kld(self):
        sum_value = 0

        for i,(smap, gen_smap) in enumerate(zip(self.smap_val_dataloader, self.gen_smap_val_dataloader)):
            value = (torch.log(self.eps + (smap/(self.eps + gen_smap)))) * smap
            sum_value += value.sum()

        kld = sum_value / self.num_of_images
        return kld 

def aw():
    print('aw shucks')
