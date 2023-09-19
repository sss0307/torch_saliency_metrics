'''
Evaluation metrics

Saliency metrics for evaluating saliency prediction:
1. Normalized Scanpath Saliency(NSS), 
2. Pearson's Correlation Coefficient(PCC), 
3. KL-Divergence(KL), 
4. Information Ganin(IG),
5. Similarity Measure(SIM)
'''

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

        
    def all_metrics(self):
        ''' This function returns a tuple of all metrics in the following order : nss, pcc, ig, sim, kld'''
        nss_sum = 0
        pcc_sum = 0
        ig_sum = 0
        sim_sum = 0
        kld_sum = 0

        for i,(smap, gen_smap) in enumerate(zip(self.smap_val_dataloader, self.gen_smap_val_dataloader)):
            discrete_smap = torch.tensor(smap > 0.5, dtype=torch.float)
            discrete_smap_sum = discrete_smap.sum()
            smap_unity = self.normalize_image(smap, method='unity')
            gen_smap_unity = self.normalize_image(gen_smap, method='unity')
            gen_smap_gaussian = self.normalize_image(gen_smap, method='gaussian')

            # NSS
            nss_value = discrete_smap * gen_smap_gaussian
            nss_sum += nss_value.sum() / discrete_smap_sum

            # PCC
            pcc_sum += self.pearson_correlation(smap, gen_smap).item()

            # IG
            ig_value = (torch.log2(self.eps + gen_smap) - torch.log2(self.eps + self.baseline_smap)) * discrete_smap
            ig_sum += ig_value.sum() / discrete_smap_sum

            # SIM
            sim_value = torch.min(smap_unity, gen_smap_unity)
            sim_sum += sim_value.sum()

            # KLD
            kld_value = (torch.log(self.eps + (smap/(self.eps + gen_smap)))) * smap
            kld_sum += kld_value.sum()

        nss = nss_sum / self.num_of_images
        pcc = pcc_sum / self.num_of_images
        ig = ig_sum / self.num_of_images
        sim = sim_sum / self.num_of_images
        kld = kld_sum / self.num_of_images

        return (nss, pcc, ig, sim, kld)
