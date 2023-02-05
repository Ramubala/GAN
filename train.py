import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import time
import seaborn as sns
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

class Trainer:
    def __init__(self, model) -> None:
        self.model = model
        with open( 'parameters.yml', 'r') as file:
            self.parameters = yaml.safe_load(file)

    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs


    def train(self):
        
        # load parameters
        n_epochs = self.parameters['train']['n_epochs']
        batches_per_epoch = self.parameters['train']['batches_per_epoch']
        
        noise_mean = self.parameters['train']['noise_mean']
        noise_cov =  self.parameters['train']['noise_cov']
        target_mean = self.parameters['train']['target_mean']
        target_cov = self.parameters['train']['target_cov']
        batch_size = self.parameters['train']['batch_size']
        discriminator_steps = self.parameters['train']['discriminator_steps_per_gen_step']

        # noise_dataloader = DataLoader(input_data, batch_size, True)
        # target_dataloader = DataLoader(target_data, batch_size, True)

        criterion = nn.BCELoss()
        optimizer_gen = optim.Adam(self.model.generator.parameters())
        optimizer_disc = optim.Adam(self.model.discriminator.parameters())

        generator_loss, discriminator_loss = [], [] 

        fig, ax = plt.subplots(6,5, figsize=(20,24), sharex=True, sharey=True)
        for epoch in range(n_epochs):
            start_time = time.time()
            epoch_disc_loss, epoch_gen_loss = 0, 0
            for j in range(batches_per_epoch):

                noise_samples = np.random.multivariate_normal(noise_mean, noise_cov, batch_size)
                true_samples =  np.random.multivariate_normal(target_mean, target_cov, batch_size)

                true_sample_predictions = self.model.discriminator(torch.from_numpy(true_samples))
                noise_sample_predictions = self.model(torch.from_numpy(noise_samples))
                disc_preds = torch.vstack((true_sample_predictions, noise_sample_predictions)).squeeze(1)
                targets = torch.vstack((torch.full((batch_size,1), 1, dtype=torch.float64), torch.full((batch_size,1), 0, dtype=torch.float64))).squeeze(1)
                loss = criterion(disc_preds, targets)
                epoch_disc_loss += loss.item()
                optimizer_disc.zero_grad()
                loss.backward()
                optimizer_disc.step()

                # update generator
                if j%discriminator_steps==0:
                    noise_samples = np.random.multivariate_normal(noise_mean, noise_cov, batch_size)
                    noise_sample_predictions = self.model(torch.from_numpy(noise_samples)).squeeze(1)
                    targets = torch.full((batch_size,1), 1, dtype=torch.float64).squeeze(1)
                    loss = criterion(noise_sample_predictions, targets)
                    epoch_gen_loss += loss.item()
                    optimizer_gen.zero_grad()
                    loss.backward()
                    optimizer_gen.step()

            epoch_disc_loss = epoch_disc_loss/batches_per_epoch
            epoch_gen_loss = epoch_gen_loss*5/batches_per_epoch
            generator_loss.append((epoch, epoch_gen_loss))
            discriminator_loss.append((epoch, epoch_disc_loss))    
            end_time = time.time()
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)
            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s | Loss: Disc: {epoch_disc_loss} Gen: {epoch_gen_loss}')

            test_samples = np.random.multivariate_normal(noise_mean, noise_cov, 10000)
            generated = self.model.generator(torch.from_numpy(test_samples)).detach().numpy()
            sns.scatterplot(x = generated[:, 0], y = generated[:, 1], color='black', ax = ax[epoch//5][epoch%5]).set_title('Epoch {}'.format(epoch+1))
            sns.scatterplot(x = [noise_mean[0], target_mean[0]],y =  [noise_mean[1], target_mean[1]], color='g', ax = ax[epoch//5][epoch%5], markers=['o', 's'], s=35)
            
        fig.savefig('generated_distribution.png')

        fig,ax = plt.subplots(1,2,figsize=(18,5))
        ax[0].plot(*zip(*generator_loss))
        ax[0].set_title('Generator loss')
        ax[1].plot(*zip(*discriminator_loss))
        ax[1].set_title('Discriminator loss')
        
        fig.savefig('Training Loss.png')
