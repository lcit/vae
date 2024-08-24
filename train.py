import torch
import torchvision
import numpy as np
import os
from vae import VAE

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MNISTDataset(torch.utils.data.Dataset):
    
    def __init__(self, ):
        self.mnist = torchvision.datasets.MNIST(".", download=True)
        
    def __len__(self):
        return len(self.mnist)
    
    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        img_np = np.float32(img)[None]/255.0
        img_t = torch.from_numpy(img_np)
        label_t = torch.LongTensor([label])
        return img_t, label_t

if __name__=="__main__":
    
    dataset = torch.utils.data.DataLoader(MNISTDataset(), 
                                          batch_size=4, 
                                          pin_memory=True, 
                                          shuffle=True,
                                          num_workers=8)
    
    model = VAE().to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), 
    #                              eps=1e-08, weight_decay=1e-5)
    
    lr_scheduler = None
    
    iteration = 0
    for epoch in range(10):
        
        for images, labels in dataset:
            
            optimizer.zero_grad()
            
            images = images.to(device)
            labels = labels.to(device)
            
            out = model(images)
            loss = out['loss']
            
            if iteration%100==0:
                print(f"Iteration {iteration}")
                for name, value in out.items():
                    print(f"\t{name}: {value}")
            
            loss.backward()
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()            
            
            iteration += 1
            
            name = "model"
            if iteration%10000==0:
                torch.save(model.state_dict(), 
                           os.path.join(".", f"{name}_{iteration:06d}.pickle"))
