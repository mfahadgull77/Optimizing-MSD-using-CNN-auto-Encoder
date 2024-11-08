import os
import time
import numpy as np
from skimage import io, img_as_float, transform as sk_transform, restoration
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.optim as optim
from skimage.metrics import peak_signal_noise_ratio as psnr, mean_squared_error, normalized_mutual_information as mutual_info
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Define the dataset paths
data_dir = "Dataset/images 600nm/color-o2/"
output_dir = "Dataset/images 600nm/denoised/"
os.makedirs(output_dir, exist_ok=True)

# Define the Denoising Autoencoder model
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # (B, 64, H/2, W/2)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (B, 128, H/4, W/4)
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # (B, 256, H/8, W/8)
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 128, H/4, W/4)
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 64, H/2, W/2)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 3, H, W)
            nn.Sigmoid()  # to ensure the output values are in [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Instantiate the model
model = DenoisingAutoencoder()

# Custom Dataset class for images
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform
        print(f"Dataset initialized with {len(self.image_files)} samples.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        try:
            image = io.imread(img_name)
            image = img_as_float(image)
            image = sk_transform.resize(image, (256, 256), anti_aliasing=True)

            if self.transform:
                image = self.transform(image)

            return image, img_name  # Return image and filename
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            dummy_image = np.zeros((256, 256, 3), dtype=np.float32)
            if self.transform:
                dummy_image = self.transform(dummy_image)
            return dummy_image, img_name

# Paths and parameters
batch_size = 4
num_epochs = 10
learning_rate = 1e-3

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Dataset and DataLoader
dataset = ImageDataset(data_dir, transform=transform)
print(f"Number of samples in dataset: {len(dataset)}")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Lists to store training metrics
train_loss_history = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in dataloader:
        inputs = inputs.float()
        targets = targets.float()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataset)
    train_loss_history.append(epoch_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# Plot the training loss
plt.plot(range(1, num_epochs+1), train_loss_history, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.show()

# Save the model
torch.save(model.state_dict(), 'denoising_autoencoder.pth')
print("Model saved!")

# Denoising and saving images
model.eval()
psnr_list_autoencoder = []
psnr_list_nlm = []
rmse_list_autoencoder = []
rmse_list_nlm = []
mutual_info_list_autoencoder = []
mutual_info_list_nlm = []
correlation_list_autoencoder = []
correlation_list_nlm = []
registration_times_autoencoder = []
registration_times_nlm = []

def process_image(image, img_name, idx):
    print(f"Processing image {idx} with shape: {image.shape}")
    image = image.unsqueeze(0).float()  # Ensure the tensor is of type float

    # Start timing for autoencoder
    start_time = time.time()
    with torch.no_grad():
        denoised_image_autoencoder = model(image)
    end_time = time.time()
    registration_time_autoencoder = end_time - start_time
    registration_times_autoencoder.append(registration_time_autoencoder)

    denoised_image_autoencoder = denoised_image_autoencoder.squeeze(0).numpy().transpose(1, 2, 0)

    # Ensure denoised image has the same shape as original image
    denoised_image_autoencoder = denoised_image_autoencoder.transpose(2, 0, 1)

    # Convert denoised image to the same data type as original image
    image_dtype = image.dtype
    denoised_image_autoencoder = torch.tensor(denoised_image_autoencoder, dtype=image_dtype)

    # Save denoised image
    denoised_image_autoencoder_np = denoised_image_autoencoder.numpy().transpose(1, 2, 0)
    output_path = os.path.join(output_dir, f"denoised_autoencoder_{os.path.basename(img_name)}")
    io.imsave(output_path, denoised_image_autoencoder_np)

    # Apply Non-Local Means Denoising and time it
    image_np = image.squeeze(0).numpy().transpose(1, 2, 0)  # Convert image to NumPy array
    start_time_nlm = time.time()
    denoised_image_nlm = restoration.denoise_nl_means(image_np, h=1.15 * np.std(image_np), fast_mode=True)
    end_time_nlm = time.time()
    registration_time_nlm = end_time_nlm - start_time_nlm
    registration_times_nlm.append(registration_time_nlm)

    # Save NLM denoised image
    output_path_nlm = os.path.join(output_dir, f"denoised_nlm_{os.path.basename(img_name)}")
    io.imsave(output_path_nlm, denoised_image_nlm)

    # Calculate metrics for Autoencoder
    psnr_value_autoencoder = psnr(image_np, denoised_image_autoencoder.numpy().transpose(1, 2, 0))
    rmse_value_autoencoder = mean_squared_error(image_np, denoised_image_autoencoder.numpy().transpose(1, 2, 0))
    mutual_info_value_autoencoder = mutual_info(image_np.flatten(), denoised_image_autoencoder.numpy().transpose(1, 2, 0).flatten())
    correlation_value_autoencoder = pearsonr(image_np.flatten(), denoised_image_autoencoder.numpy().transpose(1, 2, 0).flatten())[0]

    # Calculate metrics for NLM
    psnr_value_nlm = psnr(image_np, denoised_image_nlm)
    rmse_value_nlm = mean_squared_error(image_np, denoised_image_nlm)
    mutual_info_value_nlm = mutual_info(image_np.flatten(), denoised_image_nlm.flatten())
    correlation_value_nlm = pearsonr(image_np.flatten(), denoised_image_nlm.flatten())[0]

    return (psnr_value_autoencoder, rmse_value_autoencoder, mutual_info_value_autoencoder, correlation_value_autoencoder,
            psnr_value_nlm, rmse_value_nlm, mutual_info_value_nlm, correlation_value_nlm)

# Parallel processing
results = Parallel(n_jobs=-1)(delayed(process_image)(dataset[idx][0], dataset[idx][1], idx) for idx in range(len(dataset)))

# Unpack results
for res in results:
    psnr_value_autoencoder, rmse_value_autoencoder, mutual_info_value_autoencoder, correlation_value_autoencoder, \
    psnr_value_nlm, rmse_value_nlm, mutual_info_value_nlm, correlation_value_nlm = res

    psnr_list_autoencoder.append(psnr_value_autoencoder)
    rmse_list_autoencoder.append(rmse_value_autoencoder)
    mutual_info_list_autoencoder.append(mutual_info_value_autoencoder)
    correlation_list_autoencoder.append(correlation_value_autoencoder)

    psnr_list_nlm.append(psnr_value_nlm)
    rmse_list_nlm.append(rmse_value_nlm)
    mutual_info_list_nlm.append(mutual_info_value_nlm)
    correlation_list_nlm.append(correlation_value_nlm)

# Print average metrics and registration time
print(f'Average PSNR (Autoencoder): {np.mean(psnr_list_autoencoder):.4f}')
print(f'Average RMSE (Autoencoder): {np.mean(rmse_list_autoencoder):.4f}')
print(f'Average Mutual Information (Autoencoder): {np.mean(mutual_info_list_autoencoder):.4f}')
print(f'Average Correlation Coefficient (Autoencoder): {np.mean(correlation_list_autoencoder):.4f}')
print(f'Average Registration Time (Autoencoder): {np.mean(registration_times_autoencoder):.4f} seconds')

print(f'Average PSNR (NLM): {np.mean(psnr_list_nlm):.4f}')
print(f'Average RMSE (NLM): {np.mean(rmse_list_nlm):.4f}')
print(f'Average Mutual Information (NLM): {np.mean(mutual_info_list_nlm):.4f}')
print(f'Average Correlation Coefficient (NLM): {np.mean(correlation_list_nlm):.4f}')
print(f'Average Registration Time (NLM): {np.mean(registration_times_nlm):.4f} seconds')

print("Denoised images processed and saved!")
