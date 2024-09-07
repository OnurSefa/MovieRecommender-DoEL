from loader import MovieDataset
from torch.utils.data import DataLoader
import torch
from model import Model
import torch.optim as o
import torch.nn as nn
import mlflow
import torch.nn.functional as F


def masked_bce_loss(predictions, targets, mask):
    # Apply sigmoid to predictions
    predictions = torch.sigmoid(predictions)

    # Calculate BCE loss
    bce_loss = F.binary_cross_entropy(predictions, targets, reduction='none')

    # Apply mask
    masked_loss = bce_loss * mask

    # Return mean of masked loss
    return masked_loss.sum() / mask.sum()


def train(model, optimizer, loss_function, epoch_count, model_dir, model_prefix, save_interval, batch_size, names_path, movie_data_dir, aux_path, shuffle=True):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print('device is mps')
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print('device is cuda')
    else:
        device = torch.device('cpu')
        print('device is cpu')

    model = model.to(device)
    model.train()

    dataset = MovieDataset(names_path, movie_data_dir, aux_path, device, mode='train')
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    save_loss = 0
    epoch_loss = 0
    save_iteration = 0
    epoch_iteration = 0
    step = 0

    for epoch in range(epoch_count):
        for i, data in enumerate(data_loader):
            past_data, past_rating_data, labels, masks = data
            optimizer.zero_grad()
            predictions = model(past_data, past_rating_data)
            loss = loss_function(predictions, labels, masks)
            loss.backward()
            optimizer.step()
            print(f'{epoch} - {i} - loss: {loss.item():.8f}')

            mlflow.log_metric('batch loss', loss.item(), step=step)

            save_loss += loss.item()
            epoch_loss += loss.item()
            save_iteration += 1
            epoch_iteration += 1
            if i % save_interval == 0 and i != 0:
                torch.save(model, f'{model_dir}/{model_prefix}_{epoch}_{i}.pth')
                mlflow.log_metric('save loss', save_loss / save_iteration, step=step)
                print(f"SAVE LOSS {epoch} - {i}: {save_loss/save_iteration}")
                save_loss = 0
                save_iteration = 0
                print()
            step += 1
        mlflow.log_metric('epoch loss', epoch_loss/epoch_iteration, step=step)
        epoch_loss = 0
        epoch_iteration = 0


if __name__ == '__main__':
    lf = masked_bce_loss
    m = Model()
    ec = 10
    md = '../movie_recommender_data/models'
    si = 300
    bs = 16

    learning_rate = 0.0002
    optim = o.Adam(m.parameters(), lr=learning_rate)
    mp = "003"

    mlflow.start_run(run_name=mp)
    mlflow.log_param('learning rate', learning_rate)
    mlflow.log_param('batch size', bs)
    mlflow.log_param('loss function', 'masked bce loss')
    mlflow.log_param('model name', mp)
    mlflow.log_param('model initiated', 'scratch')
    mlflow.log_param('notes', 'initial run')

    train(m, optim, lf, ec, md, mp, si, bs, '../movie_recommender_data/train_names.json', "../movie_recommender_data/movies", '../movie_recommender_data/aux_data.json', True)

    mlflow.end_run()