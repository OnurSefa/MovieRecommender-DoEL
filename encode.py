import json
import torch


class Encoder:
    def __init__(self, movie_data_dir, aux_path, device):
        with open(aux_path, 'r') as f:
            aux_data = json.load(f)
        self.movie_mapping = aux_data['movie_indices']
        self.movie_count = aux_data['movie_count']
        self.year_count = aux_data['year_count']
        self.tag_count = aux_data['tag_count']
        self.genre_count = aux_data['genre_count']
        self.mdr = movie_data_dir
        self.d = device

    def encode(self, line):
        past_split = line.split("<past>")
        future_split = past_split[1].split("<future>")
        future_mask_split = future_split[1].split("<future_mask>")
        past_mask_split = future_mask_split[1].split("<past_mask>")

        past = future_split[0]
        future = future_mask_split[0]
        future_mask = past_mask_split[0]
        past_mask = past_mask_split[1]

        past_data = torch.zeros((10, self.movie_count+self.year_count+self.genre_count+2*self.tag_count), device=self.d)
        past_rating_data = torch.zeros((10, 3), device=self.d)

        past_info = past.split('-')
        for i in range(-1, -11, -1):
            past_rating = int(past_info[i*2+1])
            past_rating_data[-i-1, past_rating] = 1

            past_movie = int(past_info[i*2])
            with open(f'{self.mdr}/{past_movie}.json', 'r') as f:
                past_movie_data = json.load(f)
            past_movie_index = self.movie_mapping.index(past_movie)
            past_data[-i-1, past_movie_index] = 1
            year = past_movie_data['year_index']
            past_data[-i-1, self.movie_count + year] = 1
            for genre in past_movie_data['genre_indices']:
                past_data[-i-1, self.movie_count + self.year_count + genre] = 1
            for tag_index, tag_count in zip(past_movie_data['tag_indices'], past_movie_data['tag_counts']):
                past_data[-i-1, self.movie_count + self.year_count + self.genre_count + tag_index] = 1
                past_data[-i-1, self.movie_count + self.year_count + self.genre_count + self.tag_count + tag_index] = tag_count

        label = torch.zeros(self.movie_count, device=self.d)
        for future_movie_id in future.split('-'):
            label[self.movie_mapping.index(int(future_movie_id))] = 1

        mask = torch.ones(self.movie_count, device=self.d)
        for mask_id in future_mask.split('-'):
            try:
                mask[self.movie_mapping.index(int(mask_id))] = 0
            except:
                pass
        for mask_id in past_mask.split('-'):
            try:
                mask[self.movie_mapping.index(int(mask_id))] = 0
            except:
                pass
        return past_data, past_rating_data, label, mask


if __name__ == '__main__':
    encoder = Encoder(movie_data_dir="../movie_recommender_data/movies", aux_path='../movie_recommender_data/aux_data.json', device=torch.device('cpu'))
    with open('../movie_recommender_data/dataset/16_0.txt', 'r') as f:
        datum = f.readline()
    encoder.encode(datum)


