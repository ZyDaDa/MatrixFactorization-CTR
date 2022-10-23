from torch import nn
import math

class MF(nn.Module):

    def __init__(self, args, user_num, item_num) -> None:
        super().__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dim = args.dim

        self.user_emb = nn.Embedding(user_num, self.dim)
        self.item_emb = nn.Embedding(item_num, self.dim)

        self.loss_function = nn.MSELoss()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1 / math.sqrt(self.dim)
        for weight in self.parameters():
            nn.init.normal_(weight.data,0,stdv)

    def forward(self, data):

        user_emb = self.user_emb(data['user'])
        item_emb = self.item_emb(data['item'])

        return (user_emb * item_emb).sum(-1)