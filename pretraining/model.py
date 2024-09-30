import torch.nn as nn
import torch
import open_clip
class Vision_adapter(nn.Module):
    def __init__(self):
        super(Vision_adapter, self).__init__()
        #self.model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.conv = nn.Conv1d(in_channels=2, out_channels=768, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU()
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.residual_layer = nn.Conv1d(in_channels=2, out_channels=768, kernel_size=1, stride=1)

        #self.adapt_layer_vis = nn.Sequential(nn.Linear(768, 384),
        #                                     nn.LeakyReLU(),
        #                                     nn.Linear(384, 768),
        #                                     nn.LeakyReLU())
        #self.classify = nn.Linear(768, 12)

    def forward(self, x):
        #image_features = model.encode_image(x)
        bsz = x.shape[0]
        residual_connnection = self.residual_layer(x)

        x = self.conv(x)  # Apply 1D convolution
        x = self.relu(x)  # Apply ReLU activation

        x = x + residual_connnection
        x = self.global_pooling(x)  # Global mean pooling
        x = x.view(bsz, -1)  # Flatten to [BSZ, output_dim]
        return x
        '''
        emb_li = []
        for idx in range(x.shape[0]):
            embed = self.enc_transformers.encode_image(x[idx])
            emb_li.append(embed)
        emb_li = torch.stack(emb_li, dim=0)
        '''
        #x_org = self.shortcut(image_features)
        #output = self.adapt_layer_vis(x_org) + x_org
        #output = self.classify(output)

        #return output

if __name__ == '__main__':
    model = Vision_adapter()
    test = torch.randn((128, 512))
    o = model(test)
    print(model.state_dict().keys())



