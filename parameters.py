class Parameters():
    color_space='RGB'
    spatial_size=(32, 32)
    hist_bins=8
    orient=9
    pix_per_cell=8
    cell_per_block=2
    hog_channel=0
    hist_range = (0, 256)
    spatial_feat=True
    hist_feat=True
    hog_feat=True
    def __init__(self, color_space='RGB', spatial_size=(32, 32),
                 hist_bins=8, orient=9, 
                 pix_per_cell=8, cell_per_block=2, hog_channel=0, scale = 1.5,hist_range = (0, 256),
                 spatial_feat=True, hist_feat=True, hog_feat=True):
        # HOG parameters
        self.color_space = color_space
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hog_channel = hog_channel
        self.scale = scale
        self.spatial_feat = spatial_feat
        self.hist_feat = hist_feat
        self.hog_feat = hog_feat
        self.hist_range = hist_range