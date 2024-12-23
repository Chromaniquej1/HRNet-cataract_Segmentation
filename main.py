IM_SIZE = 512
N_CLASSES = 5
N_BANDS = 3
BATCH_SIZE = 4
N_EPOCHS = 50
weights_path = '/kaggle/working/models/hrnet_attention_best_model.keras'
logpath = '/kaggle/working/cataract_multiclass_model.csv'

tr_imdir = '/kaggle/input/cataract-data/train/images'
tr_maskdir = '/kaggle/input/cataract-data/train/masks'
val_imdir = '/kaggle/input/cataract-data/val/images'
val_maskdir = '/kaggle/input/cataract-data/val/masks'

input_shape = (IM_SIZE, IM_SIZE, N_BANDS)
